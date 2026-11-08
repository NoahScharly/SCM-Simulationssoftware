"""Laden, Zusammenführen und Prüfen der Konfiguration (Modul data).

Kennzeichnung: KI (Claude Opus 5.5), Prompts siehe KI_NACHWEIS.md.

Ablauf:
1. Stammdaten aus CSV (Knoten, Kanten, Routen, Komponenten, Bezugsquellen)
2. Parameter aus base.yaml
3. Szenario-Datei: überschreibt einzelne Parameter und liefert Störungen
4. Overrides aus der Oberfläche (Regler)
5. Prüfung von Referenzen und Plausibilitätsregeln, Berechnung der Mindest-WBZ
"""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
from typing import Any

import yaml

from .domain import EU_LAENDER, Bezugsquelle, Kante, Knoten, Komponente, Modell, Route, Stoerung

KONFIG_STANDARD = Path(__file__).resolve().parent.parent / "config"


class KonfigurationsFehler(ValueError):
    """Konfiguration ist unvollständig oder verletzt eine harte Plausibilitätsregel."""


# --------------------------------------------------------------------------- Hilfen
def _lies_csv(pfad: Path) -> list[dict[str, str]]:
    with open(pfad, encoding="utf-8") as f:
        return [{k: (v or "").strip() for k, v in zeile.items()} for zeile in csv.DictReader(f)]


def tief_zusammenfuehren(basis: dict, aenderung: dict) -> dict:
    """Rekursives Überschreiben: Werte aus `aenderung` ersetzen die aus `basis`."""
    ergebnis = copy.deepcopy(basis)
    for schluessel, wert in (aenderung or {}).items():
        if isinstance(wert, dict) and isinstance(ergebnis.get(schluessel), dict):
            ergebnis[schluessel] = tief_zusammenfuehren(ergebnis[schluessel], wert)
        else:
            ergebnis[schluessel] = copy.deepcopy(wert)
    return ergebnis


def dreieck_mittel(werte: list[float]) -> float:
    a, b, c = werte
    return (a + b + c) / 3.0


def _wegpunkte(text: str) -> tuple[tuple[float, float], ...]:
    if not text:
        return ()
    return tuple(tuple(float(x) for x in p.split(",")) for p in text.split(";"))  # type: ignore[misc]


# --------------------------------------------------------------------------- Stammdaten
def _lade_stammdaten(ordner: Path):
    knoten = {
        z["id"]: Knoten(z["id"], z["name"], z["typ"], z["stufe"], z["land"], float(z["lat"]),
                        float(z["lon"]), z["zollgrenze"] == "1")
        for z in _lies_csv(ordner / "nodes.csv")
    }
    kanten = {
        z["id"]: Kante(z["id"], z["von"], z["nach"], z["modus"], float(z["km"]),
                       float(z["abfahrt_intervall_tage"]), float(z["co2e_g_tkm"]), z["aktiv"] == "1",
                       z["engpass"], _wegpunkte(z["wegpunkte"]))
        for z in _lies_csv(ordner / "edges.csv")
    }
    routen: dict[str, list[Route]] = {}
    for z in _lies_csv(ordner / "routes.csv"):
        r = Route(z["lieferant"], z["route_id"], int(z["prioritaet"]), z["typ"], tuple(z["kanten"].split(";")))
        routen.setdefault(r.lieferant, []).append(r)
    for liste in routen.values():
        liste.sort(key=lambda r: r.prioritaet)
    komponenten = {
        z["komponente"]: Komponente(z["komponente"], z["bezeichnung"], float(z["gewicht_g"]), float(z["zoll_satz"]))
        for z in _lies_csv(ordner / "components.csv")
    }
    quellen: dict[str, list[Bezugsquelle]] = {}
    for z in _lies_csv(ordner / "suppliers.csv"):
        q = Bezugsquelle(
            komponente=z["komponente"], lieferant=z["lieferant"], rolle=z["rolle"],
            preis_eur=float(z["preis_eur"]), fertigung_tage=float(z["fertigung_tage"]),
            wbz_tage=float(z["wbz_tage"]) if z["wbz_tage"] else None,
            tier2=tuple(t for t in z["tier2"].split(";") if t),
        )
        quellen.setdefault(q.komponente, []).append(q)
    for liste in quellen.values():
        liste.sort(key=lambda q: 0 if q.rolle == "primaer" else 1)
    return knoten, kanten, routen, komponenten, quellen


# --------------------------------------------------------------------------- Szenario
def _lade_szenario(szenario: str | Path | dict | None) -> dict:
    if szenario is None:
        return {"id": "S0", "name": "Basis", "stoerungen": []}
    if isinstance(szenario, dict):
        return szenario
    with open(szenario, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _baue_stoerungen(roh: list[dict], einschwingen: float) -> list[Stoerung]:
    ergebnis = []
    for i, s in enumerate(roh or []):
        typ = s.get("typ")
        if typ not in {"knoten_ausfall", "kanten_sperre", "korridor", "nachfrage", "kosten"}:
            raise KonfigurationsFehler(f"Störung {i}: unbekannter Typ '{typ}'")
        start = einschwingen + float(s.get("start_tag", 0))
        dauer = s.get("dauer_tage")
        ende = start + float(dauer) if dauer is not None else math.inf
        ergebnis.append(Stoerung(
            typ=typ, start=start, ende=ende, ziel=str(s.get("ziel", "")),
            faktor=float(s.get("faktor", 1.0)), monate=tuple(s.get("monate", ()) or ()),
            aktivieren=tuple(s.get("aktivieren", ()) or ()), deaktivieren=tuple(s.get("deaktivieren", ()) or ()),
            parameter=str(s.get("parameter", "")), wert=float(s.get("wert", 0.0)),
        ))
    return ergebnis


# --------------------------------------------------------------------------- Zeiten
def erwartete_transitzeit(modell: Modell, route: Route, lieferant_land: str) -> float:
    """Erwartete Transitzeit einer Route in Tagen: Warten auf Abfahrt, Fahrt, Umschlag, Zoll."""
    tp = modell.param["transport"]
    tage = 0.0
    for i, kid in enumerate(route.kanten):
        k = modell.kanten[kid]
        tage += k.abfahrt_intervall_tage / 2.0
        tage += k.km / tp["geschwindigkeit_km_tag"][k.modus]
        if i < len(route.kanten) - 1:
            tage += dreieck_mittel(tp["umschlag_tage"][k.modus])
    if lieferant_land not in EU_LAENDER:
        tage += dreieck_mittel(tp["zoll_tage"])
    return tage


def beste_regulaere_route(modell: Modell, lieferant: str) -> Route:
    """Erste reguläre Route nach Priorität, deren Kanten im Basisfall aktiv sind."""
    for r in modell.routen.get(lieferant, []):
        if r.typ == "regulaer" and all(modell.kanten[k].aktiv for k in r.kanten):
            return r
    raise KonfigurationsFehler(f"Lieferant {lieferant}: keine aktive reguläre Route")


# --------------------------------------------------------------------------- Prüfung
def _pruefe_referenzen(m: Modell) -> None:
    for k in m.kanten.values():
        for n in (k.von, k.nach):
            if n not in m.knoten:
                raise KonfigurationsFehler(f"Kante {k.id}: unbekannter Knoten '{n}'")
        if k.modus not in m.param["transport"]["geschwindigkeit_km_tag"]:
            raise KonfigurationsFehler(f"Kante {k.id}: unbekannter Verkehrsträger '{k.modus}'")
    for lieferant, liste in m.routen.items():
        for r in liste:
            for kid in r.kanten:
                if kid not in m.kanten:
                    raise KonfigurationsFehler(f"Route {r.id}: unbekannte Kante '{kid}'")
            # Kanten müssen eine zusammenhängende Kette vom Lieferanten zum Werk bilden
            pos = lieferant
            for kid in r.kanten:
                if m.kanten[kid].von != pos:
                    raise KonfigurationsFehler(f"Route {r.id}: Kante {kid} beginnt nicht bei {pos}")
                pos = m.kanten[kid].nach
            if pos != m.werk:
                raise KonfigurationsFehler(f"Route {r.id}: endet bei {pos} statt beim Werk")
    for komp in m.komponenten:
        if komp not in m.quellen or not any(q.rolle == "primaer" for q in m.quellen[komp]):
            raise KonfigurationsFehler(f"Komponente {komp}: kein primärer Lieferant")
    for liste in m.quellen.values():
        for q in liste:
            if q.lieferant not in m.knoten:
                raise KonfigurationsFehler(f"Bezugsquelle {q.komponente}: unbekannter Lieferant {q.lieferant}")
            for t2 in q.tier2:
                if t2 not in m.knoten:
                    raise KonfigurationsFehler(f"Bezugsquelle {q.komponente}: unbekannter Tier-2 {t2}")
    for s in m.stoerungen:
        if s.typ == "knoten_ausfall" and s.ziel not in m.knoten:
            raise KonfigurationsFehler(f"Störung: unbekannter Knoten '{s.ziel}'")
        if s.typ == "kanten_sperre" and s.ziel not in m.kanten:
            raise KonfigurationsFehler(f"Störung: unbekannte Kante '{s.ziel}'")
        for kid in s.aktivieren + s.deaktivieren:
            if kid not in m.kanten:
                raise KonfigurationsFehler(f"Störung: unbekannte Kante '{kid}'")


def _pruefe_plausibilitaet(m: Modell) -> None:
    """Plausibilitätsregeln aus Folie 2: harte Grenzen werfen Fehler, weiche erzeugen Warnungen."""
    g = m.param["grenzen"]
    menge = float(m.param["nachfrage"]["uhren_pro_jahr"])
    kap = float(m.param["montage"]["kapazitaet_pro_tag"])
    lo, hi = g["uhren_pro_jahr"]
    if not lo <= menge <= hi:
        raise KonfigurationsFehler(f"Uhren pro Jahr {menge:.0f} außerhalb von {lo} bis {hi}")
    lo, hi = g["kapazitaet_pro_tag"]
    if not lo <= kap <= hi:
        raise KonfigurationsFehler(f"Montagekapazität {kap:.0f}/Tag außerhalb von {lo} bis {hi}")
    gut = float(m.param["montage"]["gutquote"])
    jahreskapazitaet = kap * g["arbeitstage_pro_jahr"] * gut
    if jahreskapazitaet < menge:
        m.warnungen.append(
            f"Montagekapazität reicht nicht: {kap:.0f}/Tag × {g['arbeitstage_pro_jahr']} AT × {str(gut).replace(".", ",")} "
            f"= {jahreskapazitaet:,.0f} < {menge:,.0f} Uhren. Mindestens "
            f"{math.ceil(menge / (g['arbeitstage_pro_jahr'] * gut))}/Tag nötig.".replace(",", ".")
        )
    # WBZ: nie unter Fertigung + Transit + Zoll der regulären Route
    for liste in m.quellen.values():
        for q in liste:
            land = m.knoten[q.lieferant].land
            route = beste_regulaere_route(m, q.lieferant)
            q.wbz_min_tage = q.fertigung_tage + erwartete_transitzeit(m, route, land)
            if q.wbz_tage is None:
                q.wbz_zuschlag_tage = 0.0
            elif q.wbz_tage < q.wbz_min_tage - 1e-9:
                raise KonfigurationsFehler(
                    f"WBZ {q.komponente} bei {q.lieferant}: {q.wbz_tage:.0f} Tage liegt unter dem physischen "
                    f"Minimum von {q.wbz_min_tage:.1f} Tagen (Fertigung + Transit + Zoll)")
            elif q.wbz_tage > g["wbz_max_tage"]:
                raise KonfigurationsFehler(
                    f"WBZ {q.komponente}: {q.wbz_tage:.0f} Tage über dem Maximum von {g['wbz_max_tage']} Tagen")
            else:
                q.wbz_zuschlag_tage = q.wbz_tage - q.wbz_min_tage
    laengste = max(q.wbz_min_tage + q.wbz_zuschlag_tage for l in m.quellen.values() for q in l)
    if m.einschwingen < laengste:
        m.warnungen.append(
            f"Einschwingphase {m.einschwingen:.0f} Tage ist kürzer als die längste WBZ ({laengste:.0f} Tage)")


# --------------------------------------------------------------------------- Hauptfunktion
def lade_modell(
    szenario: str | Path | dict | None = None,
    overrides: dict[str, Any] | None = None,
    konfig_ordner: str | Path = KONFIG_STANDARD,
) -> Modell:
    """Lädt Basis, Szenario und Overrides und liefert ein geprüftes Modell.

    overrides: verschachteltes dict wie base.yaml; zusätzlich {"wbz": {komponente: tage}}
    für die gewünschte Wiederbeschaffungszeit des primären Lieferanten.
    """
    ordner = Path(konfig_ordner)
    knoten, kanten, routen, komponenten, quellen = _lade_stammdaten(ordner)
    with open(ordner / "base.yaml", encoding="utf-8") as f:
        param = yaml.safe_load(f)

    sz = _lade_szenario(szenario)
    param = tief_zusammenfuehren(param, sz.get("aenderungen", {}))
    overrides = copy.deepcopy(overrides or {})
    wbz_wunsch = overrides.pop("wbz", {}) or {}
    param = tief_zusammenfuehren(param, overrides)

    for komp, tage in wbz_wunsch.items():
        if komp not in quellen:
            raise KonfigurationsFehler(f"WBZ für unbekannte Komponente '{komp}'")
        for q in quellen[komp]:
            if q.rolle == "primaer":
                q.wbz_tage = None if tage is None else float(tage)

    einschwingen = float(param["simulation"]["einschwingen_tage"])
    modell = Modell(
        knoten=knoten, kanten=kanten, routen=routen, komponenten=komponenten, quellen=quellen,
        param=param, szenario_id=str(sz.get("id", "S?")), szenario_name=str(sz.get("name", "")),
        stoerungen=_baue_stoerungen(sz.get("stoerungen", []), einschwingen),
    )
    _pruefe_referenzen(modell)
    _pruefe_plausibilitaet(modell)
    return modell


def szenario_dateien(konfig_ordner: str | Path = KONFIG_STANDARD) -> list[Path]:
    """Szenario-Dateien, Basis (s0_*) zuerst."""
    dateien = (Path(konfig_ordner) / "scenarios").glob("*.yaml")
    return sorted(dateien, key=lambda p: (not p.name.startswith("s0"), p.name))
