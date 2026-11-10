"""Handlungsoptionen je Szenario automatisch vergleichen und eine Empfehlung ableiten (Modul handlungsoptionen).

Kennzeichnung: G (Ablauf und Fachlogik N. Scharly, Code mit Claude Opus 5.5 umgesetzt), siehe KI_NACHWEIS.md.

Ablauf in der Oberfläche:
    1. Normalbetrieb (S0) und die gewählten Störungsszenarien werden festgelegt.
    2. Für jedes Szenario wird jede Handlungsoption aus base.yaml (Block handlungsoptionen) simuliert,
       alle mit denselben Zufallszahlen (Common Random Numbers).
    3. Je Szenario: Vergleich gegen "Ohne Maßnahme", Vor- und Nachteile, Empfehlung nach Priorität.
    4. Je Option: Break-even-Wahrscheinlichkeit p* (lohnt sich die Vorsorge, wenn die Störung mit p eintritt?).
    5. Bandbreite: Stärke der Störung variieren und prüfen, ab wann die Empfehlung kippt.

Empfehlungslogik (transparent, keine Blackbox):
    Stufe 0: harte Ausschlusskriterien (z. B. Sanktionsländer). Wer einen ausgeschlossenen Standort nutzt, wird
             nicht gerankt (Modul netzanalyse).
    Stufe 1: Mindestanforderungen (Standard: Lieferfähigkeit >= 95 %, Lieferzeit 95 % <= 3 Tage, einstellbar).
    Stufe 2: Priorisierung durch Reihenfolge der Zielkriterien, daraus Rank-Order-Centroid-Gewichte.
    Erfüllt keine Option alle Anforderungen, wird die mit dem kleinsten Abstand zu den Anforderungen empfohlen.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from .data import KONFIG_STANDARD, lade_modell
from .evaluation import RICHTUNG, pareto_front
from .experiment import _setze_pfad, fuehre_aus
from .netzanalyse import genutzte_knoten

REFERENZ = "keine"
KENNZAHLEN = ("beta", "kosten_pro_uhr", "co2e_pro_uhr_kg", "lieferzeit_p95", "fehlteil_tage",
              "luftfracht_sendungen", "kapital_gebunden_eur", "kapitalbindung_pro_uhr", "bereitschaft_pro_uhr",
              "aushilfe_tage")
# Schwellen, ab denen eine Differenz als Vor- oder Nachteil gilt (darunter: Rauschen bzw. keine Wirkung)
SCHWELLEN = {"beta": 0.005, "kosten_pro_uhr": 0.10, "co2e_pro_uhr_kg": 0.0005, "lieferzeit_p95": 0.5,
             "fehlteil_tage": 0.5, "kapital_gebunden_eur": 1000.0}


def lade_entscheidungsregel(konfig_ordner: str | Path = KONFIG_STANDARD) -> tuple[dict[str, float], list[str]]:
    """Standard-Anforderungen und -Priorisierung aus base.yaml (Block entscheidung)."""
    e = lade_modell(None, None, konfig_ordner).param.get("entscheidung") or {}
    anf = {k: float(v) for k, v in (e.get("anforderungen") or {"beta": 0.95, "lieferzeit_p95": 3}).items()}
    prio = list(e.get("prioritaet") or ["beta", "kosten_pro_uhr", "co2e_pro_uhr_kg", "lieferzeit_p95"])
    return anf, [k for k in prio if k in KRITERIEN]


def lade_optionen(konfig_ordner: str | Path = KONFIG_STANDARD) -> list[dict[str, Any]]:
    """Handlungsoptionen aus base.yaml. Die Referenz "keine" steht immer zuerst."""
    roh = lade_modell(None, None, konfig_ordner).param.get("handlungsoptionen") or []
    optionen = [dict(o) for o in roh]
    if not any(o.get("id") == REFERENZ for o in optionen):
        optionen.insert(0, {"id": REFERENZ, "name": "Ohne Maßnahme", "kurz": "Ohne Maßnahme", "massnahmen": {}})
    optionen.sort(key=lambda o: 0 if o["id"] == REFERENZ else 1)
    for o in optionen:
        o.setdefault("kurz", o.get("name", o["id"]))
        o.setdefault("massnahmen", {})
    return optionen


def _overrides_mit(overrides: dict[str, Any] | None, option: dict[str, Any]) -> dict[str, Any]:
    ov = copy.deepcopy(overrides or {})
    ov["massnahmen"] = copy.deepcopy(option.get("massnahmen") or {})
    return ov


def _optionen_fuer(szenario: str | Path | dict, optionen: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Nur die für das Szenario relevanten Optionen (Szenariodatei: relevante_optionen) plus die Referenz.

    Ohne Eintrag werden alle Optionen gerechnet. So wächst die Rechenzeit nicht mit jeder neuen Bezugsquelle
    für jedes Szenario, sondern nur dort, wo die Option eine Rolle spielt.
    """
    if isinstance(szenario, dict):
        roh = szenario
    else:
        with open(szenario, encoding="utf-8") as f:
            roh = yaml.safe_load(f) or {}
    gewuenscht = list(roh.get("relevante_optionen") or [])
    if not gewuenscht:
        return optionen
    return [o for o in optionen if o["id"] == REFERENZ or o["id"] in gewuenscht]


def vergleiche_optionen(
    szenarien: list[str | Path],
    optionen: list[dict[str, Any]] | None = None,
    overrides: dict[str, Any] | None = None,
    replikationen: int | None = None,
    fortschritt: Callable[[int, int], None] | None = None,
    nur_relevante: bool = True,
) -> dict[str, Any]:
    """Simuliert jede Option auf jedem Szenario (mit nur_relevante nur die in der Szenariodatei genannten).

    Liefert:
        detail:      eine Zeile je (Szenario, Option) mit Mittelwerten, KI und gepaarten Differenzen zu "Ohne Maßnahme"
        ergebnisse:  {(szenario_id, option_id): Rückgabe von fuehre_aus} für Karte und Diagramme
        optionen:    die verwendeten Optionen
    """
    optionen = optionen or lade_optionen()
    je_szenario = [_optionen_fuer(pfad, optionen) if nur_relevante else optionen for pfad in szenarien]
    gesamt, i = sum(len(o) for o in je_szenario), 0
    ergebnisse: dict[tuple[str, str], dict[str, Any]] = {}
    for pfad, opts in zip(szenarien, je_szenario):
        for opt in opts:
            e = fuehre_aus(pfad, _overrides_mit(overrides, opt), replikationen)
            ergebnisse[(e["szenario_id"], opt["id"])] = e
            i += 1
            if fortschritt:
                fortschritt(i, gesamt)

    zeilen = []
    for (sid, oid), e in ergebnisse.items():
        z = e["zusammenfassung"]
        ref = ergebnisse.get((sid, REFERENZ))
        zeile: dict[str, Any] = {"szenario": sid, "szenario_name": e["szenario_name"], "option": oid,
                                 "knoten_genutzt": ";".join(sorted(genutzte_knoten(e["beispiellauf"])))}
        for kz in z.index:
            zeile[kz] = float(z.loc[kz, "mittel"])
            zeile[f"{kz}_ki95"] = float(z.loc[kz, "ki95"])
        for kz in KENNZAHLEN:
            if ref is None or kz not in e["laeufe"]:
                zeile[f"delta_{kz}"] = math.nan
                continue
            d = (e["laeufe"][kz] - ref["laeufe"][kz]).dropna()   # gepaart: gleiche Replikationsnummern
            zeile[f"delta_{kz}"] = float(d.mean()) if not d.empty else math.nan
        zeilen.append(zeile)
    return {"detail": pd.DataFrame(zeilen), "ergebnisse": ergebnisse, "optionen": optionen}


# Zielkriterien, aus denen in der Oberfläche gewählt und priorisiert wird: Name, Einheit, Richtung (+1 = mehr ist besser)
KRITERIEN: dict[str, tuple[str, str, int]] = {
    "beta": ("Lieferfähigkeit", "%", +1),
    "kosten_pro_uhr": ("Kosten je Uhr", "€", -1),
    "co2e_pro_uhr_kg": ("CO2e je Uhr", "g", -1),
    "lieferzeit_p95": ("Lieferzeit (95 %)", "Tage", -1),
    "fehlteil_tage": ("Montagestillstand", "Tage", -1),
    "kapital_gebunden_eur": ("Gebundenes Kapital", "€", -1),
}


def richtung(kz: str) -> int:
    return KRITERIEN[kz][2] if kz in KRITERIEN else RICHTUNG.get(kz, -1)


def roc_gewichte(n: int) -> list[float]:
    """Rank-Order-Centroid-Gewichte (Barron und Barrett 1996): aus einer reinen Reihenfolge abgeleitete Gewichte.

    w_i = 1/n * Summe_{k=i..n} 1/k. Beispiel n = 4: 52 %, 27 %, 15 %, 6 %.
    """
    return [sum(1 / k for k in range(i, n + 1)) / n for i in range(1, n + 1)]


def rangiere(df: pd.DataFrame, anforderungen: dict[str, float], prioritaet: list[str],
             spalten: dict[str, str] | None = None) -> pd.DataFrame:
    """Zweistufige, offengelegte Rangfolge. Rang 1 = Empfehlung.

    Stufe 1 (Mindestanforderungen): zulässig ist eine Option, wenn sie jede Anforderung erfüllt
        (z. B. Lieferfähigkeit >= 95 %, Lieferzeit 95 % <= 3 Tage). spalten erlaubt, eine Anforderung gegen eine
        andere Spalte zu prüfen (robust: schlechtestes Szenario statt Mittelwert).
    Stufe 2 (Priorisierung): die Reihenfolge der Kriterien ergibt ROC-Gewichte. Jedes Kriterium wird über die
        zulässigen Optionen auf 0 (bester Wert) bis 1 (schlechtester Wert) normiert, der gewichtete Abstand zum
        Besten ist die Punktzahl (kleiner = besser).
    Unzulässige Optionen stehen dahinter, sortiert nach ihrem relativen Abstand zu den Anforderungen.
    """
    spalten = spalten or {}
    out = df.copy().reset_index(drop=True)
    verletzt = [[] for _ in range(len(out))]
    abstand = pd.Series(0.0, index=out.index)
    # Stufe 0: harte Ausschlusskriterien (Spalte "ausschluss" = Text der verletzten Regel, leer = zulässig)
    if "ausschluss" in out:
        out["ausgeschlossen"] = out["ausschluss"].fillna("").astype(str).str.len() > 0
    else:
        out["ausgeschlossen"] = False
    for kz, grenze in (anforderungen or {}).items():
        sp = spalten.get(kz, kz)
        werte = out[sp].astype(float)
        fehl = (grenze - werte) if richtung(kz) > 0 else (werte - grenze)
        fehl = fehl.clip(lower=0).where(fehl > 1e-9, 0.0)
        abstand += fehl / (abs(grenze) or 1.0)
        for i in out.index[fehl > 0]:
            verletzt[i].append(KRITERIEN.get(kz, (kz,))[0])
    out["zulaessig"] = (abstand <= 0) & ~out["ausgeschlossen"]
    for i in out.index[out["ausgeschlossen"]]:
        verletzt[i].insert(0, "Ausschlusskriterium")
    out["verletzt"] = [", ".join(v) for v in verletzt]
    out["abstand_anforderung"] = abstand

    gewichte = dict(zip(prioritaet, roc_gewichte(len(prioritaet))))
    basis = out[out.zulaessig] if out.zulaessig.any() else out[~out.ausgeschlossen] if (~out.ausgeschlossen).any() else out
    punkte = pd.Series(0.0, index=out.index)
    for kz, w in gewichte.items():
        x = out[kz].astype(float)
        bx = basis[kz].astype(float)
        bester, schlechtester = (bx.max(), bx.min()) if richtung(kz) > 0 else (bx.min(), bx.max())
        # Mindestspanne = Schwelle, ab der ein Unterschied als echt gilt. Sonst würde z. B. ein Unterschied von
        # 0,03 Prozentpunkten Lieferfähigkeit (Zufallsrauschen) auf die volle Spanne 0 bis 1 gestreckt.
        spanne = max(abs(schlechtester - bester), SCHWELLEN.get(kz, 0.0))
        norm = ((x - bester).abs() / spanne) if spanne > 1e-12 else pd.Series(0.0, index=out.index)
        out[f"norm_{kz}"] = norm
        punkte += w * norm
    out["punkte"] = punkte
    out["pareto"] = pareto_front(out, list(gewichte)) if gewichte else False
    out = out.sort_values(["ausgeschlossen", "zulaessig", "abstand_anforderung", "punkte"],
                          ascending=[True, False, True, True])
    out["rang"] = range(1, len(out) + 1)
    out.attrs["gewichte"] = gewichte
    return out


def empfehlung_je_szenario(detail: pd.DataFrame, anforderungen: dict[str, float],
                           prioritaet: list[str]) -> dict[str, pd.DataFrame]:
    """Rangfolge der Optionen getrennt für jedes Szenario."""
    return {sid: rangiere(d, anforderungen, prioritaet) for sid, d in detail.groupby("szenario", sort=False)}


def robuste_option(detail: pd.DataFrame, anforderungen: dict[str, float], prioritaet: list[str]) -> pd.DataFrame:
    """Eine Option für alle Szenarien zugleich.

    Anforderungen gelten im schlechtesten Szenario (z. B. kleinste Lieferfähigkeit, längste Lieferzeit),
    priorisiert wird nach dem Mittel über die Szenarien.
    """
    g = detail.groupby("option", sort=False)
    df = g[[k for k in KENNZAHLEN if k in detail]].mean()
    for kz in KRITERIEN:
        if kz in detail:
            df[f"{kz}_schlecht"] = g[kz].min() if richtung(kz) > 0 else g[kz].max()
    df = df.reset_index()
    df["beta_min"], df["kosten_max"] = df["beta_schlecht"], df["kosten_pro_uhr_schlecht"]
    return rangiere(df, anforderungen, prioritaet, spalten={kz: f"{kz}_schlecht" for kz in anforderungen})


def _de(x: float, n: int = 1) -> str:
    return f"{x:,.{n}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def vor_und_nachteile(zeile: pd.Series | dict[str, Any]) -> tuple[list[str], list[str]]:
    """Vor- und Nachteile einer Option gegenüber "Ohne Maßnahme" im selben Szenario (gepaarte Differenzen)."""
    z = dict(zeile)
    vor, nach = [], []

    def d(kz: str) -> float:
        v = z.get(f"delta_{kz}", math.nan)
        return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

    regeln = [  # Kennzahl, Text für Verbesserung, Text für Verschlechterung, Formatierung
        ("beta", "Lieferfähigkeit +{} Pp", "Lieferfähigkeit {} Pp", lambda v: _de(v * 100)),
        ("fehlteil_tage", "{} Tage weniger Montagestillstand", "{} Tage mehr Montagestillstand", lambda v: _de(abs(v))),
        ("lieferzeit_p95", "Lieferzeit (95 %) {} Tage kürzer", "Lieferzeit (95 %) {} Tage länger", lambda v: _de(abs(v))),
        ("kosten_pro_uhr", "{} € je Uhr günstiger", "+{} € je Uhr", lambda v: _de(abs(v), 2)),
        ("co2e_pro_uhr_kg", "{} g CO2e je Uhr weniger", "+{} g CO2e je Uhr", lambda v: _de(abs(v) * 1000)),
        ("kapital_gebunden_eur", "{} € weniger Kapital gebunden", "bindet {} € mehr Kapital", lambda v: _de(abs(v), 0)),
    ]
    for kz, gut, schlecht, fmt in regeln:
        v = d(kz)
        if abs(v) < SCHWELLEN[kz]:
            continue
        besser = (v > 0) if RICHTUNG.get(kz, -1) > 0 else (v < 0)
        (vor if besser else nach).append((gut if besser else schlecht).format(fmt(v)))
    b = z.get("bereitschaft_pro_uhr", 0.0)
    if b and not (isinstance(b, float) and math.isnan(b)) and float(b) >= 0.005:
        nach.append(f"Bereitschaftskosten {_de(float(b), 2)} € je Uhr, auch ohne Ausfall")
    a = z.get("aushilfe_tage", 0.0)
    if a and not (isinstance(a, float) and math.isnan(a)) and float(a) >= 0.5:
        nach.append(f"{_de(float(a))} Arbeitstage Aushilfe")
    if z.get("luftfracht_sendungen", 0) and float(z["luftfracht_sendungen"]) >= 0.5:
        nach.append(f"{_de(float(z['luftfracht_sendungen']))} Eilsendungen per Flugzeug")
    return vor, nach


def wirkt(zeile: pd.Series | dict[str, Any]) -> bool:
    """False, wenn sich keine Kennzahl spürbar von "Ohne Maßnahme" unterscheidet (Notfallplan nicht ausgelöst).

    Feste Bereitschaftskosten einer Reservequelle zählen dabei nicht: Sie fallen auch an, wenn der Notfallplan
    nie ausgelöst wird, und werden getrennt als Nachteil genannt.
    """
    z = dict(zeile)
    b = z.get("bereitschaft_pro_uhr", 0.0)
    if b is not None and not (isinstance(b, float) and math.isnan(b)) and "delta_kosten_pro_uhr" in z:
        dk = z["delta_kosten_pro_uhr"]
        if dk is not None and not (isinstance(dk, float) and math.isnan(dk)):
            z["delta_kosten_pro_uhr"] = float(dk) - float(b)
    z["bereitschaft_pro_uhr"] = 0.0
    v, n = vor_und_nachteile(z)
    return bool(v or n)


# --------------------------------------------------------------------------- Stufe 0: Ausschluss
def markiere_ausschluss(detail: pd.DataFrame, regeln: dict[str, str], modell: Any) -> pd.DataFrame:
    """Spalte "ausschluss": verletzte Ausschlussregeln der tatsächlich genutzten Standorte (leer = zulässig)."""
    from .netzanalyse import pruefe_ausschluss
    out = detail.copy()
    out["ausschluss"] = [pruefe_ausschluss(str(g).split(";") if g else [], regeln, modell)
                         for g in out.get("knoten_genutzt", pd.Series([""] * len(out), index=out.index))]
    return out


# --------------------------------------------------------------------------- Break-even p*
def break_even(detail: pd.DataFrame, basis: str = "S0") -> pd.DataFrame:
    """Break-even-Wahrscheinlichkeit p* je (Szenario, Option), Entscheidungsregel aus dem Gruppenposter.

    Eine Vorsorge-Option kostet im Normalbetrieb M (Mehrkosten je Jahr gegenüber "Ohne Maßnahme" in S0) und spart
    im Störungsjahr E (Kosten "Ohne Maßnahme" minus Kosten mit Option im Szenario, inkl. Verspätungskosten).
    Kosten werden je Uhr verglichen und auf die Jahresnachfrage hochgerechnet: Ohne Maßnahme werden im
    Störungsjahr weniger Uhren ausgeliefert, die Gesamtkosten des Jahres wären deshalb scheinbar niedriger.
    Tritt die Störung in einem Jahr mit Wahrscheinlichkeit p ein, lohnt sich die Option, wenn
        p · E  >  (1 − p) · M     also     p  >  p* = M / (M + E).
    Die Eintrittswahrscheinlichkeit wird nicht geschätzt, die Software zeigt nur die Schwelle p*.
    Bewertung:
        "ohne"    M und E praktisch null: Option wirkt in diesem Szenario nicht und kostet nichts
        "immer"   M <= 0 und E >= 0: günstiger im Normalbetrieb und nicht schlechter im Szenario
        "nie"     E <= 0 und M >= 0: hilft im Szenario nicht und kostet etwas
        "ab"      lohnt sich ab p > p*
        "bis"     M < 0, E < 0: spart im Normalbetrieb, verliert im Szenario, lohnt bis p < p*
    """
    zeilen = []
    if "kosten_pro_uhr" not in detail:
        return pd.DataFrame(columns=["szenario", "option", "mehrkosten_normal", "ersparnis_stoerung", "p_stern",
                                     "p_art"])
    idx = detail.set_index(["szenario", "option"])
    k = idx["kosten_pro_uhr"]
    menge = idx["auftraege"] if "auftraege" in idx else pd.Series(1.0, index=idx.index)
    for (sid, oid), _ in k.items():
        if sid == basis or oid == REFERENZ or (basis, oid) not in k or (basis, REFERENZ) not in k \
                or (sid, REFERENZ) not in k:
            continue
        M = float((k[(basis, oid)] - k[(basis, REFERENZ)]) * menge[(basis, REFERENZ)])
        E = float((k[(sid, REFERENZ)] - k[(sid, oid)]) * menge[(sid, REFERENZ)])
        # Differenzen unter 0,05 % der Jahreskosten gelten als null (Rauschen der Simulation)
        tol = max(1.0, 0.0005 * float(k[(basis, REFERENZ)] * menge[(basis, REFERENZ)]))
        M0, E0 = (0.0 if abs(M) < tol else M), (0.0 if abs(E) < tol else E)
        if M0 == 0 and E0 == 0:
            art, p = "ohne", math.nan
        elif M0 <= 0 and E0 >= 0:
            art, p = "immer", 0.0
        elif E0 <= 0 and M0 >= 0:
            art, p = "nie", math.nan
        elif M0 > 0 and E0 > 0:
            art, p = "ab", M0 / (M0 + E0)
        else:
            art, p = "bis", M0 / (M0 + E0)
        zeilen.append({"szenario": sid, "option": oid, "mehrkosten_normal": M, "ersparnis_stoerung": E,
                       "p_stern": p, "p_art": art})
    return pd.DataFrame(zeilen)


def p_text(art: str, p: float) -> str:
    """Lesbare Fassung von p* für die Oberfläche."""
    if art == "ohne":
        return "ohne Wirkung (wird nicht ausgelöst, kostet nichts)"
    if art == "immer":
        return "lohnt sich immer (keine Mehrkosten im Normalbetrieb)"
    if art == "nie":
        return "lohnt sich nie"
    pct = _de(p * 100, 1 if p < 0.1 else 0)
    if art == "ab":
        return f"lohnt sich ab {pct} % Eintrittswahrscheinlichkeit pro Jahr"
    return f"lohnt sich nur bis {pct} % Eintrittswahrscheinlichkeit pro Jahr"


# --------------------------------------------------------------------------- Bandbreite
def bandbreite_definition(szenario: str | Path | dict) -> tuple[str, list[Any]] | None:
    """Bandbreite aus der Szenariodatei (Block sensitivitaet), z. B. ('stoerungen.0.dauer_tage', [15, 30, 60])."""
    import yaml
    if isinstance(szenario, dict):
        roh = szenario
    else:
        with open(szenario, encoding="utf-8") as f:
            roh = yaml.safe_load(f) or {}
    sens = roh.get("sensitivitaet") or {}
    if not sens:
        return None
    pfad, werte = next(iter(sens.items()))
    return str(pfad), list(werte)


def bandbreite(
    szenario: str | Path,
    optionen: list[dict[str, Any]],
    basis_detail: pd.DataFrame,
    anforderungen: dict[str, float],
    prioritaet: list[str],
    overrides: dict[str, Any] | None = None,
    replikationen: int | None = None,
    ausschluss: tuple[dict[str, str], Any] | None = None,
    fortschritt: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Rechnet ein Szenario für jeden Wert der Bandbreite durch und zeigt, ab wann die Empfehlung kippt.

    basis_detail: Zeilen des Normalbetriebs S0 (für p*). Liefert je (Wert, Option) Kennzahlen, Rang und p*.
    """
    import yaml
    defi = bandbreite_definition(szenario)
    if defi is None:
        return pd.DataFrame()
    pfad, werte = defi
    with open(szenario, encoding="utf-8") as f:
        roh = yaml.safe_load(f) or {}
    gesamt, i = len(werte) * len(optionen), 0
    teile = []
    for w in werte:
        sz = _setze_pfad(roh, pfad, w)
        zeilen = []
        for opt in optionen:
            e = fuehre_aus(sz, _overrides_mit(overrides, opt), replikationen)
            z = e["zusammenfassung"]
            zeile = {"szenario": e["szenario_id"], "option": opt["id"], "wert": w,
                     "knoten_genutzt": ";".join(sorted(genutzte_knoten(e["beispiellauf"])))}
            for kz in z.index:
                zeile[kz] = float(z.loc[kz, "mittel"])
            zeilen.append(zeile)
            i += 1
            if fortschritt:
                fortschritt(i, gesamt)
        d = pd.DataFrame(zeilen)
        if ausschluss is not None:
            d = markiere_ausschluss(d, *ausschluss)
        r = rangiere(d, anforderungen, prioritaet)
        spalten = ["szenario", "option", "kosten_pro_uhr", "auftraege"]
        be = break_even(pd.concat([basis_detail[spalten], d[spalten]], ignore_index=True))
        r = r.merge(be[["option", "p_stern", "p_art"]], on="option", how="left")
        teile.append(r)
    out = pd.concat(teile, ignore_index=True)
    out.attrs["parameter"] = pfad
    return out
