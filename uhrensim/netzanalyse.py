"""Netzanalyse: Single Points of Failure und harte Ausschlusskriterien (Modul netzanalyse).

Kennzeichnung: G (Fachlogik aus den Gruppenpostern, Ausarbeitung N. Scharly, Code mit Claude Opus 5.5 umgesetzt),
siehe KI_NACHWEIS.md.

Single Point of Failure (SPOF, Idee Gruppenposter):
    Ein Element des regulären Lieferwegs (Lieferant, Vorlieferant, Hafen, Strecke) ist ein SPOF, wenn es auf
    JEDEM möglichen Weg der Komponente nach Heilbronn liegt, also auch dann, wenn alle Handlungsoptionen
    (Ausweichroute, Luftfracht, Ersatzlieferant, Bezugsoption) genutzt werden. Fällt es aus, gibt es keinen Ersatz.
    Elemente, die nur im Normalbetrieb ohne Ausweichweg sind, werden mit der Option ausgewiesen, die sie absichert.
    Annahme: Ein Lieferant braucht alle seine Vorlieferanten (jeder liefert ein anderes Material).

Ausschlusskriterien (Stufe 0 der Entscheidung, Idee Gruppenposter):
    Länder oder einzelne Standorte, die aus rechtlichen oder ethischen Gründen nicht genutzt werden dürfen
    (z. B. EU-Sanktionen). Geprüft wird, welche Standorte eine Handlungsoption im simulierten Lauf tatsächlich
    nutzt; nutzt sie einen ausgeschlossenen Standort, wird sie vor jeder Bewertung aussortiert.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .domain import Modell

# Welche Maßnahme (base.yaml, Block massnahmen) einen zusätzlichen Weg freischaltet
MASSNAHME_FUER_ROUTENTYP = {"ausweich": "ausweichroute", "notfall": "luftfracht_notfall"}
MASSNAHME_FUER_ROLLE = {"alternativ": "sellita_notfall", "option": "armband_italien"}


def _wege(m: Modell, komponente: str) -> list[dict[str, Any]]:
    """Alle möglichen Wege einer Komponente: jede Bezugsquelle × jede Route ihres Lieferanten."""
    wege = []
    for q in m.quellen.get(komponente, []):
        for r in m.routen.get(q.lieferant, []):
            knoten = [q.lieferant, *q.tier2]
            for kid in r.kanten:
                k = m.kanten[kid]
                knoten += [k.von, k.nach]
            massnahme = MASSNAHME_FUER_ROLLE.get(q.rolle) or MASSNAHME_FUER_ROUTENTYP.get(r.typ)
            wege.append({"quelle": q, "route": r, "knoten": set(knoten), "kanten": set(r.kanten),
                         "regulaer": q.rolle == "primaer" and r.typ == "regulaer", "massnahme": massnahme})
    return wege


def single_points_of_failure(m: Modell, optionen: Iterable[dict[str, Any]] = ()) -> pd.DataFrame:
    """Je Komponente die Elemente des regulären Wegs und ob es dafür einen Ausweichweg gibt.

    Spalten: komponente, element, typ (Knoten | Strecke), spof (kein Ausweichweg, auch nicht mit Optionen),
    abgesichert_durch (Namen der Handlungsoptionen, deren Weg das Element umgeht).
    """
    name_fuer_massnahme: dict[str, str] = {}
    for o in optionen:
        for key in (o.get("massnahmen") or {}):
            name_fuer_massnahme.setdefault(key, o.get("kurz") or o.get("name") or o["id"])
    werk = m.werk
    zeilen = []
    for komp in m.komponenten:
        wege = _wege(m, komp)
        basis = next((w for w in wege if w["regulaer"]), None)
        if basis is None:
            continue
        elemente = [(n, "Knoten") for n in sorted(basis["knoten"] - {werk})] + \
                   [(k, "Strecke") for k in basis["route"].kanten]
        for el, typ in elemente:
            umgehen = [w for w in wege if el not in (w["knoten"] if typ == "Knoten" else w["kanten"])]
            namen = sorted({name_fuer_massnahme.get(w["massnahme"], w["massnahme"] or "") for w in umgehen} - {""})
            zeilen.append({"komponente": komp, "element": el, "typ": typ, "spof": not umgehen,
                           "abgesichert_durch": ", ".join(namen)})
    df = pd.DataFrame(zeilen, columns=["komponente", "element", "typ", "spof", "abgesichert_durch"])
    df.attrs["werk"] = werk
    return df


def element_name(m: Modell, element: str) -> str:
    if element in m.knoten:
        k = m.knoten[element]
        return f"{k.name} ({k.land})"
    if element in m.kanten:
        k = m.kanten[element]
        return f"{m.knoten[k.von].name} → {m.knoten[k.nach].name} ({k.modus})"
    return element


# --------------------------------------------------------------------------- Ausschluss
def genutzte_knoten(erg: Any) -> set[str]:
    """Standorte, die ein Simulationslauf tatsächlich genutzt hat: Lieferanten mit Bestellungen und deren
    Vorlieferanten, alle Knoten der befahrenen Strecken."""
    if erg is None or erg.log is None or len(erg.log) == 0:
        return set()
    m, log = erg.modell, erg.log
    knoten: set[str] = set()
    lieferanten = set(log.loc[log.ereignis == "bestellung", "ort"].astype(str))
    knoten |= lieferanten
    for qs in m.quellen.values():
        for q in qs:
            if q.lieferant in lieferanten:
                knoten |= set(q.tier2)
    for kid in set(log.loc[log.ereignis == "abfahrt", "kante"].astype(str)):
        if kid in m.kanten:
            knoten |= {m.kanten[kid].von, m.kanten[kid].nach}
    return {k for k in knoten if k in m.knoten}


def ausschluss_regeln(m: Modell, laender: Iterable[str] | None = None,
                      knoten: dict[str, str] | None = None) -> dict[str, str]:
    """Ausgeschlossene Standorte des Netzes mit Grund. Standard aus base.yaml (Block ausschluss)."""
    cfg = m.param.get("ausschluss") or {}
    laender = {str(x).upper() for x in (cfg.get("laender") or [])} if laender is None else \
        {str(x).upper() for x in laender}
    gruende = dict(cfg.get("knoten") or {}) if knoten is None else dict(knoten)
    for k in m.knoten.values():
        if k.land.upper() in laender and k.id not in gruende:
            gruende[k.id] = f"Land {k.land} ausgeschlossen"
    return {k: v for k, v in gruende.items() if k in m.knoten}


def pruefe_ausschluss(genutzt: Iterable[str], regeln: dict[str, str], m: Modell) -> str:
    """Leerer Text = zulässig, sonst die verletzten Regeln."""
    treffer = [f"{m.knoten[k].name}: {regeln[k]}" for k in sorted(set(genutzt)) if k in regeln]
    return "; ".join(treffer)
