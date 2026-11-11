# Urheberschaft: Ferdinand Lautenbach (F); KI-gestützt (Claude)
"""Netzanalyse: Single Points of Failure (Modul netzanalyse).


Single Point of Failure (SPOF):
    Ein Element des regulären Lieferwegs (Lieferant, Vorlieferant, Hafen, Strecke) ist ein SPOF, wenn es auf
    JEDEM möglichen Weg der Komponente nach Heilbronn liegt, also auch dann, wenn alle Handlungsoptionen
    (Ausweichroute, Luftfracht, Ersatzlieferant, Bezugsoption) genutzt werden. Fällt es aus, gibt es keinen Ersatz.
    Elemente, die nur im Normalbetrieb ohne Ausweichweg sind, werden mit der Option ausgewiesen, die sie absichert.
    Annahme: Ein Lieferant braucht alle seine Vorlieferanten (jeder liefert ein anderes Material).
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .domain import Modell

# Welche Maßnahme (base.yaml, Block massnahmen) einen zusätzlichen Weg freischaltet
MASSNAHME_FUER_ROUTENTYP = {"ausweich": "ausweichroute", "notfall": "luftfracht_notfall"}
MASSNAHME_FUER_ROLLE = {"alternativ": "sellita_notfall"}


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

