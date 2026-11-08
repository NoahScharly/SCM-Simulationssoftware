"""Simulationsbasierte Optimierung über diskrete Alternativen (Modul optimization).

Kennzeichnung: G (Architektur und Fachlogik N. Scharly, Code teilweise selbst, teilweise mit Claude Opus 5.5 umgesetzt, Fehlerbehebung mit KI), siehe KI_NACHWEIS.md.

Rastersuche (Folie 4, Schritt 6): Jede Kombination von Maßnahmen wird auf denselben
Szenarien mit denselben Zufallszahlen simuliert. Bewertet wird der Mittelwert über alle
Szenarien, damit eine Maßnahme nicht nur für eine einzige Störung gut aussieht.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .evaluation import bewerte
from .experiment import fuehre_aus
from .data import lade_modell

STANDARD_RASTER: dict[str, list[Any]] = {
    "puffer_uhrwerk": [0, 500, 1000],
    "dual_sourcing": [False, True],
    "luftfracht_notfall": [False, True],
}


def _massnahmen(kombination: dict[str, Any]) -> dict[str, Any]:
    return {"massnahmen": {
        "puffer_extra": {"uhrwerk": kombination["puffer_uhrwerk"]},
        "dual_sourcing": kombination["dual_sourcing"],
        "luftfracht_notfall": kombination["luftfracht_notfall"],
    }}


def rastersuche(
    szenarien: list[str | Path],
    raster: dict[str, list[Any]] | None = None,
    overrides: dict[str, Any] | None = None,
    replikationen: int = 10,
    bewertung: dict[str, Any] | None = None,
    fortschritt: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Alle Kombinationen simulieren und mit evaluation.bewerte einordnen."""
    raster = raster or STANDARD_RASTER
    namen = list(raster)
    kombis = [dict(zip(namen, werte)) for werte in itertools.product(*raster.values())]
    zeilen = []
    for i, kombi in enumerate(kombis, 1):
        ov = (overrides or {}) | _massnahmen(kombi)
        if overrides and "massnahmen" in overrides:
            ov["massnahmen"] = overrides["massnahmen"] | _massnahmen(kombi)["massnahmen"]
        werte = []
        for sz in szenarien:
            z = fuehre_aus(sz, ov, replikationen)["zusammenfassung"]["mittel"]
            werte.append(z)
        mittel = pd.concat(werte, axis=1).mean(axis=1)
        zeilen.append(kombi | {
            "beta": mittel["beta"], "beta_min": min(w["beta"] for w in werte),
            "kosten_pro_uhr": mittel["kosten_pro_uhr"], "co2e_pro_uhr_kg": mittel["co2e_pro_uhr_kg"],
            "lieferzeit_p95": mittel["lieferzeit_p95"], "fehlteil_tage": mittel["fehlteil_tage"],
        })
        if fortschritt:
            fortschritt(i, len(kombis))
    df = pd.DataFrame(zeilen)
    bewertung = bewertung or lade_modell().param["bewertung"]
    return bewerte(df, bewertung)
