"""Simulationsbasierte Optimierung über diskrete Alternativen (Modul optimization).

Kennzeichnung: G (Architektur und Fachlogik N. Scharly, Code teilweise selbst, teilweise mit Claude Opus 5.5 umgesetzt, Fehlerbehebung mit KI), siehe KI_NACHWEIS.md.

Rastersuche (Folie 5, Schritt 6): Jede Kombination von Maßnahmen wird auf jedem Szenario
mit denselben Zufallszahlen simuliert. Daraus entstehen zwei Empfehlungen:

- Optimum je Szenario: die beste Kombination, wenn man wüsste, welche Störung eintritt.
- Robuste Empfehlung: die beste Kombination über alle Szenarien zugleich. Zulässig ist sie nur,
  wenn das Lieferversprechen auch im schlechtesten Szenario hält (beta_min statt Mittelwert).
  Das ist die eigentliche Managemententscheidung, weil man vorher nicht weiß, welche Störung kommt.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .data import lade_modell
from .evaluation import bewerte
from .experiment import fuehre_aus

STANDARD_RASTER: dict[str, list[Any]] = {
    "puffer_uhrwerk": [0, 500, 1000],
    "sellita_notfall": [False, True],
    "luftfracht_notfall": [False, True],
}
KENNZAHLEN = ("beta", "kosten_pro_uhr", "co2e_pro_uhr_kg", "lieferzeit_p95", "fehlteil_tage")


def _massnahmen(kombination: dict[str, Any]) -> dict[str, Any]:
    return {
        "puffer_extra": {"uhrwerk": kombination["puffer_uhrwerk"]},
        "sellita_notfall": kombination["sellita_notfall"],
        "luftfracht_notfall": kombination["luftfracht_notfall"],
    }


def kombinationen(raster: dict[str, list[Any]] | None = None) -> list[dict[str, Any]]:
    raster = raster or STANDARD_RASTER
    return [dict(zip(raster, werte)) for werte in itertools.product(*raster.values())]


def raster_detail(
    szenarien: list[str | Path],
    raster: dict[str, list[Any]] | None = None,
    overrides: dict[str, Any] | None = None,
    replikationen: int = 10,
    fortschritt: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Simuliert jede Kombination auf jedem Szenario. Eine Zeile je Kombination und Szenario."""
    kombis = kombinationen(raster)
    gesamt, i = len(kombis) * len(szenarien), 0
    zeilen = []
    for k_nr, kombi in enumerate(kombis):
        ov = copy.deepcopy(overrides or {})
        ov["massnahmen"] = (ov.get("massnahmen") or {}) | _massnahmen(kombi)
        for sz in szenarien:
            e = fuehre_aus(sz, ov, replikationen)
            mittel = e["zusammenfassung"]["mittel"]
            zeilen.append(kombi | {"kombination": k_nr, "szenario": e["szenario_id"]}
                          | {kz: float(mittel[kz]) for kz in KENNZAHLEN})
            i += 1
            if fortschritt:
                fortschritt(i, gesamt)
    return pd.DataFrame(zeilen)


def optimum_je_szenario(detail: pd.DataFrame, bewertung: dict[str, Any]) -> pd.DataFrame:
    """Beste Kombination je Szenario nach der gewählten Bewertungslogik."""
    beste = []
    for sid, d in detail.groupby("szenario", sort=False):
        b = bewerte(d.reset_index(drop=True), bewertung).iloc[0]
        beste.append(b)
    return pd.DataFrame(beste).reset_index(drop=True)


def robuste_empfehlung(detail: pd.DataFrame, bewertung: dict[str, Any]) -> pd.DataFrame:
    """Mittel über alle Szenarien; Mindestwert für beta gilt für das schlechteste Szenario."""
    namen = [c for c in STANDARD_RASTER if c in detail.columns]
    g = detail.groupby("kombination")
    df = g[namen].first().join(g[list(KENNZAHLEN)].mean())
    df["beta_min"] = g["beta"].min()
    df["kosten_max"] = g["kosten_pro_uhr"].max()
    df = df.reset_index()
    bew = copy.deepcopy(bewertung)
    mw = dict(bew.get("mindestwerte") or {})
    if "beta" in mw:
        mw["beta_min"] = mw.pop("beta")
    bew["mindestwerte"] = mw
    return bewerte(df, bew)


def szenario_optimierung(
    szenarien: list[str | Path],
    raster: dict[str, list[Any]] | None = None,
    overrides: dict[str, Any] | None = None,
    replikationen: int = 10,
    bewertung: dict[str, Any] | None = None,
    fortschritt: Callable[[int, int], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Alle Szenarien × alle Maßnahmen: Detail, Optimum je Szenario und robuste Empfehlung."""
    bewertung = bewertung or lade_modell().param["bewertung"]
    detail = raster_detail(szenarien, raster, overrides, replikationen, fortschritt)
    return {"detail": detail, "je_szenario": optimum_je_szenario(detail, bewertung),
            "robust": robuste_empfehlung(detail, bewertung)}


def rastersuche(
    szenarien: list[str | Path],
    raster: dict[str, list[Any]] | None = None,
    overrides: dict[str, Any] | None = None,
    replikationen: int = 10,
    bewertung: dict[str, Any] | None = None,
    fortschritt: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Kompatible Kurzform: nur die robuste Empfehlung über alle Szenarien."""
    return szenario_optimierung(szenarien, raster, overrides, replikationen, bewertung, fortschritt)["robust"]
