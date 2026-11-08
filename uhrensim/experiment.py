"""Monte-Carlo-Experimente, Szenariovergleich und Sensitivitätsanalyse.

Kennzeichnung: KI (Claude Opus 5.5), Prompts siehe KI_NACHWEIS.md.

Alle Szenarien eines Vergleichs nutzen denselben Seed und dieselben Replikationsnummern.
Dadurch sind die Zufallsströme gleich (Common Random Numbers), und Differenzen zur Basis
sind weniger verrauscht als bei unabhängigen Läufen.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from .data import KONFIG_STANDARD, lade_modell, szenario_dateien
from .evaluation import kennzahlen, resilienz
from .simulation import Ergebnis, simuliere

# t-Quantile (zweiseitig 95 %) für kleine Stichproben, sonst Normalverteilung
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000}


def t95(freiheitsgrade: int) -> float:
    if freiheitsgrade <= 0:
        return math.nan
    passend = [k for k in _T95 if k <= freiheitsgrade]
    return _T95[max(passend)] if freiheitsgrade <= 60 else 1.96


def zusammenfassen(df: pd.DataFrame) -> pd.DataFrame:
    """Mittelwert und Halbbreite des 95-%-Konfidenzintervalls je Kennzahl."""
    zeilen = []
    for spalte in df.columns:
        x = pd.to_numeric(df[spalte], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            zeilen.append({"kennzahl": spalte, "mittel": math.nan, "ki95": math.nan, "n": 0})
            continue
        hb = t95(len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else math.nan
        zeilen.append({"kennzahl": spalte, "mittel": float(x.mean()), "ki95": float(hb), "n": len(x)})
    return pd.DataFrame(zeilen).set_index("kennzahl")


def fuehre_aus(
    szenario: str | Path | dict | None = None,
    overrides: dict[str, Any] | None = None,
    replikationen: int | None = None,
    deterministisch: bool = False,
    fortschritt: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Alle Replikationen eines Szenarios. Liefert Kennzahlen je Lauf, Zusammenfassung und Lauf 0."""
    modell = lade_modell(szenario, overrides)
    n = 1 if deterministisch else int(replikationen or modell.param["simulation"]["replikationen"])
    laeufe, res, erstes = [], [], None
    for r in range(n):
        erg = simuliere(modell, replikation=r, deterministisch=deterministisch)
        laeufe.append(kennzahlen(erg) | {"replikation": r})
        res.extend(x | {"replikation": r} for x in resilienz(erg))
        if r == 0:
            erstes = erg
        if fortschritt:
            fortschritt(r + 1, n)
    df = pd.DataFrame(laeufe).set_index("replikation")
    return {
        "modell": modell, "szenario_id": modell.szenario_id, "szenario_name": modell.szenario_name,
        "laeufe": df, "zusammenfassung": zusammenfassen(df), "resilienz": pd.DataFrame(res),
        "beispiellauf": erstes, "warnungen": modell.warnungen,
    }


def vergleiche(
    szenarien: list[str | Path] | None = None,
    overrides: dict[str, Any] | None = None,
    replikationen: int | None = None,
    fortschritt: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Szenariovergleich: Mittelwert, KI und Differenz zur Basis (erstes Szenario)."""
    szenarien = szenarien or szenario_dateien()
    ergebnisse = {}
    for pfad in szenarien:
        e = fuehre_aus(pfad, overrides, replikationen)
        ergebnisse[e["szenario_id"]] = e
        if fortschritt:
            fortschritt(e["szenario_id"])
    basis_id = next(iter(ergebnisse))
    basis = ergebnisse[basis_id]["laeufe"]
    zeilen = []
    for sid, e in ergebnisse.items():
        z = e["zusammenfassung"]
        zeile = {"szenario": sid, "name": e["szenario_name"]}
        for kz in z.index:
            zeile[kz] = z.loc[kz, "mittel"]
            zeile[f"{kz}_ki95"] = z.loc[kz, "ki95"]
        # gepaarte Differenz zur Basis (Common Random Numbers)
        for kz in ("beta", "kosten_pro_uhr", "co2e_pro_uhr_kg", "lieferzeit_p95"):
            d = (e["laeufe"][kz] - basis[kz]).dropna()
            zeile[f"delta_{kz}"] = float(d.mean()) if not d.empty else math.nan
        zeilen.append(zeile)
    return {"tabelle": pd.DataFrame(zeilen).set_index("szenario"), "ergebnisse": ergebnisse}


def _setze_pfad(d: dict, pfad: str, wert: Any) -> dict:
    """Setzt einen Wert über einen Punkt-Pfad, z. B. 'stoerungen.0.dauer_tage'."""
    d = copy.deepcopy(d)
    ziel = d
    teile = pfad.split(".")
    for t in teile[:-1]:
        ziel = ziel[int(t)] if isinstance(ziel, list) else ziel.setdefault(t, {})
    letzter = teile[-1]
    if isinstance(ziel, list):
        ziel[int(letzter)] = wert
    else:
        ziel[letzter] = wert
    return d


def sensitivitaet(
    szenario: str | Path | None,
    pfad: str,
    werte: list[Any],
    replikationen: int | None = None,
    im_szenario: bool = False,
) -> pd.DataFrame:
    """Ein Parameter je Lauf (One-at-a-time). pfad zeigt in base.yaml oder, mit im_szenario, ins Szenario."""
    zeilen = []
    for w in werte:
        if im_szenario:
            with open(szenario, encoding="utf-8") as f:
                sz = _setze_pfad(yaml.safe_load(f), pfad, w)
            e = fuehre_aus(sz, None, replikationen)
        else:
            e = fuehre_aus(szenario, _setze_pfad({}, pfad, w), replikationen)
        z = e["zusammenfassung"]
        zeilen.append({"parameter": pfad, "wert": w,
                       **{kz: z.loc[kz, "mittel"] for kz in ("beta", "kosten_pro_uhr", "co2e_pro_uhr_kg",
                                                              "lieferzeit_p95", "fehlteil_tage")}})
    return pd.DataFrame(zeilen)


def szenario_pfad(kennung: str, konfig_ordner: str | Path = KONFIG_STANDARD) -> Path:
    """Findet eine Szenario-Datei über ihre ID (z. B. 'S1')."""
    for p in szenario_dateien(konfig_ordner):
        with open(p, encoding="utf-8") as f:
            if str((yaml.safe_load(f) or {}).get("id")) == kennung:
                return p
    raise KeyError(f"Szenario {kennung} nicht gefunden")
