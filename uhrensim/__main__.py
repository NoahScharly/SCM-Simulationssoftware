"""Kommandozeile: Szenariovergleich ohne Oberfläche, Ergebnisse als CSV.

Kennzeichnung: G (Architektur und Fachlogik N. Scharly, Code teilweise selbst, teilweise mit Claude Opus 5.5 umgesetzt, Fehlerbehebung mit KI), siehe KI_NACHWEIS.md.

Beispiele:
    python -m uhrensim                         # alle Szenarien, Replikationen aus base.yaml
    python -m uhrensim -r 30 -s S0 S1          # nur S0 und S1, 30 Replikationen
    python -m uhrensim --menge 12000 --kapazitaet 60
    python -m uhrensim --optimierung -s S0 S1 S2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import lade_modell
from .experiment import szenario_pfad, vergleiche
from .optimization import rastersuche
from .visualization import kennzahlentabelle


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulationsmodell Uhren-Supply-Chain")
    ap.add_argument("-s", "--szenarien", nargs="*", help="Szenario-IDs, z. B. S0 S1 (Standard: alle)")
    ap.add_argument("-r", "--replikationen", type=int)
    ap.add_argument("--menge", type=int, help="Uhren pro Jahr (0 bis 25.000)")
    ap.add_argument("--kapazitaet", type=int, help="Montagen pro Tag (1 bis 150)")
    ap.add_argument("--optimierung", action="store_true", help="Rastersuche über Maßnahmen")
    ap.add_argument("-o", "--ausgabe", default="ergebnisse")
    args = ap.parse_args()

    overrides: dict = {}
    if args.menge is not None:
        overrides["nachfrage"] = {"uhren_pro_jahr": args.menge}
    if args.kapazitaet is not None:
        overrides["montage"] = {"kapazitaet_pro_tag": args.kapazitaet}
    for w in lade_modell(None, overrides).warnungen:
        print("WARNUNG:", w)

    pfade = [szenario_pfad(s) for s in args.szenarien] if args.szenarien else None
    ziel = Path(args.ausgabe)
    ziel.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    if args.optimierung:
        df = rastersuche(pfade or [szenario_pfad("S0")], overrides=overrides, replikationen=args.replikationen or 5,
                         fortschritt=lambda i, n: print(f"Kombination {i}/{n}"))
        df.to_csv(ziel / "optimierung.csv", index=False)
        print(df.round(3).to_string(index=False))
        return

    v = vergleiche(pfade, overrides, args.replikationen, fortschritt=lambda s: print(f"{s} fertig"))
    v["tabelle"].to_csv(ziel / "szenariovergleich.csv")
    for sid, e in v["ergebnisse"].items():
        e["laeufe"].to_csv(ziel / f"laeufe_{sid}.csv")
    print(kennzahlentabelle(v["tabelle"]).to_string())
    print(f"\nCSV-Dateien in {ziel.resolve()}")


if __name__ == "__main__":
    main()
