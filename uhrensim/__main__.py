"""Kommandozeile: Szenariovergleich ohne Oberfläche, Ergebnisse als CSV.



Beispiele:
    python -m uhrensim                         # alle Szenarien, Replikationen aus base.yaml
    python -m uhrensim -r 30 -s S0 S1          # nur S0 und S1, 30 Replikationen
    python -m uhrensim --menge 12000 --kapazitaet 60
    python -m uhrensim --optimierung              # alle Szenarien × alle Maßnahmen, Optimum je Szenario + robust
    python -m uhrensim --sensitivitaet -s S1      # Parameter aus base.yaml (Block sensitivitaet), einzeln variiert
    python -m uhrensim --optionen                 # je Szenario alle Handlungsoptionen, Empfehlung je Szenario + robust
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import lade_modell
from .experiment import sensitivitaet, szenario_pfad, vergleiche
from .optimization import szenario_optimierung
from .visualization import kennzahlentabelle


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulationsmodell Uhren-Supply-Chain")
    ap.add_argument("-s", "--szenarien", nargs="*", help="Szenario-IDs, z. B. S0 S1 (Standard: alle)")
    ap.add_argument("-r", "--replikationen", type=int)
    ap.add_argument("--menge", type=int, help="Uhren pro Jahr (0 bis 25.000)")
    ap.add_argument("--kapazitaet", type=int, help="Montagen pro Tag (1 bis 150)")
    ap.add_argument("--optimierung", action="store_true", help="Rastersuche über Maßnahmen je Szenario")
    ap.add_argument("--optionen", action="store_true", help="Handlungsoptionen je Szenario vergleichen")
    ap.add_argument("--sensitivitaet", action="store_true", help="One-at-a-time-Sensitivität aus base.yaml")
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
        from .data import szenario_dateien
        erg = szenario_optimierung(pfade or szenario_dateien(), overrides=overrides,
                                   replikationen=args.replikationen or 5,
                                   fortschritt=lambda i, n: print(f"Lauf {i}/{n}", end="\r"))
        for name, df in erg.items():
            df.to_csv(ziel / f"optimierung_{name}.csv", index=False)
        print("\nOptimum je Szenario:")
        print(erg["je_szenario"].round(3).to_string(index=False))
        print("\nRobuste Empfehlung über alle Szenarien (Rang 1 oben):")
        print(erg["robust"].round(3).head(5).to_string(index=False))
        return

    if args.optionen:
        from .data import szenario_dateien
        from .handlungsoptionen import (empfehlung_je_szenario, lade_entscheidungsregel, robuste_option,
                                        vergleiche_optionen)
        anf, prio = lade_entscheidungsregel()
        v = vergleiche_optionen(pfade or szenario_dateien(), overrides=overrides,
                                replikationen=args.replikationen or 5,
                                fortschritt=lambda i, n: print(f"Lauf {i}/{n}", end="\r"))
        v["detail"].to_csv(ziel / "handlungsoptionen.csv", index=False)
        print(f"\nEmpfehlung je Szenario (Anforderungen {anf}, Priorität {prio}):")
        for sid, r in empfehlung_je_szenario(v["detail"], anf, prio).items():
            b = r.iloc[0]
            print(f"  {sid}: {b.option:16s} beta {b.beta:.3f}  {b.kosten_pro_uhr:.2f} EUR/Uhr")
        print("\nRobuste Option über alle Szenarien:")
        print(robuste_option(v["detail"], anf, prio)[["rang", "option", "beta_min", "kosten_pro_uhr",
                                                             "co2e_pro_uhr_kg", "zulaessig"]].round(3).to_string(index=False))
        return

    if args.sensitivitaet:
        basis = lade_modell(None, overrides)
        sz = pfade[0] if pfade else None
        teile = [sensitivitaet(sz, pfad, werte, args.replikationen)
                 for pfad, werte in (basis.param.get("sensitivitaet") or {}).items()]
        df = pd.concat(teile, ignore_index=True)
        df.to_csv(ziel / "sensitivitaet.csv", index=False)
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
