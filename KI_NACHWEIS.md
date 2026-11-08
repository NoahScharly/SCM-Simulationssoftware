# KI-Nutzungsnachweis (Aufgabenstellung 9.3)

**System und Version:** Claude Opus 5.5 (Anthropic), claude.ai, November 2026

**Verwendungszweck:** Programmierung der Simulationssoftware nach dem eigenen Simulationskonzept
(Prüfungselement 2): Konfigurationsdateien, alle Python-Module, Oberfläche, Tests, Dokumentation.

**Betroffene Programmteile:** alle Dateien in `config/`, `uhrensim/`, `app/`, `tests/` sowie `README.md`.

## Wesentliche Prompts (sinngemäß, chronologisch)

1. "Lass uns die Software jetzt zusammen bauen. Du sollst so viel wie möglich übernehmen."
   Vorgaben aus dem Konzept: ereignisdiskrete Simulation mit SimPy, Python und Streamlit, Netz und Szenarien
   als Daten, vier Kernkriterien plus Resilienz und Bestand, drei Sichten aus einer Datenquelle.
2. Netzdaten: eigenes Poster als Startkonfiguration, Kap-Route, ab Rotterdam Binnenschiff, ab Hamburg Bahn,
   später durch Gruppendaten ersetzbar.
3. Uhren pro Jahr frei einstellbar von 0 bis 25.000, Montagen pro Tag und Wiederbeschaffungszeit anpassbar,
   aber so, dass es realistisch bleibt.

## Von der KI getroffene Modellentscheidungen (vom Studierenden zu prüfen)

- Basis-Montagekapazität 22 statt 20 Uhren/Tag, weil 20 × 250 × 0,99 < 5.000
- Abfahrtsintervalle, Umschlag- und Zollzeiten, Transportkostensätze, Gewichte, Preise außer Miyota: Annahmen [A]
- Lieferzeit in Werktagen statt Kalendertagen
- Sicherheitsbestand nach SS = z · sqrt(L · σd² + d² · σL²)
- TTR als Eingabe (Störungsdauer plus Transit), TTS aus der Simulation

## Vorgenommene Änderungen

[hier eigene Änderungen eintragen]

## Fachliche und technische Prüfungen

- 19 automatische Tests (`pytest -q`): Mengenbilanz, deterministischer Transit gegen Handrechnung,
  Reproduzierbarkeit, Common Random Numbers, Extremwerte, Umleitung bei Niedrigwasser, Bewertung
- [hier eigene Prüfungen eintragen, z. B. Parameter gegen Quellen geprüft, Ergebnisse plausibilisiert]
