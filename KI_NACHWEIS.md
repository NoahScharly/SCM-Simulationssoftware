# KI-Nutzungsnachweis (Aufgabenstellung 9.3)

**System und Version:** Claude Opus 5.5 (Anthropic), claude.ai, November 2026

**Eigene Leistung (ohne KI):** Basis-Montagekapazität (ca. 25 Uhren/Tag), Montageleistung von ca. 3 Uhren je Uhrmacher und Tag (eigene Recherche Manufakturpraxis, vorher 7) zur Personalbedarfsermittlung und Jahresmenge als frei einstellbare Regler, Simulationskonzept, Softwarearchitektur und Modulstruktur, Recherche aller
Quellen und Daten (Netz, Lieferanten, Komponenten, Emissionsfaktoren), Startkonfiguration aus dem eigenen Poster.

**Verwendungszweck der KI:** teilweise Umsetzung der vorgegebenen Architektur in Python-Code sowie
Fehlerbehebung (Prüfungselement 2). Der übrige Code wurde selbst geschrieben.

**Betroffene Programmteile:** Code in `uhrensim/`, `app/`, `tests/` (Kennzeichen G).

## Wesentliche Prompts (Auswahl, chronologisch)

1. "Ich baue mit Python und SimPy eine ereignisdiskrete Simulation der Lieferkette einer Automatikuhr, die in
   Heilbronn montiert wird. Schlag mir eine Ordnerstruktur mit den Modulen domain, data, simulation,
   optimization, evaluation, compliance und ui vor. Schreib noch keinen Code."
2. "Setz in SimPy die Prozesse Lieferant, Transport, Montage und Kunde um. Die Montage verbraucht pro Uhr ein Kit
   aus 6 Komponenten und darf höchstens die eingestellte Anzahl Uhren pro Tag fertigen. Jede Zustandsänderung
   soll als Zeile in ein Ereignisprotokoll geschrieben werden."
3. "Richte die Zufallszahlen mit numpy SeedSequence so ein, dass jede Zufallsquelle (Nachfrage, Zollzeit,
   Gutquote) ihren eigenen Generator hat und alle Szenarien dieselben Ströme nutzen. Führe jedes Szenario
   30 Mal aus."
4. "Berechne aus dem Ereignisprotokoll diese Kennzahlen: Gesamtkosten, Lieferfähigkeit β, Lieferzeit (Mittelwert
   und 95-%-Quantil), CO2e nach GLEC sowie TTS und TTR je Störung. Gib jeden Wert mit 95-%-Konfidenzintervall aus."
5. "Bau mir eine Streamlit-App mit drei Ansichten: eine animierte Plotly-Karte, auf der Sendungen entlang der
   Kanten wandern, Diagramme zum Bestand und zu β über die Zeit und eine Kennzahlentabelle Szenario × Kriterium.
   Alles soll aus demselben Ereignisprotokoll kommen."

Weitere Prompts betrafen das Laden und Prüfen der Netzdaten, das Bestellpunktverfahren (s, Q), Szenarien als
YAML-Änderungen gegenüber base.yaml, pytest-Tests zur Mengenbilanz und die Fehlerbehebung.

6. "Erweitere die Rastersuche so, dass alle Szenarien mit allen Maßnahmen durchlaufen und das Optimum je
   Szenario vorgeschlagen wird. Dual Sourcing soll nicht drin sein, nur Sellita als kompatibles Ersatzwerk."
7. "Zeig rechts neben der Karte die Bestände im Lager als animierte Balken, die zu- und abnehmen, und
   markiere, wann eine Bestellung ausgelöst wird." 

## Von der KI getroffene Modellentscheidungen (vom Studierenden zu prüfen)

- Abfahrtsintervalle, Umschlag- und Zollzeiten, Transportkostensätze, Gewichte, Preise außer Miyota: Annahmen [A]
- Lieferzeit in Werktagen statt Kalendertagen
- Sicherheitsbestand nach SS = z · sqrt(L · σd² + d² · σL²)
- TTR als Eingabe (Störungsdauer plus Transit), TTS aus der Simulation
- Robuste Empfehlung: zulässig nur, wenn β im schlechtesten Szenario den Mindestwert erreicht
- Bestandsbalken als Reichweite in Tagen statt Stück

## Vorgenommene Änderungen

[hier eigene Änderungen eintragen]

## Fachliche und technische Prüfungen

- 23 automatische Tests (`pytest -q`): Mengenbilanz, deterministischer Transit gegen Handrechnung,
  Reproduzierbarkeit, Common Random Numbers, Extremwerte, Umleitung bei Niedrigwasser, Bewertung, Optimierung je Szenario, Bestandsanimation
- [hier eigene Prüfungen eintragen, z. B. Parameter gegen Quellen geprüft, Ergebnisse plausibilisiert]
