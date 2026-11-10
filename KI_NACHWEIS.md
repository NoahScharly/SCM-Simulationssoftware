# KI-Nutzungsnachweis (Aufgabenstellung 9.3)

**System und Version:** Claude Opus 5.5 (Anthropic), claude.ai, November 2026

**Eigene Leistung (ohne KI):** Basis-Montagekapazität (ca. 25 Uhren/Tag), Montageleistung von 5 Uhren je Uhrmacher und Tag (eigene Festlegung nach Recherche Manufakturpraxis, vorher 7) zur Personalbedarfsermittlung und Jahresmenge als frei einstellbare Regler, Simulationskonzept, Softwarearchitektur und Modulstruktur, Recherche aller
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
8. "Sellita soll nur als Notfallplan gelten: zuerst kommt das Uhrwerk von Miyota, nach dem Ausfall aus der
   Schweiz. Das soll man auf der Karte sehen, dazu eine kleine Karte mit Zoom auf Europa und halbe
   Abspielgeschwindigkeit." 
9. "Es gibt Uhrwerk mit Notfallplan mit Sellita, kein Dual Sourcing. Dann noch Bestand hoch als Möglichkeit. Das
   bindet aber Kapital und das muss auch irgendwie mit in den Vorschlag gerechnet werden. Vielleicht noch
   Lederarmband in Italien komplett."

10. (09.11.2026, Umbau des Ablaufs) "Man braucht als Basis Lieferanten, verschiedene Transportmittel und die
   gesamten Daten, darauf den Normalbetrieb simulieren. Vor dem Start: welches Szenario soll simuliert werden. Nach
   dem Start wird der Fall angezeigt, dann mit einem Schalter anzeigen, was die Ausweichsituationen bringen, welche
   Vor- und Nachteile man hat. Oder es simuliert direkt mit allen Alternativen durch und macht Vorschläge. Seed eher
   versteckt." Umsetzung: Modul `handlungsoptionen.py`, Optionen in `base.yaml`, App-Ablauf neu, Seed unter
   Experteneinstellungen, 6 neue Tests.

11. (09.11.2026, Luftfracht in S1) "Bei S1: Uhrwerk fällt 60 Tage aus. Weil es Notfall ist und schneller Uhrwerke
   gebraucht werden, um Stillstand zu minimieren, holt man die erste Lieferung per Flugzeug. Das ist dann aber bei
   den Werken schon teuer, das muss in der Datenbasis richtig sein." Umsetzung: Brückenlieferung beim Wiederanlauf,
   Eilzuschlag, Luftfracht nur bei Versorgungsstörung, Route über Narita, Auslöser über Luft-WBZ, 2 neue Tests.

12. (09.11.2026, Zielkriterien und Bestandsbalken) "Das Uhrwerk wiegt 79 g. Der Balken beim Bestand, der die
   Neubestellung triggert, ist verbuggt. Zielkriterien besser auswählen lassen: aus Optionen wählen und von oben nach
   unten anordnen, Lieferfähigkeit mit Standardwert einstellbar, Lieferzeit unter 3 bis 4 Tagen als Standard, oder
   ohne Werte nur priorisieren, das kannst du entscheiden." Umsetzung: Gewicht 79 g, Bestandsbalken und
   "bestellt" aus demselben Zeitfenster, Mindestanforderungen plus Drag-and-drop-Priorisierung mit ROC-Gewichten.

13. (09.11.2026, Szenarien und Anzeige) "Das Bestellen blinkt nur auf und irritiert, die Balken sind ja immer an
   der gleichen Stelle. Was dunkler, heller Balken und rot bedeuten, muss rechts klein hin. Je Szenario nur die
   relevanten Handlungsoptionen. Niedrigwasser ohne Maßnahme: die Ware über Kaub kommt nicht an, Hamburg ebenso,
   dazu die besprochenen Ausweichmöglichkeiten. Saisonspitze: Bestand aufbauen; mehr Arbeiter eher nicht."
   Umsetzung: Statusspalte (↻ / ✓) und Legende, Ausweichroute als eigene Option, relevante Optionen je Szenario,
   Saisonbestand, Hinweis auf festhängende Sendungen, 2 neue Tests.

14. (10.11.2026, Abgleich mit der Präsentation) "Baue mir alle Dinge, die wir jetzt noch geändert haben, in die
   Software ein": Suez-Szenario C1 entfernt, Luftfracht nur in S1, Gutquote fest 99 %, Montagestillstand nicht mehr
   Standardkriterium, robuste Empfehlung und Kombinationsabschnitt aus der Oberfläche entfernt, zwei Zielkonflikt-
   Diagramme mit frei wählbaren Achsen und Pareto-Markierung.

## Von der KI getroffene Modellentscheidungen (vom Studierenden zu prüfen)

- Abfahrtsintervalle, Umschlag- und Zollzeiten, Transportkostensätze, Gewichte, Preise außer Miyota: Annahmen [A]
- Lieferzeit in Werktagen statt Kalendertagen
- Sicherheitsbestand nach SS = z · sqrt(L · σd² + d² · σL²)
- TTR als Eingabe (Störungsdauer plus Transit), TTS aus der Simulation
- Robuste Empfehlung: zulässig nur, wenn β im schlechtesten Szenario den Mindestwert erreicht
- Bestandsbalken als Reichweite in Tagen statt Stück
- Kunden als zufällige deutsche Städte nach Einwohnergewicht, Paketlaufzeit nach Entfernung (Vorgabe N. Scharly: nur online, nur Deutschland)
- Werksausfall unterbricht die Fertigung (vorher lief sie weiter, nur der Versand war blockiert)
- Notfallplan Sellita: Reaktionszeit 3 Tage, Storno nicht versandter Miyota-Aufträge mit Gutschrift, Rückwechsel nach Wiederanlauf
- Bezugsoption Lederarmband Italien: Santa Croce sull'Arno, 28 € je Armband, 10 Tage Fertigung, 880 km Lkw [A]
- Gebundenes Kapital und Kapitalbindung je Uhr als eigene Spalten in der Optimierung
- Empfehlungslogik zweistufig: Mindest-Lieferfähigkeit, dann Priorität; ohne zulässige Option höchste Lieferfähigkeit
- Schwellen für Vor- und Nachteile (z. B. 0,5 Pp Lieferfähigkeit, 0,10 € je Uhr), darunter gilt eine Option als
  "nicht ausgelöst"
- Puffer-Option mit 1.000 Uhrwerken als einzelne Handlungsoption
- Brückenmenge = Bedarf bis zur ersten regulären Ankunft + Sicherheitsbestand - Lager - versandte Ware
- Eilzuschlag 25 % [A], Nachlauf 90 Tage [A], Auslöser 8 Tage (über Luft-WBZ von rund 6 bis 7 Tagen)
- Vorlauf Lkw Nagano-Narita 290 km [A], Flug NRT-FRA 9.350 km [A]
- Hybride Entscheidungsregel: Mindestanforderungen (Lieferzeit-Standard 3 Tage = Lieferzusage) plus
  Rank-Order-Centroid-Gewichte aus der Reihenfolge, Min-Max-Normierung über die zulässigen Optionen
- Bestandsbalken: Zustand kurz vor dem Bildzeitpunkt, ruhige Statusspalte statt Aufblinken, Strich kupferfarben
- Saisonbestand August bis November +20 Tage Bedarf [A] (Varianten mit 30 Tagen und Fertigwaren-Vorproduktion
  getestet; Vorproduktion allein erzeugte mehr Fehlteile, weil Komponenten früher verbraucht werden)
- Standard-Priorisierung: Lieferfähigkeit, Kosten, CO2e, Lieferzeit (Stillstand optional, Vorgabe N. Scharly)
- Gutquote fest mit Übertrag der Ausschuss-Bruchteile statt Runden (Runden ergäbe bei 25 Uhren am Tag 0 % Ausschuss)

## Vorgenommene Änderungen

[hier eigene Änderungen eintragen]

## Fachliche und technische Prüfungen

- 39 automatische Tests (`pytest -q`): Mengenbilanz, deterministischer Transit gegen Handrechnung,
  Reproduzierbarkeit, Common Random Numbers, Extremwerte, Umleitung bei Niedrigwasser, Bewertung, Optimierung je Szenario, Bestandsanimation
- [hier eigene Prüfungen eintragen, z. B. Parameter gegen Quellen geprüft, Ergebnisse plausibilisiert]
