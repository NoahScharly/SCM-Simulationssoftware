# Simulationsmodell Supply Chain: High-End Cars-Uhr

Ereignisdiskrete Simulation (SimPy) einer globalen Lieferkette für eine High-End Cars-Automatikuhr (fiktive
Lizenzedition, 800 bis 1.200 €) mit Montage in Heilbronn.
Portfolioprüfung Logistik und Supply Chain Management, DHBW Heilbronn, Gruppe 04 Die Logistiker.

Das Modell setzt das Simulationskonzept (Prüfungselement 2) um: ein Modellkern, in den Netz, Mengen,
Kapazitäten und Szenarien als Daten geladen werden. Eine andere Netzvariante der Gruppe ist eine andere
Konfiguration, kein neuer Code.

## Schnellstart

Am besten in einer eigenen virtuellen Umgebung (Windows PowerShell; Mac/Linux: `source .venv/bin/activate`):

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py          # Entscheidungsunterstützung: Fall, Optionen, Empfehlung
python -m uhrensim -r 20          # Szenariovergleich in der Konsole, CSV in ergebnisse/
python -m uhrensim --optionen                 # je Szenario alle Handlungsoptionen + Empfehlung
python -m uhrensim --optimierung              # alle Szenarien × alle Maßnahmen-Kombinationen
python -m uhrensim --sensitivitaet -s S1      # One-at-a-time aus base.yaml
pytest -q                         # Verifikationstests
```

Python 3.11 oder neuer.

## Version 2: Erweiterungen aus den Gruppenpostern

Grundkonzept, Netz und Daten stammen aus dem Poster von N. Scharly. Aus den Postern und Simulationskonzepten der
übrigen Gruppenmitglieder (B = F. Lautenbach, C = P. Maywald, D = B. Hufgard) wurde die Logik übernommen und nur
dort eine Zahl, wo sie belegt ist:

| Erweiterung | Idee aus | Wo |
|---|---|---|
| Break-even-Wahrscheinlichkeit p* je Handlungsoption: p* = Mehrkosten im Normalbetrieb ÷ (Mehrkosten + Ersparnis im Störungsjahr). Die Software fragt keine Eintrittswahrscheinlichkeit ab, sondern zeigt, ab welcher sich die Vorsorge lohnt | Poster B (Entscheidungsregel p · vermiedener Schaden > Mehrkosten) | `handlungsoptionen.break_even`, Oberfläche Abschnitt 1 und 2 |
| Bandbreite je Szenario: die Stärke der Störung wird für mehrere Werte durchgerechnet, die Oberfläche zeigt, ob die Empfehlung kippt | Poster B (Szenarien mit Band) | Block `sensitivitaet` in jeder Szenariodatei, `handlungsoptionen.bandbreite`, Abschnitt 2 |
| Single Points of Failure: Elemente ohne Ausweichweg, auch mit allen Handlungsoptionen | Poster B (SPOF-Liste) | `netzanalyse.single_points_of_failure`, Oberfläche unter der Datenbasis |
| Szenario S5 Nachfrageeinbruch -30 % für 60 Tage nach einem Imageschaden, Bandbreite -15/-30/-50 %: testet Überbestand, Kapitalbindung und ungenutztes Personal statt Fehlmengen; Handlungsoptionen Bestand an Nachfrage anpassen und Kurzarbeit | Poster C (Nachfrageschock, Skandalstufen) | `config/scenarios/s5_nachfrageeinbruch.yaml`, `massnahmen.bestand_anpassen`, `massnahmen.kurzarbeit` |
| Montagepersonal als Fixkosten: Stellen × 1.680 h × 49,50 €/h (Destatis 2026), fällig auch ohne Montage, Stellen auf halbe Stellen aufgerundet (4,5 bei 5.000 Uhren). Erst dadurch kostet ein Nachfrageeinbruch spürbar | Konzept C (Lohnkosten, 140 h je Monat) | `kosten.montage_kostenmodell`, `simulation._montage` |
| Handlungsoption Aushilfe: liegt die Montage mehr als 5 Tagesbedarfe hinter dem Plan und sind Teile da, kommt Zeitarbeit dazu | Konzept C (Aushilfe bei Rückstand) | `massnahmen.aushilfe`, Optionen in S4 |
| Stresstest: alle Szenarien × alle Optionen × 500 und 25.000 Uhren, Mengenbilanz je Komponente geht auf | Konzept C (Stresstest, Mengenbilanz) | `tests/test_modell.py` |
| Reservequelle realistisch: Sellita fertigt erst nach 14 Tagen Hochlauf, höchstens 150 Stück je Woche, und der Notfallplan kostet 4.000 € im Jahr Bereitschaft, auch ohne Ausfall | Konzept D (Reserve mit Hochlauf, Kapazität, Freigabekosten) | `massnahmen.notfall_*`, `simulation._reserve_ende` |
| Zufälliger Startbestand je Replikation: sonst trifft eine Störung bei Losgrößen von 100 bis 200 Tagen immer dieselbe Stelle im Bestellzyklus oder gar keine Sendung | Konzept D (fester Starttag fällt zwischen die Lose) | `simulation.Simulation.__init__` |
| Lagerhaltungskostensatz 15 % p. a. (8 % Zins + 7 % Versicherung, Schwund, Veralterung) | Konzepte C und D (beide 15 %) | `kosten.kapitalzins_pa`, `kosten.lagerrisiko_pa` |
| Strenge Feldnamen: Tippfehler in Szenario oder Eingabe bricht das Laden mit Vorschlag ab | Konzept C (Datenklassen statt loser Wörterbücher) | `data.pruefe_schluessel` |
| Konfigurations-Hash, Version und Seed im Ereignis `laufstart` jedes Protokolls | Konzept D (Reproduzierbarkeit) | `data.konfig_hash`, `Simulation.lauf` |
| CO2e getrennt nach Beschaffung (Inbound) und Zustellung (letzte Meile) | Konzept B (getrennte Minimierung) | `evaluation.kennzahlen` |

Außerdem: Unterschiede unterhalb einer Relevanzschwelle (Kosten 1 €/Uhr, Lieferfähigkeit 0,5 Prozentpunkte, CO2e
5 g/Uhr, Lieferzeit 0,5 Tage) gelten als gleichwertig und entscheiden die Rangfolge nicht. Bei Gleichstand
entscheiden die Kosten, dann die übrigen Kriterien, nie die Reihenfolge der Liste.

## Was das Modell abbildet

| Bereich | Umsetzung |
|---|---|
| Netz | Gerichteter Graph: 6 Tier-1-Lieferanten (Armband mit Schließe aus einer Hand) und Sellita als Notfallplan, 4 Tier-2, Häfen, Flughafen, Werk Heilbronn, Kunden |
| Transport | Seeschiff, Bahn, Binnenschiff, Lkw, Luftfracht (Notfall); Fahrplan je Kante, Umschlag, Zoll an der EU-Grenze |
| Nachschub | Bestellpunktverfahren (s, Q) je Komponente, Sicherheitsbestand aus Nachfrage- und Lieferzeitstreuung, Losgröße nach EOQ |
| Montage | Kapazität aus Jahresmenge abgeleitet (Mitarbeiter × 5 Uhren/Tag), Gutquote, Fertigwarenlager mit Zielreichweite |
| Nachfrage | Poisson-Prozess, Saisonfaktoren, Lieferzusage in Werktagen |
| Störungen | 5 generische Typen: `knoten_ausfall`, `kanten_sperre`, `korridor`, `nachfrage`, `kosten` |
| Vorsorge und Notfallpläne | Uhrwerk-Puffer (wirkt immer); Notfallplan Sellita: bei Ausfall von Miyota nach 3 Tagen Bestellung bei Sellita SW300-1 (La Chaux-de-Fonds), Storno nicht versandter Miyota-Aufträge, nach Wiederanlauf zurück; Notfallplan Luftfracht: nach Lieferantenausfall Brückenlieferung per Flugzeug beim Wiederanlauf (Eilfertigung 2 Tage, 25 % Eilzuschlag [A], Lkw Nagano-Narita, Flug nach Frankfurt), sonst bei Versorgungsstörung und Reichweite < 8 Tage |
| Kennzahlen | Kosten je Uhr, Lieferfähigkeit β, Lieferzeit (Mittel, 95-%-Quantil), CO2e je Uhr, TTS/TTR, Bestandsreichweite |
| Zufall | Getrennte Zufallsströme je Quelle (SeedSequence), gleiche Ströme in allen Szenarien (Common Random Numbers) |

### Einstellbare Parameter mit Plausibilitätsregeln

| Parameter | Bereich | Regel |
|---|---|---|
| Uhren pro Jahr | 0 bis 25.000 | Nachfragerate = Menge ÷ 365 |
| Montagen pro Tag | automatisch | Nötig = Menge ÷ (250 AT × Gutquote); Stellen = nötig ÷ 5, auf halbe Stellen aufgerundet; Kapazität = Stellen × 5 |
| Wiederbeschaffungszeit | automatisch | Fertigung + Transit + Umschlag + Zoll der besten Route; dauerhafte Routenänderungen eines Szenarios werden übernommen |

## Ordnerstruktur

```
config/
  nodes.csv          Knoten mit Koordinaten, Typ, Stufe, Zollgrenze
  edges.csv          Kanten mit Verkehrsträger, km, Abfahrtsintervall, CO2e-Faktor, Wegpunkte
  routes.csv         Routen je Lieferant mit Priorität (regulär, Ausweich, Notfall)
  components.csv     Komponenten mit Gewicht und Zollsatz
  suppliers.csv      Bezugsquellen mit Preis, Fertigungszeit, Tier-2
  base.yaml          alle Parameter des Basisszenarios
  scenarios/*.yaml   Szenarien als Änderung gegenüber der Basis
uhrensim/
  domain.py          fachliche Objekte (Knoten, Kante, Route, Bezugsquelle, Störung, Modell)
  data.py            Laden, Zusammenführen, Referenz- und Plausibilitätsprüfung, Mindest-WBZ
  simulation.py      SimPy-Prozesse, Ereignisprotokoll
  evaluation.py      Kennzahlen, Resilienz, Pareto-Front, Bewertung
  experiment.py      Replikationen, Konfidenzintervalle, Szenariovergleich, Sensitivität
  handlungsoptionen.py  Optionen je Szenario vergleichen, Vor- und Nachteile, Empfehlung je Szenario
  optimization.py    Rastersuche über Maßnahmen-Kombinationen (nur Kommandozeile)
  visualization.py   animierte Karte, Diagramme, Kennzahlentabelle (Plotly)
  __main__.py        Kommandozeile
app/app.py           Streamlit-Oberfläche
tests/               Verifikationstests (pytest)
```

## Schnittstellen

```python
from uhrensim.data import lade_modell
from uhrensim.simulation import simuliere
from uhrensim.evaluation import kennzahlen
from uhrensim.experiment import fuehre_aus, vergleiche, sensitivitaet, szenario_pfad

modell = lade_modell(szenario_pfad("S1"), {"nachfrage": {"uhren_pro_jahr": 8000}})
erg = simuliere(modell, replikation=0)        # Ereignisprotokoll in erg.log
print(kennzahlen(erg))
print(vergleiche(replikationen=20)["tabelle"])
print(sensitivitaet(szenario_pfad("S1"), "stoerungen.0.dauer_tage", [30, 60, 90], im_szenario=True))
```

## Neues Szenario anlegen

Datei in `config/scenarios/` anlegen, z. B. `s5_zoll.yaml`:

```yaml
id: S5
name: Zollerhöhung Lederarmband
stoerungen:
  - typ: kosten
    parameter: zoll:armband
    wert: 0.25
    start_tag: 0
```

Weitere Beispiele: `knoten_ausfall` mit `ziel` und `dauer_tage`, `kanten_sperre` mit Kanten-ID,
`korridor` mit `aktivieren`/`deaktivieren`, `nachfrage` mit `faktor` und `monate` oder `start_tag`/`dauer_tage`.
Mit `aenderungen:` lassen sich beliebige Werte aus `base.yaml` überschreiben. Ein Tippfehler in einem Feldnamen
bricht das Laden mit Hinweis ab.

## Wichtige Modellannahmen und Grenzen

- Werte mit `[A]` in den Konfigurationsdateien sind Annahmen und in der Sensitivitätsanalyse zu prüfen.
- Montagekapazität folgt der Jahresmenge: bei 5.000 Uhren/Jahr sind 20,2 Montagen/Tag nötig, also 4,04 Stellen, besetzt mit 4,5 Stellen (halbe Stellen möglich) × 5 = 22,5 Montagen/Tag (Auslastung 90 %). Montagekosten damit rund 76 € je Uhr (4,5 × 1.680 h × 49,50 €). Mit `montage.stellen_raster: 1.0` nur volle Stellen. Annahme 5 Uhren je Uhrmacher und Tag (Handarbeit mit zugekauften Werken: Einschalen, Zeiger, Armband, Endkontrolle), Sensitivität 2/3/5/7.
- Zollsatz überall 0 % (Freihandels- bzw. Partnerschaftsabkommen mit JP, CH, MU, Mercosur, EU-US-Zollverordnung 2026). Die Zollabfertigung (0,5 bis 2 Tage) bleibt, weil Anmeldung, Ursprungsnachweis und Einfuhrumsatzsteuer für Drittlandsware immer nötig sind.
- Fehlmengenkosten: 50 € je verspätetem Auftrag (High-End-Sammleruhr), Sensitivität 20/50/100 €.
- Optimierung: alle 12 Maßnahmenkombinationen je Szenario. Robuste Empfehlung = zulässig nur, wenn β auch im schlechtesten Szenario ≥ 95 %; Rang nach mittleren Kosten.
- Stillstand mit Kundenwirkung: Arbeitstage, an denen ein Teil fehlt und dadurch Bestellungen warten oder der normale Fertigwarenvorrat unterschritten ist; Fehlteile beim Vorproduzieren zählen nicht.
- Netzwerksicht: Weltkarte, Ausschnitt Europa und animierte Bestandsbalken (Reichweite in Tagen, Meldebestand, Bestellzeitpunkte), synchron und aus demselben Lauf. Goldener Ring und Statuskasten zeigen, welcher Lieferant das Uhrwerk gerade liefert. Abspielen im Standardtempo (240 ms je Bild), Auswahlfeld für ½× und 2×. Die Bestandsbalken stehen über die ganze Breite unter den Karten, damit die Teilenamen vollständig lesbar sind.

## Ablauf in der Oberfläche

Die Hauptseite hat drei Reiter: **Überblick** (alle Szenarien: Empfehlung je Szenario, was die Priorität ändert, wie
sicher die Ergebnisse sind), **Szenario im Detail** (ein gewähltes Szenario mit einer Option: Karte, Kennzahlen,
Optionsvergleich, Bandbreite, Zielkonflikte) und **Netz und Daten** (Datenbasis, Single Points of Failure). Jede
Überschrift trägt eine Markierung, worauf sie sich bezieht: grau „alle Szenarien“, rot das gewählte Szenario.

1. **Normalbetrieb:** Uhren pro Jahr. Lieferanten, Verkehrsmittel (Kosten, Dauer, CO2e) und Bestände kommen aus
   der Datenbasis in `config/`. Montagekapazität und Wiederbeschaffungszeiten folgen automatisch.
2. **Störungen wählen:** welche Szenarien neben der Basis simuliert werden (S1 bis S5).
3. **Automatischer Optionsvergleich:** Nach dem Start wird jedes Szenario mit jeder Handlungsoption aus
   `base.yaml` (Block `handlungsoptionen`) simuliert, alle mit denselben Zufallszahlen. Neue Optionen werden dort
   ohne Programmierung ergänzt.
4. **Fall ansehen:** Szenario wählen, mit dem Schalter zwischen den Optionen wechseln. Karte, Bestandsbalken und
   Kennzahlen zeigen die gewählte Option, die Differenzen beziehen sich auf "Ohne Maßnahme".
5. **Empfehlung:** zweistufig und offengelegt. Stufe 1: Mindestanforderungen (Standard Lieferfähigkeit
   mindestens 95 %, Lieferzeit 95 % höchstens 3 Tage, beide abschaltbar und einstellbar). Stufe 2: Priorisierung,
   Zielkriterien auswählen und per Drag-and-drop ordnen; aus der Reihenfolge werden Rank-Order-Centroid-Gewichte
   (Barron und Barrett 1996), z. B. bei vier Kriterien 52 / 27 / 15 / 6 %. Standardwerte in `base.yaml`, Block
   `entscheidung`. Ergebnis: eine Empfehlung je Szenario. Zielkonflikte in zwei Diagrammen mit frei wählbaren
   Achsen (Standard Kosten gegen Lieferfähigkeit und Kosten gegen CO2e), Pareto-optimale Optionen grün.
   Zielkriterien: Lieferfähigkeit, Kosten, CO2e, Lieferzeit und optional gebundenes Kapital (Liquidität).
   Montagestillstand ist nur Kennzahl, weil seine Folgen schon in Lieferfähigkeit und Kosten stecken.
6. **Was ändert die Priorität?** Unter dem Gesamtbild: Empfehlung je Szenario, wenn Lieferfähigkeit, Kosten,
   CO2e oder gebundenes Kapital an erster Stelle stehen. Zeigt, in welchen Szenarien die Priorität die Entscheidung kippt.

Seed und Wiederholungen liegen unter "Experteneinstellungen". Standard ist der feste Seed aus `base.yaml`, damit
jeder Start dieselben Zahlen liefert.

Je Szenario werden nur die relevanten Optionen gerechnet und angezeigt (Szenariodatei, `relevante_optionen`), im
Normalbetrieb S0 alle, weil p* die Mehrkosten jeder Option im Normalbetrieb braucht. Die Läufe verteilen sich auf
alle Prozessorkerne (höchstens 8); jeder Lauf hat seinen festen Seed, deshalb sind die Zahlen dieselben wie
nacheinander gerechnet. Die Kommandozeile rechnet weiterhin alle Kombinationen.

| Handlungsoption | Wirkung | Angezeigt in |
|---|---|---|
| Ohne Maßnahme | nur Bestellpunktverfahren; bei Sperre wartet die Ware | allen (Referenz) |
| Ausweichroute | Bahn statt Binnenschiff über Kaub, Bremerhaven statt Hamburg; Schiffe, die schon unterwegs sind, laufen Bremerhaven an (150 km Umweg, 0,5 Tage) | S2, S3 |
| Vorproduktion | Juli bis Dezember fertige Uhren für 40 statt 10 Tage Bedarf auf Lager, nutzt freie Montagekapazität vor der Spitze, bindet Kapital | S4 |
| Teilzeit aufstocken | die halbe Stelle geht bei Montagerückstand auf Vollzeit, ohne Aufschlag | S4 |
| Aushilfe | Zeitarbeit bei Montagerückstand, 30 % teurer | S4 |
| Bestand an Nachfrage anpassen | Meldebestand und Bestellmengen sinken mit der erkannten Nachfrage (Reaktion 7 Tage) | S5 |
| Kurzarbeit | Montage arbeitet nur entsprechend der Nachfrage, 25 % Remanenzkosten (Reaktion 14 Tage) | S5 |
| Uhrwerk-Puffer +500 / +750 / +1.000 | mehr Sicherheitsbestand, bindet Kapital; drei Stufen zeigen, wie viel Puffer genug ist (S1: 500 knapp, 750 reicht, 1.000 Überversicherung) | S1, S2 |
| Notfallplan Sellita | Ersatz-Uhrwerk aus der Schweiz, doppelter Preis | S1 |
| Notfallplan Luftfracht | Brückenlieferung per Flugzeug mit Eilzuschlag, danach wieder Seeweg | S1, S3 |

| Szenario | Was passiert | Puffer | Notfallplan Sellita | Notfallplan Luftfracht |
|---|---|---|---|---|
| S0 Basis | Normalbetrieb | wirkt | nicht ausgelöst | nur bei Fehlteil |
| S1 Uhrwerk-Ausfall | Miyota liefert ab Tag 120 für 60 Tage nicht | wirkt | **ausgelöst** (Tag 123 bis 180) | Uhrwerk nicht (Lieferant steht), andere Teile bei Bedarf |
| S2 Niedrigwasser | Binnenschiff über Kaub gesperrt, Umleitung Bahn | wirkt | nicht ausgelöst | nur bei Fehlteil |
| S3 Hamburg gestört | Hafen 21 Tage zu, Ausweichen Bremerhaven | wirkt | nicht ausgelöst | nur bei Fehlteil |
| S4 Saisonspitze | Nachfrage Nov/Dez +50 % | wirkt | nicht ausgelöst | nur bei Fehlteil |
- Einkauf wird als Wareneinsatz bewertet (verbaute Komponenten × Einkaufspreis), nicht nach Bestelldatum.
- Emissionsfaktoren: GLEC-Standardwerte (Well-to-Wheel), konservativ, keine Primärdaten.
- TTR ist eine Eingabe (Störungsdauer, bei Lieferanten plus Transit der ersten neuen Sendung), TTS wird simuliert.
- Ausfälle sind unabhängig; keine Retouren, kein Bullwhip-Effekt bis Tier 2, Kapitalbindung nur auf Lagerbestand.
- Lieferzeit in Werktagen (`distribution.lieferzeit_einheit`), auf Kalendertage umstellbar.

**Kapitalbindung im Vorschlag:** Der Puffer ist Vermögen, kein Aufwand. In die Kosten je Uhr gehen die Zinsen auf den
durchschnittlich gebundenen Bestandswert ein (`kosten.kapitalzins_pa`). Die Optimierung zeigt zusätzlich
"davon Kapitalbindung [€/Uhr]" und "Gebundenes Kapital Ø [€]", damit sichtbar ist, was eine Empfehlung an Kapital
bindet. Beispiel: +1.000 Uhrwerke binden rund 70.000 € mehr und kosten bei 10 % rund 1,40 € je Uhr.

## Kennzeichnung der Urheberschaft (Aufgabenstellung 6.2)

| Datei / Modul | Kennzeichen | Bemerkung |
|---|---|---|
| `config/*` | E | Daten und Quellen selbst recherchiert (Poster N. Scharly), Annahmen mit [A] markiert |
| `uhrensim/*.py`, `app/app.py`, `tests/*` | G | Architektur und Fachlogik selbst erarbeitet, Code teilweise selbst, teilweise mit Claude Opus 5.5 umgesetzt, Fehlerbehebung mit KI, siehe `KI_NACHWEIS.md` |
| SimPy, NumPy, pandas, Plotly, Streamlit, PyYAML | F | Fremdbibliotheken, siehe `requirements.txt` |

## Kunden und Distribution

- Nur Onlineverkauf, Versand per Paket direkt ab Heilbronn (eine Distributionsstufe).
- Jede Bestellung geht an eine zufällige Stadt aus `config/kunden.csv` (44 deutsche Großstädte), gewichtet nach Einwohnern; die Stadt steht stellvertretend für ihr Umland.
- Paketlaufzeit nach Luftlinie ab Heilbronn (`distribution.versand_zonen` in `base.yaml`): bis 250 km 1 bis 1,5 Tage, bis 450 km 1 bis 2 Tage, darüber 1 bis 2,5 Tage (Dreieck).
- CO2e je Paket: Hauptlauf Lkw (Straßen-km = Luftlinie × 1,3, GLEC Sattelzug 101 g/tkm) plus 30 km letzte Meile (Transporter 840 g/tkm).
- Karte: Kundenorte als Kreise (Größe = Anteil), Pakete als blaue Quadrate von Heilbronn zum Kunden.
