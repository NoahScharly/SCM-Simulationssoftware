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

Grundkonzept, Netz und Daten stammen aus dem Poster von N. Scharly. Aus den beiden anderen Gruppenpostern wurde
die Logik übernommen, nicht die Zahlen:

| Erweiterung | Idee aus | Wo |
|---|---|---|
| Break-even-Wahrscheinlichkeit p* je Handlungsoption: p* = Mehrkosten im Normalbetrieb ÷ (Mehrkosten + Ersparnis im Störungsjahr). Die Software fragt keine Eintrittswahrscheinlichkeit ab, sondern zeigt, ab welcher sich die Vorsorge lohnt | Poster B (Entscheidungsregel p · vermiedener Schaden > Mehrkosten) | `handlungsoptionen.break_even`, Oberfläche Abschnitt 1 und 2 |
| Bandbreite je Szenario: die Stärke der Störung wird für mehrere Werte durchgerechnet, die Oberfläche zeigt, ob die Empfehlung kippt | Poster B (Szenarien mit Band) | Block `sensitivitaet` in jeder Szenariodatei, `handlungsoptionen.bandbreite`, Abschnitt 2 |
| Single Points of Failure: Elemente ohne Ausweichweg, auch mit allen Handlungsoptionen | Poster B (SPOF-Liste) | `netzanalyse.single_points_of_failure`, Oberfläche unter der Datenbasis |
| Stufe 0 der Entscheidung: harte Ausschlusskriterien (Länder, Standorte) vor jeder Bewertung | Poster C (erst Grenzen, dann Ziele) | Block `ausschluss` in `base.yaml`, Seitenleiste, `netzanalyse` |
| Szenario S5 Nachfrageeinbruch -40 % für 60 Tage: testet Überbestand und Kapitalbindung statt Fehlmengen | Poster C (Nachfrageschock) | `config/scenarios/s5_nachfrageeinbruch.yaml` |

Außerdem: Unterschiede unterhalb der Rauschschwelle (z. B. 0,03 Prozentpunkte Lieferfähigkeit) entscheiden die
Rangfolge nicht mehr, weil die Normierung mindestens die Schwelle als Spanne nutzt.

## Was das Modell abbildet

| Bereich | Umsetzung |
|---|---|
| Netz | Gerichteter Graph: 7 Tier-1-Lieferanten, 4 Tier-2, Häfen, Flughafen, Werk Heilbronn, Kunden |
| Transport | Seeschiff, Bahn, Binnenschiff, Lkw, Luftfracht (Notfall); Fahrplan je Kante, Umschlag, Zoll an der EU-Grenze |
| Nachschub | Bestellpunktverfahren (s, Q) je Komponente, Sicherheitsbestand aus Nachfrage- und Lieferzeitstreuung, Losgröße nach EOQ |
| Montage | Kapazität aus Jahresmenge abgeleitet (Mitarbeiter × 5 Uhren/Tag), Gutquote, Fertigwarenlager mit Zielreichweite |
| Nachfrage | Poisson-Prozess, Saisonfaktoren, Lieferzusage in Werktagen |
| Störungen | 5 generische Typen: `knoten_ausfall`, `kanten_sperre`, `korridor`, `nachfrage`, `kosten` |
| Vorsorge und Notfallpläne | Uhrwerk-Puffer (wirkt immer); Notfallplan Sellita: bei Ausfall von Miyota nach 3 Tagen Bestellung bei Sellita SW300-1 (La Chaux-de-Fonds), Storno nicht versandter Miyota-Aufträge, nach Wiederanlauf zurück; Notfallplan Luftfracht: nach Lieferantenausfall Brückenlieferung per Flugzeug beim Wiederanlauf (Eilfertigung 2 Tage, 25 % Eilzuschlag [A], Lkw Nagano-Narita, Flug nach Frankfurt), sonst bei Versorgungsstörung und Reichweite < 8 Tage; Bezugsoption Lederarmband komplett aus Italien (Santa Croce, Lkw über den Brenner, kein Zoll) |
| Kennzahlen | Kosten je Uhr, Lieferfähigkeit β, Lieferzeit (Mittel, 95-%-Quantil), CO2e je Uhr, TTS/TTR, Bestandsreichweite |
| Zufall | Getrennte Zufallsströme je Quelle (SeedSequence), gleiche Ströme in allen Szenarien (Common Random Numbers) |

### Einstellbare Parameter mit Plausibilitätsregeln

| Parameter | Bereich | Regel |
|---|---|---|
| Uhren pro Jahr | 0 bis 25.000 | Nachfragerate = Menge ÷ 365 |
| Montagen pro Tag | automatisch | Nötig = Menge ÷ (250 AT × Gutquote); Mitarbeiter = aufgerundet nötig ÷ 5; Kapazität = Mitarbeiter × 5 |
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
Mit `aenderungen:` lassen sich beliebige Werte aus `base.yaml` überschreiben.

## Wichtige Modellannahmen und Grenzen

- Werte mit `[A]` in den Konfigurationsdateien sind Annahmen und in der Sensitivitätsanalyse zu prüfen.
- Montagekapazität folgt der Jahresmenge: bei 5.000 Uhren/Jahr sind 20,2 Montagen/Tag nötig, also 5 Uhrmacher × 5 = 25 Montagen/Tag (Auslastung 81 %). Annahme 5 Uhren je Uhrmacher und Tag (Handarbeit mit zugekauften Werken: Einschalen, Zeiger, Armband, Endkontrolle), Sensitivität 2/3/5/7.
- Zollsatz überall 0 % (Freihandels- bzw. Partnerschaftsabkommen mit JP, CH, MU, Mercosur, EU-US-Zollverordnung 2026). Die Zollabfertigung (0,5 bis 2 Tage) bleibt, weil Anmeldung, Ursprungsnachweis und Einfuhrumsatzsteuer für Drittlandsware immer nötig sind.
- Fehlmengenkosten: 50 € je verspätetem Auftrag (High-End-Sammleruhr), Sensitivität 20/50/100 €.
- Optimierung: alle 12 Maßnahmenkombinationen je Szenario. Robuste Empfehlung = zulässig nur, wenn β auch im schlechtesten Szenario ≥ 95 %; Rang nach mittleren Kosten.
- Netzwerksicht: Weltkarte, Ausschnitt Europa und animierte Bestandsbalken (Reichweite in Tagen, Meldebestand, Bestellzeitpunkte), synchron und aus demselben Lauf. Goldener Ring und Statuskasten zeigen, welcher Lieferant das Uhrwerk gerade liefert. Abspielen in halber Geschwindigkeit (240 ms je Bild), ▶▶ doppelt so schnell.

## Ablauf in der Oberfläche

1. **Normalbetrieb:** Uhren pro Jahr. Lieferanten, Verkehrsmittel (Kosten, Dauer, CO2e) und Bestände kommen aus
   der Datenbasis in `config/`. Montagekapazität und Wiederbeschaffungszeiten folgen automatisch.
2. **Störungen wählen:** welche Szenarien neben der Basis simuliert werden (S1 bis S4).
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

Seed und Wiederholungen liegen unter "Experteneinstellungen". Standard ist der feste Seed aus `base.yaml`, damit
jeder Start dieselben Zahlen liefert.

Je Szenario werden nur die relevanten Optionen angezeigt (Szenariodatei, `relevante_optionen`); gerechnet werden
alle (für die Matrix und die Kommandozeile).

| Handlungsoption | Wirkung | Angezeigt in |
|---|---|---|
| Ohne Maßnahme | nur Bestellpunktverfahren; bei Sperre wartet die Ware | allen (Referenz) |
| Ausweichroute | Bahn statt Binnenschiff über Kaub, Bremerhaven statt Hamburg | S2, S3 |
| Saisonbestand | Meldebestand aller Komponenten August bis November +20 Tage Bedarf | S4 |
| Uhrwerk-Puffer +1.000 | mehr Sicherheitsbestand, bindet Kapital | S1, S2 |
| Notfallplan Sellita | Ersatz-Uhrwerk aus der Schweiz, doppelter Preis | S1 |
| Notfallplan Luftfracht | Brückenlieferung per Flugzeug mit Eilzuschlag, danach wieder Seeweg | S1 |
| Lederarmband aus Italien | Lkw statt Seeweg über Hamburg, teurer | S3 |

| Szenario | Was passiert | Puffer | Notfallplan Sellita | Notfallplan Luftfracht |
|---|---|---|---|---|
| S0 Basis | Normalbetrieb | wirkt | nicht ausgelöst | nur bei Fehlteil |
| S1 Uhrwerk-Ausfall | Miyota liefert ab Tag 120 für 60 Tage nicht | wirkt | **ausgelöst** (Tag 123 bis 180) | Uhrwerk nicht (Lieferant steht), andere Teile bei Bedarf |
| S2 Niedrigwasser | Binnenschiff über Kaub gesperrt, Umleitung Bahn | wirkt | nicht ausgelöst | nur bei Fehlteil |
| S3 Hamburg gestört | Hafen 14 Tage zu, Ausweichen Bremerhaven | wirkt | nicht ausgelöst | nur bei Fehlteil |
| S4 Saisonspitze | Nachfrage Nov/Dez +30 % | wirkt | nicht ausgelöst | nur bei Fehlteil |
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
