# Simulationssoftware SCM

Ereignisdiskrete Simulation (SimPy) der globalen Supply Chain einer High-End Cars-Automatikuhr (fiktive
Lizenzedition, 800 bis 1.200 €) mit Montage in Heilbronn. Die Software simuliert Normalbetrieb und Störungen,
rechnet für jede Störung mehrere Handlungsoptionen durch und leitet daraus eine nachvollziehbare Empfehlung ab.

Portfolioprüfung Logistik und Supply Chain Management, DHBW Heilbronn, Gruppe 04 „Die Logistiker“:
Ferdinand Lautenbach, Noah Scharly, Philipp Maywald, Benedict Hufgard.

**Inhalt**

1. [Startvoraussetzungen](#1-startvoraussetzungen)
2. [Dateistruktur](#2-dateistruktur)
3. [Ausführungshinweise](#3-ausführungshinweise)
4. [Modell im Überblick](#4-modell-im-überblick)
5. [Annahmen und Grenzen](#5-annahmen-und-grenzen)
6. [Zuordnung der persönlichen Beiträge](#6-zuordnung-der-persönlichen-beiträge)

---

## 1. Startvoraussetzungen

| Voraussetzung | Details |
|---|---|
| Python | Version 3.11 oder neuer |
| Betriebssystem | Windows, macOS oder Linux |
| Bibliotheken | `simpy`, `numpy`, `pandas`, `pyyaml`, `networkx`, `plotly`, `streamlit`, `streamlit-sortables`, `pytest` (siehe `requirements.txt`) |
| Browser | für die Oberfläche, öffnet sich beim Start automatisch |
| Internet | für die Installation der Bibliotheken |

**Installation** (einmalig, am besten in einer eigenen virtuellen Umgebung):

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
source .venv/bin/activate           # macOS und Linux
pip install -r requirements.txt
```

---

## 2. Dateistruktur

```
SCM-Simulationssoftware/
├── app/
│   └── app.py                 Streamlit-Oberfläche (Entscheidungsunterstützung)
├── config/                    Datenbasis: alles, was das Modell rechnet, steht hier, nicht im Code
│   ├── nodes.csv              Knoten: Lieferanten, Häfen, Flughäfen, Werk, mit Koordinaten und Zollgrenze
│   ├── edges.csv              Kanten: Verkehrsträger, km, Abfahrtsintervall, CO2e-Faktor, Wegpunkte
│   ├── routes.csv             Routen je Lieferant mit Priorität (regulär, Ausweich, Notfall)
│   ├── components.csv         Komponenten mit Gewicht und Zollsatz
│   ├── suppliers.csv          Bezugsquellen mit Preis, Fertigungszeit, Tier-2
│   ├── kunden.csv             44 deutsche Großstädte mit Einwohnerzahl
│   ├── base.yaml              alle Parameter des Normalbetriebs, Handlungsoptionen, Entscheidungsregel
│   └── scenarios/             Szenarien S0 bis S5 als Änderung gegenüber base.yaml
├── uhrensim/                  Modellkern (Python-Paket)
│   ├── __init__.py            Paket, Versionsnummer
│   ├── __main__.py            Kommandozeile
│   ├── domain.py              fachliche Datenklassen: Knoten, Kante, Route, Bezugsquelle, Störung, Modell
│   ├── data.py                Laden, Zusammenführen und Prüfen der Konfiguration
│   ├── simulation.py          SimPy-Prozesse und Ereignisprotokoll
│   ├── evaluation.py          Kennzahlen, Resilienz, Pareto-Front
│   ├── experiment.py          Replikationen, Konfidenzintervalle, Szenariovergleich, Sensitivität
│   ├── handlungsoptionen.py   Optionsvergleich, Rangfolge, Empfehlung, Break-even p*, Bandbreite
│   ├── netzanalyse.py         Single Points of Failure
│   ├── optimization.py        Rastersuche über Maßnahmen-Kombinationen
│   └── visualization.py       animierte Karte, Bestandsbalken, Diagramme, Tabellen
├── tests/                     Verifikationstests (pytest)
│   ├── test_modell.py
│   ├── test_handlungsoptionen.py
│   └── test_erweiterungen.py
├── .streamlit/config.toml     Farbschema der Oberfläche
├── requirements.txt           benötigte Bibliotheken
├── pytest.ini                 Testeinstellungen
├── KI_NACHWEIS.md             Nachweis der KI-Nutzung je Person
└── README.md
```

Ergebnisse der Kommandozeile werden als CSV im Ordner `ergebnisse/` abgelegt (wird beim ersten Lauf angelegt).

---

## 3. Ausführungshinweise

Alle Befehle werden im Hauptordner des Projekts ausgeführt.

### 3.1 Oberfläche starten

```bash
streamlit run app/app.py
```

**Ablauf in der Oberfläche**

| Schritt | Wo | Was passiert |
|---|---|---|
| 1. Normalbetrieb festlegen | Seitenleiste | Uhren pro Jahr wählen (0 bis 25.000). Montagekapazität, Stellen und Wiederbeschaffungszeiten werden automatisch abgeleitet |
| 2. Störungen wählen | Seitenleiste | Szenarien S1 bis S5 auswählen, die neben der Basis gerechnet werden |
| 3. Rechnen | Seitenleiste | Jedes Szenario wird mit jeder relevanten Handlungsoption simuliert, alle mit denselben Zufallszahlen und parallel auf bis zu 8 Kernen |
| 4. Überblick ansehen | Reiter „Überblick“ | Empfehlung je Szenario, Wirkung der Priorität, Sicherheit der Ergebnisse |
| 5. Fall im Detail | Reiter „Szenario im Detail“ | Karte, Bestandsbalken, Kennzahlen, Optionsvergleich, Bandbreite, Zielkonflikte; Schalter zum Wechseln der Option |
| 6. Daten prüfen | Reiter „Netz und Daten“ | Datenbasis, Lieferanten, Verkehrsmittel, Single Points of Failure |

**Empfehlung einstellen:** Stufe 1 sind Mindestanforderungen (Standard Lieferfähigkeit mindestens 95 %,
Lieferzeit 95 % höchstens 3 Werktage, beide abschaltbar). Stufe 2 ist die Priorisierung: Zielkriterien per
Drag-and-drop ordnen, daraus werden Rank-Order-Centroid-Gewichte (Barron und Barrett 1996).

**Seed und Wiederholungen** liegen unter „Experteneinstellungen“. Standard ist der feste Seed aus `base.yaml`, damit
jeder Start dieselben Zahlen liefert.

**Markierung der Überschriften:** grau bezieht sich auf alle Szenarien, rot auf das gewählte Szenario.

### 3.2 Kommandozeile

```bash
python -m uhrensim                        # Szenariovergleich aller Szenarien
python -m uhrensim -s S0 S1 -r 20         # nur S0 und S1, 20 Replikationen
python -m uhrensim --optionen             # je Szenario alle Handlungsoptionen und Empfehlung
python -m uhrensim --optimierung          # alle Szenarien × alle Maßnahmen-Kombinationen
python -m uhrensim --sensitivitaet -s S1  # One-at-a-time-Sensitivität aus base.yaml
```

| Option | Bedeutung |
|---|---|
| `-s`, `--szenarien` | Szenario-IDs, z. B. `S0 S1` (Standard: alle) |
| `-r`, `--replikationen` | Anzahl Monte-Carlo-Läufe je Szenario |
| `--menge` | Uhren pro Jahr (0 bis 25.000) |
| `--kapazitaet` | feste Montagen pro Tag (1 bis 150) statt automatischer Auslegung |
| `--optionen` | Handlungsoptionen je Szenario vergleichen |
| `--optimierung` | Rastersuche über Maßnahmen-Kombinationen |
| `--sensitivitaet` | Sensitivitätsanalyse nach dem Block `sensitivitaet` in `base.yaml` |
| `-o`, `--ausgabe` | Ordner für die CSV-Ergebnisse (Standard `ergebnisse`) |

### 3.3 Tests ausführen

```bash
pytest -q
```

64 Tests prüfen unter anderem Konfiguration, Mengenbilanz, Handrechnung, Reproduzierbarkeit, Störungen,
Notfallpläne, Entscheidungslogik und einen Stresstest über alle Szenarien und Optionen. Laufzeit rund 2 Minuten.

### 3.4 Aus Python nutzen

```python
from uhrensim.data import lade_modell
from uhrensim.simulation import simuliere
from uhrensim.evaluation import kennzahlen
from uhrensim.experiment import vergleiche, sensitivitaet, szenario_pfad

modell = lade_modell(szenario_pfad("S1"), {"nachfrage": {"uhren_pro_jahr": 8000}})
erg = simuliere(modell, replikation=0)          # Ereignisprotokoll in erg.log
print(kennzahlen(erg))
print(vergleiche(replikationen=20)["tabelle"])
print(sensitivitaet(szenario_pfad("S1"), "stoerungen.0.dauer_tage", [30, 60, 90], im_szenario=True))
```

### 3.5 Eigenes Szenario oder eigene Option anlegen

Neue Szenarien sind eine Datei in `config/scenarios/`, kein neuer Code:

```yaml
id: S6
name: Zollerhöhung Lederarmband
relevante_optionen: [keine]
stoerungen:
  - typ: kosten
    parameter: zoll:armband
    wert: 0.25
    start_tag: 0
```

Störungstypen: `knoten_ausfall` (mit `ziel`, `dauer_tage`), `kanten_sperre` (Kanten-ID), `korridor`
(`aktivieren`/`deaktivieren`), `nachfrage` (`faktor` mit `monate` oder `start_tag`/`dauer_tage`), `kosten`.
Mit `aenderungen:` lässt sich jeder Wert aus `base.yaml` überschreiben. Neue Handlungsoptionen werden im Block
`handlungsoptionen` in `base.yaml` ergänzt. Ein Tippfehler in einem Feldnamen bricht das Laden mit einem
Korrekturvorschlag ab.

---

## 4. Modell im Überblick

| Bereich | Umsetzung |
|---|---|
| Netz | gerichteter Graph: 6 Tier-1-Lieferanten, Sellita als Notfallplan, 4 Tier-2, Häfen, Flughafen, Werk Heilbronn, Kunden |
| Transport | Seeschiff, Bahn, Binnenschiff, Lkw, Luftfracht (Notfall); Fahrplan je Kante, Umschlag, Zoll an der EU-Grenze |
| Nachschub | Bestellpunktverfahren (s, Q) je Komponente, Sicherheitsbestand aus Nachfrage- und Lieferzeitstreuung, Losgröße nach EOQ |
| Montage | Kapazität aus der Jahresmenge (Stellen × 5 Uhren je Tag, halbe Stellen möglich), Gutquote, Fertigwarenlager mit Zielreichweite, Personal als Fixkosten |
| Nachfrage | Poisson-Prozess, Saisonfaktoren, Kunden in 44 Großstädten gewichtet nach Einwohnern, Lieferzusage in Werktagen |
| Kennzahlen | Kosten je Uhr, Lieferfähigkeit β, Lieferzeit (Mittel, 95-%-Quantil), CO2e getrennt nach Beschaffung und letzter Meile, TTS/TTR, gebundenes Kapital |
| Zufall | getrennte Zufallsströme je Quelle, gleiche Ströme in allen Szenarien (Common Random Numbers), zufälliger Startbestand je Replikation |
| Reproduzierbarkeit | Konfigurations-Hash, Version und Seed im Ereignis `laufstart` jedes Protokolls |

### Szenarien

| Szenario | Was passiert | Bandbreite |
|---|---|---|
| S0 Basis | Normalbetrieb | |
| S1 Uhrwerk-Ausfall | Miyota liefert ab Tag 120 für 60 Tage nicht | 15 / 30 / 60 / 90 Tage |
| S2 Niedrigwasser | Binnenschiff über Kaub ab Tag 200 für 50 Tage gesperrt | 20 / 50 / 80 Tage |
| S3 Hamburg gestört | Hafen ab Tag 60 für 21 Tage geschlossen | 14 / 21 / 28 Tage |
| S4 Saisonspitze | Nachfrage November und Dezember +50 % | +30 / +50 / +70 % |
| S5 Nachfrageeinbruch | Nachfrage ab Tag 150 für 60 Tage −30 % nach Imageschaden | −15 / −30 / −50 % |

### Handlungsoptionen

| Option | Wirkung | Szenarien |
|---|---|---|
| Ohne Maßnahme | nur Bestellpunktverfahren; bei Sperre wartet die Ware | alle (Referenz) |
| Ausweichroute | Bahn statt Binnenschiff, Bremerhaven statt Hamburg; Schiffe auf See werden umgeleitet | S2, S3 |
| Uhrwerk-Puffer +500 / +750 / +1.000 | mehr Sicherheitsbestand, bindet Kapital | S1, S2 |
| Notfallplan Sellita | Ersatz-Uhrwerk aus der Schweiz zum doppelten Preis, 14 Tage Hochlauf, höchstens 150 Stück je Woche, 4.000 € Bereitschaft im Jahr | S1 |
| Notfallplan Luftfracht | Brückenlieferung per Flugzeug mit Eilzuschlag, danach wieder Seeweg | S1, S3 |
| Vorproduktion | Juli bis Dezember Fertigwaren für 40 statt 10 Tage | S4 |
| Teilzeit aufstocken | halbe Stelle bei Montagerückstand auf Vollzeit | S4 |
| Aushilfe | Zeitarbeit bei Montagerückstand, 30 % teurer | S4 |
| Bestand an Nachfrage anpassen | Meldebestand und Bestellmengen folgen der Nachfrage (Reaktion 7 Tage) | S5 |
| Kurzarbeit | Montage nach Nachfrage, 25 % Remanenzkosten (Reaktion 14 Tage) | S5 |

### Entscheidungsunterstützung

- **Break-even-Wahrscheinlichkeit p\*** je Option: Mehrkosten im Normalbetrieb ÷ (Mehrkosten + Ersparnis im
  Störungsjahr). Zeigt, ab welcher Eintrittswahrscheinlichkeit sich eine Vorsorge lohnt.
- **Bandbreite:** Jedes Szenario wird mit unterschiedlich starker Störung gerechnet; die Oberfläche zeigt, ob die
  Empfehlung kippt.
- **Relevanzschwellen:** Unterschiede unter 1 €/Uhr, 0,5 Prozentpunkten Lieferfähigkeit, 5 g CO2e/Uhr oder
  0,5 Tagen Lieferzeit gelten als gleichwertig. Bei Gleichstand entscheiden die Kosten, nie die Listenreihenfolge.
- **Montagestillstand** ist nur Kennzahl, kein Zielkriterium, weil seine Folgen schon in Lieferfähigkeit und
  Kosten stecken. Er zählt nur mit Kundenwirkung.
- **Single Points of Failure:** Elemente ohne Ausweichweg, auch mit allen Handlungsoptionen.

---

## 5. Annahmen und Grenzen

- Werte mit `[A]` in den Konfigurationsdateien sind Annahmen und werden in der Sensitivitätsanalyse geprüft.
- **Montage:** 5 Uhren je Uhrmacher und Tag (Einschalen, Zeiger, Armband, Endkontrolle mit zugekauften Werken),
  Sensitivität 2/3/5/7. Bei 5.000 Uhren sind 4,04 Stellen nötig, besetzt 4,5 Stellen (Auslastung 90 %).
- **Personalkosten:** 49,50 €/h (Destatis 2026) × 1.680 h je Jahr, fällig auch ohne Montage, rund 76 € je Uhr.
- **Lagerhaltungskostensatz** 15 % p. a. (8 % Zins + 7 % Versicherung, Schwund, Veralterung).
- **Zoll:** Zollsatz überall 0 % (Abkommen mit JP, CH, MU, Mercosur, EU-US-Zollverordnung 2026); die
  Zollabfertigung (0,5 bis 2 Tage) bleibt.
- **Fehlmengenkosten:** 50 € je verspätetem Auftrag, Sensitivität 20/50/100 €.
- **Distribution:** nur Onlineverkauf, Paketversand direkt ab Heilbronn, Laufzeit nach Entfernung (1 bis 2,5 Tage).
- **Emissionen:** GLEC-Standardwerte (Well-to-Wheel), konservativ, keine Primärdaten.
- Einkauf wird als Wareneinsatz bewertet, nicht nach Bestelldatum.
- TTR ist eine Eingabe (Störungsdauer), TTS wird simuliert.
- Ausfälle sind unabhängig; keine Retouren, kein Bullwhip-Effekt bis Tier 2, Kapitalbindung nur auf Lagerbestand.

---

## 6. Zuordnung der persönlichen Beiträge

Der gesamte Code ist KI-gestützt (Claude, Anthropic) entstanden; Art der Nutzung je Person siehe
`KI_NACHWEIS.md`. Jede Datei nennt in ihrer ersten Zeile die Verantwortlichen. In geteilten Dateien ist jeder
Abschnitt bzw. Test mit `Urheberschaft dieses Abschnitts` oder `Urheberschaft dieses Tests` markiert. Die Commits
liefen gesammelt über ein Konto und sagen deshalb nichts über die Urheberschaft.

| Kürzel | Person | Schwerpunkt |
|:-:|---|---|
| **N** | Noah Scharly | Visualisierung, Routen und Umleitung, Lieferantenausfall und Notfallpläne |
| **F** | Ferdinand Lautenbach | Entscheidungslogik, Kennzahlen, Netzanalyse |
| **P** | Philipp Maywald | Nachfrage und Montage, Optionsvergleich, Optimierung, Verifikation |
| **B** | Benedict Hufgard | Laden und Prüfen der Daten, Zufall, Bestellungen, Reserve, Transport |
| **G** | gemeinsam | Oberfläche, Domänenmodell, Experimente, Datenbasis, Dokumentation |

### Noah Scharly (N)

| Datei | Beitrag |
|---|---|
| `visualization.py` | animierte Netzwerkkarte mit Sendungen und Paketen, Bestandsbalken synchron zur Karte, Anzeige des aktiven Uhrwerk-Lieferanten, alle Diagramme und die Kennzahlentabelle |
| `simulation.py` (markiert) | Routenwahl nach Priorität, Umleitung gesperrter Abschnitte und von Schiffen auf See; Luftfracht als Brücke und Eilbezug; Notfallplan Sellita mit Wechsel, Storno und Rückkehr |
| `test_modell.py`, `test_handlungsoptionen.py` (markiert) | Tests zu Ausweichroute, Umleitung auf See, Sellita-Wechsel, Notfallplan, Luftfracht und Synchronität von Karte und Balken |

### Ferdinand Lautenbach (F)

| Datei | Beitrag |
|---|---|
| `handlungsoptionen.py` (markiert) | Zielkriterien, Relevanzschwellen, zweistufige Rangfolge mit ROC-Gewichten, Empfehlung je Szenario und Profil, robuste Option, Vor- und Nachteile, Break-even p*, Bandbreite |
| `evaluation.py` | Kennzahlen aus dem Ereignisprotokoll, Kostenaufteilung, CO2e getrennt nach Beschaffung und letzter Meile, Resilienz, Pareto-Front |
| `netzanalyse.py` | Single Points of Failure |
| `test_erweiterungen.py` (markiert) | Tests zu p*, Rauschen unter der Relevanzschwelle und SPOF |

### Philipp Maywald (P)

| Datei | Beitrag |
|---|---|
| `simulation.py` (markiert) | Zeitlogik und Störungsstatus, Nachfrage und Versand, Personal als Fixkosten, Montage mit Gutquote und Stillstand mit Kundenwirkung, Aushilfe, Kurzarbeit, Bestandsanpassung, Vorproduktion, Tagesabschluss |
| `handlungsoptionen.py` (markiert) | Optionen laden, alle Szenarien mit allen Optionen parallel durchrechnen |
| `optimization.py` | Rastersuche, Optimum je Szenario, robuste Empfehlung |
| `test_modell.py`, `test_handlungsoptionen.py`, `test_erweiterungen.py` (markiert) | Konfiguration, Mengenbilanz, Handrechnung, Reproduzierbarkeit, Extremwerte, Stresstest über alle Szenarien und Optionen, Personal, Aushilfe, Kurzarbeit, S5, Rangfolge |

### Benedict Hufgard (B)

| Datei | Beitrag |
|---|---|
| `data.py` | Laden und Zusammenführen der Konfiguration, strenge Feldnamen, Referenz- und Plausibilitätsprüfung, Wiederbeschaffungszeiten, Montageauslegung, Lagerhaltungskostensatz, Konfigurations-Hash |
| `simulation.py` (markiert, inkl. Dateianfang) | Zufallsströme, zufälliger Startbestand, Bestellpolitik (s, Q), automatische Nachbestellung, Fertigung, Reservequelle mit Hochlauf, Kapazität und Bereitschaft, Transportprozess, Störungen, Simulationslauf |
| `test_modell.py` (markiert) | Tests zu Hochlauf und Wochenkapazität der Reservequelle und zum Konfigurations-Hash im Protokoll |

### Gemeinsam (G)

| Datei | Beitrag |
|---|---|
| `app/app.py` | Oberfläche, Zusammenführung aller Teile |
| `domain.py` | Domänenmodell, Grundlage aller Teile |
| `experiment.py`, `__main__.py`, `__init__.py` | Replikationen, Konfidenzintervalle, Szenariovergleich, Sensitivität, Kommandozeile |
| `config/` | Datenbasis, Parameter und Szenarien, Annahmen mit [A] markiert |
| `README.md`, `KI_NACHWEIS.md` | Dokumentation |

**Fremdbibliotheken:** SimPy, NumPy, pandas, NetworkX, Plotly, Streamlit, streamlit-sortables, PyYAML, pytest
(siehe `requirements.txt`).
