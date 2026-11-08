# Simulationsmodell Uhren-Supply-Chain

Ereignisdiskrete Simulation (SimPy) einer globalen Lieferkette für eine Automatikuhr mit Montage in Heilbronn.
Portfolioprüfung Logistik und Supply Chain Management, DHBW Heilbronn, Gruppe 04 Die Logistiker.

Das Modell setzt das Simulationskonzept (Prüfungselement 2) um: ein Modellkern, in den Netz, Mengen,
Kapazitäten und Szenarien als Daten geladen werden. Eine andere Netzvariante der Gruppe ist eine andere
Konfiguration, kein neuer Code.

## Schnellstart

```bash
pip install -r requirements.txt
streamlit run app/app.py          # Oberfläche mit Reglern, Karte, Diagrammen, Kennzahlen
python -m uhrensim -r 20          # Szenariovergleich in der Konsole, CSV in ergebnisse/
python -m uhrensim --optimierung -s S0 S1 S2
pytest -q                         # Verifikationstests
```

Python 3.11 oder neuer.

## Was das Modell abbildet

| Bereich | Umsetzung |
|---|---|
| Netz | Gerichteter Graph: 7 Tier-1-Lieferanten, 4 Tier-2, Häfen, Flughafen, Werk Heilbronn, Kunden |
| Transport | Seeschiff, Bahn, Binnenschiff, Lkw, Luftfracht (Notfall); Fahrplan je Kante, Umschlag, Zoll an der EU-Grenze |
| Nachschub | Bestellpunktverfahren (s, Q) je Komponente, Sicherheitsbestand aus Nachfrage- und Lieferzeitstreuung, Losgröße nach EOQ |
| Montage | Kapazität je Arbeitstag, Gutquote, Fertigwarenlager mit Zielreichweite |
| Nachfrage | Poisson-Prozess, Saisonfaktoren, Lieferzusage in Werktagen |
| Störungen | 5 generische Typen: `knoten_ausfall`, `kanten_sperre`, `korridor`, `nachfrage`, `kosten` |
| Maßnahmen | Zusätzlicher Puffer, Dual Sourcing (Notfallwerk), Luftfracht bei drohendem Fehlteil |
| Kennzahlen | Kosten je Uhr, Lieferfähigkeit β, Lieferzeit (Mittel, 95-%-Quantil), CO2e je Uhr, TTS/TTR, Bestandsreichweite |
| Zufall | Getrennte Zufallsströme je Quelle (SeedSequence), gleiche Ströme in allen Szenarien (Common Random Numbers) |

### Einstellbare Parameter mit Plausibilitätsregeln

| Parameter | Bereich | Regel |
|---|---|---|
| Uhren pro Jahr | 0 bis 25.000 | Nachfragerate = Menge ÷ 365 |
| Montagen pro Tag | 1 bis 150 | Warnung, wenn Kapazität × 250 AT × Gutquote < Jahresmenge |
| Wiederbeschaffungszeit | ab Minimum bis 180 Tage | Nie unter Fertigung + Transit + Zoll der Route (Fehler) |

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
  optimization.py    Rastersuche über Maßnahmen
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
- Basis-Montagekapazität 25 Uhren/Tag (eigene Schätzung), per Regler frei einstellbar; bei 5.000 Uhren/Jahr sind mindestens 21/Tag nötig (20 × 250 × 0,99 = 4.950 < 5.000).
- Emissionsfaktoren: GLEC-Standardwerte (Well-to-Wheel), konservativ, keine Primärdaten.
- TTR ist eine Eingabe (Störungsdauer, bei Lieferanten plus Transit der ersten neuen Sendung), TTS wird simuliert.
- Ausfälle sind unabhängig; keine Retouren, kein Bullwhip-Effekt bis Tier 2, Kapitalbindung nur auf Lagerbestand.
- Lieferzeit in Werktagen (`distribution.lieferzeit_einheit`), auf Kalendertage umstellbar.

## Kennzeichnung der Urheberschaft (Aufgabenstellung 6.2)

| Datei / Modul | Kennzeichen | Bemerkung |
|---|---|---|
| `config/*` | E | Daten und Quellen selbst recherchiert (Poster N. Scharly), Annahmen mit [A] markiert |
| `uhrensim/*.py`, `app/app.py`, `tests/*` | G | Architektur und Fachlogik selbst erarbeitet, Code teilweise selbst, teilweise mit Claude Opus 5.5 umgesetzt, Fehlerbehebung mit KI, siehe `KI_NACHWEIS.md` |
| SimPy, NumPy, pandas, Plotly, Streamlit, PyYAML | F | Fremdbibliotheken, siehe `requirements.txt` |

