# KI-Nachweis

## 1. Verwendetes System und Version

| System | Version | Zeitraum | Zugang |
|---|---|---|---|
| Claude (Anthropic) | Claude Opus 5.5 | Oktober bis November 2026 | claude.ai |
| Claude (Anthropic) | Claude Sonnet 5.5 | seit 1. November 2026 | claude.ai |

Weitere KI-Systeme wurden nicht verwendet.

## 2. Verwendungszweck

Erzeugen von Code nach unseren Vorgaben, Erklärung von Bibliotheken (SimPy, pandas, Plotly, Streamlit),
Fehlersuche, Kommentierung, Vorschläge zu Aufbau und Alternativen, Schreiben und Erweitern von Tests sowie
Formulierung der Dokumentation. Fachliche Vorgaben, Daten und Entscheidungen kamen von der Gruppe.

## 3. Nachweis je Person

Die Prompts sind sinngemäß wiedergegeben (inhaltlich gleichwertige Dokumentation), nicht wörtlich zitiert. Welche
Dateien und Abschnitte zu wem gehören, steht außerdem in der README (Abschnitt 6) und in den Markierungen im Code.

### Noah Scharly (N)

| Punkt | Inhalt |
|---|---|
| Betroffene Abschnitte | `visualization.py`; in `simulation.py` Routenwahl und Umleitung sowie Luftfracht und Notfallplan Sellita; markierte Tests in `test_modell.py` und `test_handlungsoptionen.py` |
| Wesentliche Prompts (sinngemäß) | „Zeichne die Lieferkette als animierte Karte mit Plotly, Sendungen sollen entlang der Wegpunkte laufen.“ · „Zeige darunter Bestandsbalken, die synchron zur Karte laufen.“ · „Wenn ein Abschnitt gesperrt ist, soll die Ware über die Ausweichroute laufen, ohne Maßnahme soll sie warten.“ · „Fällt Miyota aus, soll nach 3 Tagen bei Sellita bestellt werden.“ · „Warum stimmt die Position der Schiffe nicht mit dem Zeitpunkt überein?“ |
| Vorgenommene Änderungen | Vorschläge zu Layout, Abspieltempo und Anordnung von Karte und Balken angepasst; Luftfracht nachträglich an den Störfall gebunden; Darstellungsfehler korrigiert |
| Fachliche und technische Prüfung | fachlich, ob Routen, Umleitung und Lieferantenwechsel dem gewollten Ablauf entsprechen; technisch über eigene Tests und Sichtprüfung der Animation in der Oberfläche |

### Ferdinand Lautenbach (F)

| Punkt | Inhalt |
|---|---|
| Betroffene Abschnitte | in `handlungsoptionen.py` Rangfolge, Empfehlung, p* und Bandbreite; `evaluation.py`; `netzanalyse.py`; markierte Tests in `test_erweiterungen.py` |
| Wesentliche Prompts (sinngemäß) | „Wie kann man Optionen erst nach Mindestanforderungen filtern und dann nach einer Reihenfolge gewichten?“ · „Berechne, ab welcher Eintrittswahrscheinlichkeit sich eine Vorsorge lohnt.“ · „Kleine Unterschiede durch Zufall sollen die Empfehlung nicht entscheiden.“ · „Finde Netzelemente, ohne die eine Komponente nicht mehr geliefert werden kann.“ |
| Vorgenommene Änderungen | Schwellenwerte und Gleichstandsregel selbst festgelegt; Bewertung des Einkaufs auf Wareneinsatz umgestellt; Texte für Vor- und Nachteile angepasst |
| Fachliche und technische Prüfung | fachlich durch Nachrechnen von Beispielen (p*, ROC-Gewichte); technisch über eigene Tests zu p*, Relevanzschwellen und SPOF |

### Philipp Maywald (P)

| Punkt | Inhalt |
|---|---|
| Betroffene Abschnitte | in `simulation.py` Nachfrage, Versand, Montage und Personal; in `handlungsoptionen.py` der Optionsvergleich; `optimization.py`; markierte Tests in `test_modell.py`, `test_handlungsoptionen.py` und `test_erweiterungen.py` |
| Wesentliche Prompts (sinngemäß) | „Bilde die Nachfrage als Poisson-Prozess mit Saisonfaktor ab.“ · „Personal soll Fixkosten sein, auch wenn nicht montiert wird.“ · „Bei Rückstand soll eine Aushilfe dazukommen.“ · „Rechne alle Szenarien mit allen Optionen parallel, aber mit denselben Ergebnissen wie nacheinander.“ · „Schreibe einen Test, der prüft, dass die Mengenbilanz in allen Szenarien aufgeht.“ |
| Vorgenommene Änderungen | Montage von Stückkosten auf Fixkosten umgestellt; Auslöser und Aufschläge der Optionen selbst festgelegt; Rückfall auf serielles Rechnen ergänzt |
| Fachliche und technische Prüfung | fachlich, ob Kosten und Personalwirkung im Nachfrageeinbruch plausibel sind; technisch über Mengenbilanz, Handrechnung, Reproduzierbarkeit und Stresstest |

### Benedict Hufgard (B)

| Punkt | Inhalt |
|---|---|
| Betroffene Abschnitte | `data.py`; in `simulation.py` Zufallsströme, Startbestand, Bestellpolitik, Nachschub, Reservequelle, Transport, Störungen und Simulationslauf; markierte Tests in `test_modell.py` |
| Wesentliche Prompts (sinngemäß) | „Lade die Daten aus CSV und YAML und brich ab, wenn ein Feldname falsch geschrieben ist.“ · „Berechne Sicherheitsbestand und Losgröße für ein Bestellpunktverfahren.“ · „Die Reservequelle soll erst nach einem Hochlauf und nur begrenzt liefern.“ · „Warum wirkt die Störung in manchen Replikationen gar nicht?“ |
| Vorgenommene Änderungen | zufälliger Startbestand ergänzt; Nachbestellung auf Lagerposition inklusive offener Bestellungen umgestellt; Plausibilitätsregeln in Fehler und Warnungen aufgeteilt |
| Fachliche und technische Prüfung | fachlich, ob Bestellverhalten und Reserve realistisch sind; technisch über eigene Tests zu Reservequelle und Konfigurations-Hash sowie Testläufe mit festem Seed |

### Gemeinsam (G)

| Punkt | Inhalt |
|---|---|
| Betroffene Abschnitte | `app/app.py`, `domain.py`, `experiment.py`, `__main__.py`, `__init__.py`, Datenbasis in `config/`, README |
| Wesentliche Prompts (sinngemäß) | „Baue eine Streamlit-Oberfläche mit Überblick, Detailansicht und Datenbasis.“ · „Fasse die Replikationen mit Konfidenzintervall zusammen.“ · „Gliedere die README in Startvoraussetzungen, Dateistruktur, Ausführung und Beiträge.“ |
| Vorgenommene Änderungen | Aufbau und Texte der Oberfläche überarbeitet; Rechenzeit durch relevante Optionen je Szenario verkürzt |
| Fachliche und technische Prüfung | gemeinsamer Durchlauf aller Szenarien in der Oberfläche; alle 64 Tests grün |
