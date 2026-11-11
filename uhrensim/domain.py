# Urheberschaft: gemeinsam: Ferdinand Lautenbach, Noah Scharly, Philipp Maywald, Benedict Hufgard (G); KI-gestützt (Claude)
"""Fachliche Objekte des Modells (Domänenmodell aus der Vorlesung).


Die Klassen sind bewusst einfache Datenklassen. Das Verhalten (Prozesse) liegt in
simulation.py, die Bewertung in evaluation.py. So bleiben Daten und Logik getrennt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EU_LAENDER = {
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "HR", "HU",
    "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK",
}


@dataclass(frozen=True)
class Knoten:
    """Knoten des Liefergraphen: Lieferant, Hafen, Flughafen, Werk oder Kunde."""

    id: str
    name: str
    typ: str
    stufe: str
    land: str
    lat: float
    lon: float
    zollgrenze: bool

    @property
    def in_eu(self) -> bool:
        return self.land in EU_LAENDER


@dataclass(frozen=True)
class Kante:
    """Gerichtete Transportrelation mit Verkehrsträger, Distanz und Emissionsfaktor."""

    id: str
    von: str
    nach: str
    modus: str  # see | bahn | binnen | lkw | luft
    km: float
    abfahrt_intervall_tage: float
    co2e_g_tkm: float
    aktiv: bool
    engpass: str
    wegpunkte: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Route:
    """Feste Kantenfolge eines Lieferanten nach Heilbronn."""

    lieferant: str
    id: str
    prioritaet: int
    typ: str  # regulaer | ausweich | notfall
    kanten: tuple[str, ...]


@dataclass(frozen=True)
class Komponente:
    id: str
    bezeichnung: str
    gewicht_g: float
    zoll_satz: float


@dataclass
class Bezugsquelle:
    """Lieferant einer Komponente mit Preis, Fertigungszeit und Wiederbeschaffungszeit."""

    komponente: str
    lieferant: str
    rolle: str  # primaer | alternativ (Notfallplan) | option (wählbare Bezugsquelle)
    preis_eur: float
    fertigung_tage: float
    wbz_tage: float | None  # vom Nutzer gewünschte WBZ, None = physisches Minimum
    tier2: tuple[str, ...]
    wbz_min_tage: float = 0.0  # wird in data.py berechnet
    wbz_zuschlag_tage: float = 0.0  # gewünschte WBZ minus Minimum


@dataclass(frozen=True)
class Stoerung:
    """Eine Störung aus einem Szenario, Zeiten bereits absolut (inkl. Einschwingphase)."""

    typ: str  # knoten_ausfall | kanten_sperre | korridor | nachfrage | kosten
    start: float
    ende: float  # float('inf') = bis Simulationsende
    ziel: str = ""
    faktor: float = 1.0
    monate: tuple[int, ...] = ()
    aktivieren: tuple[str, ...] = ()
    deaktivieren: tuple[str, ...] = ()
    parameter: str = ""
    wert: float = 0.0


@dataclass
class Modell:
    """Vollständig geladenes, geprüftes Modell: Netz, Stückliste, Parameter, Szenario."""

    knoten: dict[str, Knoten]
    kanten: dict[str, Kante]
    routen: dict[str, list[Route]]  # je Lieferant, nach Priorität sortiert
    komponenten: dict[str, Komponente]
    quellen: dict[str, list[Bezugsquelle]]  # je Komponente, primär zuerst
    param: dict[str, Any]
    szenario_id: str = "S0"
    szenario_name: str = "Basis"
    stoerungen: list[Stoerung] = field(default_factory=list)
    warnungen: list[str] = field(default_factory=list)
    kunden: list[dict[str, Any]] = field(default_factory=list)  # Zielorte der Pakete (config/kunden.csv)
    konfig_hash: str = ""  # Fingerabdruck von Stammdaten, Parametern und Szenario (Reproduzierbarkeit)

    # ---- abgeleitete Größen -------------------------------------------------
    @property
    def werk(self) -> str:
        return next(k.id for k in self.knoten.values() if k.typ == "werk")

    @property
    def einschwingen(self) -> float:
        return float(self.param["simulation"]["einschwingen_tage"])

    @property
    def horizont(self) -> float:
        return self.einschwingen + float(self.param["simulation"]["auswertung_tage"])

    def primaerquelle(self, komponente: str) -> Bezugsquelle:
        return next(q for q in self.quellen[komponente] if q.rolle == "primaer")

    def alternativquelle(self, komponente: str) -> Bezugsquelle | None:
        return next((q for q in self.quellen[komponente] if q.rolle == "alternativ"), None)
