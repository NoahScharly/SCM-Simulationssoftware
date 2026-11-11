# Urheberschaft: geteilt zwischen Benedict Hufgard (B), Philipp Maywald (P) und Noah Scharly (N); KI-gestützt (Claude), siehe KI_NACHWEIS.md
# Zuordnung je Abschnitt über die Markierungen '# ---- Urheberschaft dieses Abschnitts'.
# Ohne Markierung bis zum ersten Abschnitt (Importe, Zufallsströme, Startbestand, Init): Benedict Hufgard (B)
"""Ereignisdiskrete Simulation der Supply Chain mit SimPy (Modul simulation).

Prozesse
--------
- Nachfrage:      Kundenaufträge als Poisson-Prozess, je Tag Saison- und Szenariofaktor
- Montage:        jeden Arbeitstag, begrenzt durch Kapazität und verfügbare Komponenten
- Nachschub:      Bestellpunktverfahren (s, Q) je Komponente auf die Lagerposition
- Lieferant:      Fertigung, Warten bei Ausfall, optional Umschalten auf Alternativlieferant
- Transport:      Kante für Kante mit Fahrplan, Umschlag, Zoll, Umleitung bei Sperrung
- Luftfracht:     optionaler Eilbezug bei drohendem Fehlteil
- Tagesabschluss: Bestandsabbild je Tag für Kosten und Diagramme

Zeit: Tage als Gleitkommazahl. Tag 0 bis einschwingen = Vorlauf, danach Auswertung.
Der erste Auswertungstag entspricht dem 1. Januar (ein Freitag, Jahr ohne Schaltjahr).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import simpy

from .data import dreieck_mittel, lagerhaltungssatz
from .domain import EU_LAENDER, Bezugsquelle, Modell, Route

STROEME = ("nachfrage", "versand", "zoll", "umschlag", "gutquote", "kunden", "startbestand")
MONATSENDEN = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def zufallsstroeme(seed: int, replikation: int) -> dict[str, np.random.Generator]:
    """Je Zufallsquelle ein eigener Generator (SeedSequence).

    Gleicher Seed und gleiche Replikation ergeben in jedem Szenario dieselben Ströme
    (Common Random Numbers): Unterschiede entstehen durch das Szenario, nicht durch Zufall.
    """
    wurzel = np.random.SeedSequence(entropy=int(seed), spawn_key=(int(replikation),))
    return {name: np.random.default_rng(kind) for name, kind in zip(STROEME, wurzel.spawn(len(STROEME)))}


@dataclass
class Bestellpolitik:
    """(s, Q)-Regel einer Komponente, analytisch vorbelegt."""

    komponente: str
    bedarf_tag: float
    wbz_tage: float
    sicherheitsbestand: float
    puffer_extra: float
    meldebestand: float
    losgroesse: float
    anfangsbestand: float


@dataclass
class Ergebnis:
    """Rohergebnis eines Laufs. Alle Kennzahlen werden in evaluation.py daraus berechnet."""

    modell: Modell
    replikation: int
    deterministisch: bool
    log: pd.DataFrame
    auftraege: pd.DataFrame
    tage: pd.DataFrame
    sendungen: list[dict[str, Any]]
    politik: dict[str, Bestellpolitik]
    stoerungsfenster: list[dict[str, Any]] = field(default_factory=list)


class Simulation:
    """Ein Simulationslauf für ein Modell und eine Replikation."""

    def __init__(self, modell: Modell, replikation: int = 0, deterministisch: bool = False):
        self.m = modell
        self.p = modell.param
        self.rep = replikation
        self.det = deterministisch
        self.env = simpy.Environment()
        self.rng = zufallsstroeme(self.p["simulation"]["seed"], replikation)
        self.W = modell.einschwingen
        self.H = modell.horizont
        self.werk = modell.werk
        self.gut = float(self.p["montage"]["gutquote"])
        self.ausschuss_rest = 0.0
        self.basis_rate = float(self.p["nachfrage"]["uhren_pro_jahr"]) / 365.0
        self.speed = self.p["transport"]["geschwindigkeit_km_tag"]
        self.vf = float(self.p["emissionen"]["verpackungsfaktor"])

        self._eid = 0
        self._sid = 0
        self._aid = 0
        self.log: list[dict[str, Any]] = []
        self.auftraege: list[dict[str, Any]] = []
        self.kunden = modell.kunden
        if self.kunden:
            g = np.array([k["gewicht"] for k in self.kunden], dtype=float)
            self.kunden_p = g / g.sum()
            self.kunden_kum = np.cumsum(self.kunden_p)
        self.tage: list[dict[str, Any]] = []
        self.sendungen: list[dict[str, Any]] = []
        self.rueckstand: deque = deque()

        self.politik = {k: self._bestellpolitik(k) for k in modell.komponenten}
        # Startbestand je Replikation zufällig zwischen Meldebestand und Meldebestand + Losgröße. Sonst bestellen alle
        # Replikationen im selben Takt, und eine Störung trifft immer dieselbe Stelle im Bestellzyklus (bei Losgrößen
        # von 100 bis 200 Tagen Bedarf z. B. nie eine Sendung). Deterministischer Lauf: voller Bestand wie bisher.
        u = {k: (1.0 if deterministisch else float(self.rng["startbestand"].random())) for k in self.politik}
        self.bestand = {k: float(math.ceil(pol.meldebestand + u[k] * pol.losgroesse))
                        for k, pol in self.politik.items()}
        self.pipeline = {k: 0.0 for k in modell.komponenten}
        self.offen: dict[str, list[dict[str, Any]]] = {k: [] for k in modell.komponenten}
        self.aktive_quelle = {k: modell.primaerquelle(k) for k in modell.komponenten}
        self.luft_offen = {k: False for k in modell.komponenten}
        self.reserve_bereit_ab: dict[str, float] = {}   # Reservequelle fertigt erst nach dem Hochlauf
        self.reserve_frei_ab: dict[str, float] = {}     # ab wann ihre Kapazität wieder frei ist
        self.fg = float(math.ceil(self._fg_ziel(0.0)))
        self.fg_wip = 0.0

    # ---- Urheberschaft dieses Abschnitts: Philipp Maywald (P) (Zeitlogik, Störungsstatus, Nachfragefaktor, Preise und Zoll)
    # ================================================================ Zufall
    def _dreieck(self, strom: str, abc: list[float]) -> float:
        a, b, c = abc
        if self.det or a == c:
            return dreieck_mittel(abc)
        return float(self.rng[strom].triangular(a, b, c))

    # ================================================================ Kalender
    def tag_im_jahr(self, t: float) -> int:
        return int(math.floor(t - self.W)) % 365

    def monat(self, t: float) -> int:
        return int(np.searchsorted(MONATSENDEN, self.tag_im_jahr(t), side="right")) + 1

    def ist_arbeitstag(self, tag: int) -> bool:
        wochentag = (tag - int(self.W) + 4) % 7  # Auswertungstag 0 = Freitag
        return wochentag < int(self.p["montage"]["arbeitstage_pro_woche"])

    # ================================================================ Störungen
    def _aktiv(self, s, t: float) -> bool:
        return s.start <= t < s.ende

    def knoten_unten(self, knoten: str, t: float) -> bool:
        return any(s.typ == "knoten_ausfall" and s.ziel == knoten and self._aktiv(s, t) for s in self.m.stoerungen)

    def lieferant_unten(self, q: Bezugsquelle, t: float) -> bool:
        return any(self.knoten_unten(n, t) for n in (q.lieferant, *q.tier2))

    def kante_aktiv(self, kid: str, t: float) -> bool:
        """Dauerhafter Zustand (Korridor offen oder zu)."""
        aktiv = self.m.kanten[kid].aktiv
        for s in self.m.stoerungen:
            if s.typ == "korridor" and self._aktiv(s, t):
                if kid in s.aktivieren:
                    aktiv = True
                if kid in s.deaktivieren:
                    aktiv = False
        return aktiv

    def kante_nutzbar(self, kid: str, t: float) -> bool:
        k = self.m.kanten[kid]
        if not self.kante_aktiv(kid, t):
            return False
        if any(s.typ == "kanten_sperre" and s.ziel == kid and self._aktiv(s, t) for s in self.m.stoerungen):
            return False
        return not (self.knoten_unten(k.von, t) or self.knoten_unten(k.nach, t))

    def naechste_freigabe(self, t: float) -> float:
        """Frühestes Ende einer gerade aktiven zeitlich begrenzten Störung."""
        enden = [s.ende for s in self.m.stoerungen if self._aktiv(s, t) and math.isfinite(s.ende)]
        return min(enden) if enden else t + 1.0

    def nachfragefaktor(self, t: float) -> float:
        f = float(self.p["nachfrage"].get("monatsfaktoren", {}).get(self.monat(t), 1.0))
        for s in self.m.stoerungen:
            if s.typ != "nachfrage":
                continue
            if s.monate:
                if t >= self.W and self.monat(t) in s.monate:
                    f *= s.faktor
            elif self._aktiv(s, t):
                f *= s.faktor
        return f

    def preis(self, q: Bezugsquelle, t: float) -> float:
        preis = q.preis_eur
        for s in self.m.stoerungen:
            if s.typ == "kosten" and s.parameter == f"preis:{q.komponente}" and self._aktiv(s, t):
                preis = preis * s.faktor if s.faktor != 1.0 else s.wert
        return preis

    def zollsatz(self, komponente: str, t: float) -> float:
        satz = self.m.komponenten[komponente].zoll_satz
        for s in self.m.stoerungen:
            if s.typ == "kosten" and s.parameter == f"zoll:{komponente}" and self._aktiv(s, t):
                satz = s.wert
        return satz

    # ---- Urheberschaft dieses Abschnitts: Benedict Hufgard (B) (Bestellpolitik, Ereignisprotokoll)
    # ================================================================ Politik
    def _bestellpolitik(self, komp: str) -> Bestellpolitik:
        q = self.m.primaerquelle(komp)
        bedarf = self.basis_rate / self.gut if self.basis_rate > 0 else 0.0
        L = q.wbz_min_tage + q.wbz_zuschlag_tage
        z = NormalDist().inv_cdf(float(self.p["bestand"]["servicegrad_ziel"]))
        # Sicherheitsbestand mit Nachfrage- und Lieferzeitstreuung:
        # SS = z * sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
        # sigma_d^2: Poisson-Nachfrage plus Bündelung der Montage auf Arbeitstage
        an = float(self.p["montage"]["arbeitstage_pro_woche"]) / 7.0
        var_d = bedarf + bedarf ** 2 * (1.0 / an - 1.0)
        var_L = self._lieferzeit_varianz(q)
        ss = z * math.sqrt(L * var_d + bedarf ** 2 * var_L) if bedarf > 0 else 0.0
        extra = float(self.p["massnahmen"].get("puffer_extra", {}).get(komp, 0) or 0)
        s = bedarf * L + ss + extra
        # EOQ als Referenz: Fixkosten je Sendung gegen Kapitalbindung
        route = next(r for r in self.m.routen[q.lieferant] if r.typ == "regulaer")
        K = sum(self.p["kosten"]["transport_fix_eur_sendung"][self.m.kanten[k].modus] for k in route.kanten)
        h = q.preis_eur * lagerhaltungssatz(self.p)
        D = bedarf * 365.0
        eoq = math.sqrt(2 * D * K / h) if D > 0 and h > 0 else 0.0
        Q = max(eoq, bedarf * float(self.p["bestand"]["bestellintervall_min_tage"]), 1.0) if bedarf > 0 else 0.0
        return Bestellpolitik(komp, bedarf, L, ss, extra, s, float(math.ceil(Q)), s + Q)

    def _lieferzeit_varianz(self, q: Bezugsquelle) -> float:
        """Varianz der Wiederbeschaffungszeit: Warten auf Abfahrt (gleichverteilt), Umschlag und Zoll (Dreieck)."""
        def var_dreieck(abc):
            a, b, c = abc
            return (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0
        route = next(r for r in self.m.routen[q.lieferant] if r.typ == "regulaer"
                     and all(self.m.kanten[k].aktiv for k in r.kanten))
        tp = self.p["transport"]
        var = 0.0
        for i, kid in enumerate(route.kanten):
            k = self.m.kanten[kid]
            var += k.abfahrt_intervall_tage ** 2 / 12.0
            if i < len(route.kanten) - 1:
                var += var_dreieck(tp["umschlag_tage"][k.modus])
        if self.m.knoten[q.lieferant].land not in EU_LAENDER:
            var += var_dreieck(tp["zoll_tage"])
        return var

    def _fg_ziel(self, t: float) -> float:
        """Ziel-Fertigwarenbestand. Maßnahme saison_vorproduktion: in den genannten Monaten vor der Spitze wird ein
        höherer Vorrat aufgebaut (freie Montagekapazität vor November nutzen)."""
        ziel = float(self.p["montage"]["fertigwaren_ziel_tage"])
        vp = self.p["massnahmen"].get("saison_vorproduktion") or {}
        if vp and t >= self.W and self.monat(t) in set(vp.get("monate", [])):
            ziel = float(vp.get("ziel_tage", ziel))
        return ziel * self.basis_rate * self.nachfragefaktor(t)

    # ================================================================ Protokoll
    def _log(self, ereignis: str, objekt_id: str, objekt_typ: str, ort: str, menge: float = 0.0,
             komponente: str = "", alt: str = "", neu: str = "", kosten: float = 0.0, kategorie: str = "",
             co2e_g: float = 0.0, kante: str = "") -> None:
        self._eid += 1
        self.log.append({
            "event_id": self._eid, "zeit": self.env.now, "ereignis": ereignis, "objekt_id": objekt_id,
            "objekt_typ": objekt_typ, "ort": ort, "zustand_alt": alt, "zustand_neu": neu,
            "menge": menge, "einheit": "Stück", "komponente": komponente, "kante": kante,
            "kosten_eur": kosten, "kostenart": kategorie, "co2e_g": co2e_g,
        })

    # ---- Urheberschaft dieses Abschnitts: Philipp Maywald (P) (Nachfrage, Versand, Personalkosten, Montage, Aushilfe, Kurzarbeit, Saison, Bestandsanpassung)
    # ================================================================ Nachfrage und Versand
    def _nachfrage(self):
        tag, akkumuliert = 0, 0.0
        while tag < self.H:
            lam = self.basis_rate * self.nachfragefaktor(tag + 0.5)
            if self.det:
                akkumuliert += lam
                n = int(akkumuliert)
                akkumuliert -= n
                zeiten = [tag + (i + 0.5) / n for i in range(n)] if n else []
            else:
                n = int(self.rng["nachfrage"].poisson(lam))
                zeiten = sorted(tag + self.rng["nachfrage"].random(n))
            for z in zeiten:
                yield self.env.timeout(max(0.0, z - self.env.now))
                self._neuer_auftrag()
            tag += 1
            yield self.env.timeout(max(0.0, tag - self.env.now))

    def _neuer_auftrag(self) -> None:
        self._aid += 1
        a = {"auftrag_id": f"A{self._aid}", "t_eingang": self.env.now, "t_versand": np.nan, "t_zustellung": np.nan}
        if self.kunden:
            if self.det:
                i = int(np.searchsorted(self.kunden_kum, (self._aid * 0.6180339887) % 1.0, side="right"))
            else:
                i = int(self.rng["kunden"].choice(len(self.kunden), p=self.kunden_p))
            k = self.kunden[min(i, len(self.kunden) - 1)]
            a.update(kunde=k["id"], lat=k["lat"], lon=k["lon"], km=k["km"])
        else:
            a.update(kunde="kunden_de", lat=np.nan, lon=np.nan, km=np.nan)
        self.auftraege.append(a)
        self._log("auftragseingang", a["auftrag_id"], "kundenauftrag", a["kunde"], 1, neu="offen")
        if self.fg >= 1:
            self.fg -= 1
            self.env.process(self._versende(a))
        else:
            self.rueckstand.append(a)
            self._log("rueckstand", a["auftrag_id"], "kundenauftrag", self.werk, 1, alt="offen", neu="rueckstand")

    def _versende(self, a: dict[str, Any]):
        tag = int(math.floor(self.env.now))
        while not self.ist_arbeitstag(tag):
            tag += 1
        start = max(self.env.now, tag + 0.25) if tag > self.env.now else self.env.now
        yield self.env.timeout(start - self.env.now)
        a["t_versand"] = self.env.now
        self._log("versand", a["auftrag_id"], "kundenauftrag", self.werk, 1, alt="reserviert", neu="versandt",
                  kosten=float(self.p["kosten"]["paket_eur_uhr"]), kategorie="paket")
        dist = self.p["distribution"]
        yield self.env.timeout(self._dreieck("versand", self._paketlaufzeit(a)))
        a["t_zustellung"] = self.env.now
        em = self.p["emissionen"]
        hauptlauf_km = 0.0 if math.isnan(a["km"]) else a["km"] * float(dist.get("strassenfaktor", 1.3))
        co2 = em["paket_kg"] / 1000 * (hauptlauf_km * float(dist.get("hauptlauf_g_tkm", 101))
                                        + dist["last_mile_km"] * em["last_mile_g_tkm"])
        self._log("zustellung", a["auftrag_id"], "kundenauftrag", a["kunde"], 1, alt="versandt",
                  neu="zugestellt", co2e_g=co2, kategorie="letzte_meile")

    def _paketlaufzeit(self, a: dict[str, Any]) -> list[float]:
        """Dreieck der Paketlaufzeit nach Entfernungszone (Luftlinie ab Heilbronn)."""
        dist = self.p["distribution"]
        zonen = dist.get("versand_zonen") or []
        if not zonen or math.isnan(a["km"]):
            return dist["versand_tage"]
        for z in sorted(zonen, key=lambda z: z["bis_km"]):
            if a["km"] <= z["bis_km"]:
                return z["tage"]
        return zonen[-1]["tage"]

    def _bediene_rueckstand(self) -> None:
        while self.fg >= 1 and self.rueckstand:
            a = self.rueckstand.popleft()
            self.fg -= 1
            self.env.process(self._versende(a))

    # ================================================================ Montage
    def _personal_tagessatz(self) -> float:
        """Kosten eines Montage-Mitarbeiters je Arbeitstag: Lohnkosten je Stunde × Stunden je Jahr / Arbeitstage."""
        k = self.p["kosten"]
        arbeitstage_jahr = 52.0 * float(self.p["montage"]["arbeitstage_pro_woche"])
        return float(k.get("lohnkosten_eur_h", 0.0)) * float(k.get("stunden_ma_jahr", 0.0)) / arbeitstage_jahr

    def _montage(self):
        """Tägliche Montage. Kostenmodell 'personal': die Stammbelegschaft kostet jeden Arbeitstag, auch wenn mangels
        Teilen oder Nachfrage weniger montiert wird (Fixkosten). 'stueck': Montagekosten je montierter Uhr.

        Maßnahme Aushilfe: Liegt die Montage mehr als 'ausloeser_tage' Tagesbedarf hinter dem Plan
        und sind Teile da, kommt eine Aushilfe (Zeitarbeit mit Aufschlag) für mindestens 'mindest_tage' Arbeitstage
        und bleibt, bis die Stammbelegschaft den Rückstand wieder allein schafft.
        """
        yield self.env.timeout(0.25)
        mo, ko = self.p["montage"], self.p["kosten"]
        kap_stamm = float(mo["kapazitaet_pro_tag"])
        pro_ma = float(mo.get("uhren_pro_mitarbeiter_tag", 5))
        personal = str(ko.get("montage_kostenmodell", "stueck")) == "personal"
        tagessatz = self._personal_tagessatz()
        stamm = float(mo.get("mitarbeiter") or math.ceil(kap_stamm / pro_ma))
        au = self.p["massnahmen"].get("aushilfe") or {}
        au_ma = float(au.get("mitarbeiter", 1)) if au else 0.0
        au_seit: int | None = None
        while self.env.now < self.H:
            tag = int(math.floor(self.env.now))
            if self.ist_arbeitstag(tag):
                bedarf = self._fg_ziel(self.env.now) + len(self.rueckstand) - self.fg - self.fg_wip
                # Maßnahme Kurzarbeit: bei erkannt gesunkener Nachfrage arbeitet die Stammbelegschaft nur den Anteil f;
                # für ausgefallene Stunden bleiben Remanenzkosten (Sozialversicherung, Urlaub) [A]
                ka = self.p["massnahmen"].get("kurzarbeit") or {}
                f_ka = self._erkannter_faktor(float(ka.get("reaktion_tage", 14))) if ka and tag >= self.W else 1.0
                kap_stamm = float(mo["kapazitaet_pro_tag"]) * f_ka
                if personal:
                    satz = f_ka + (1.0 - f_ka) * float(ka.get("remanenz", 0.25)) if ka else 1.0
                    self._log("personal", f"P{tag}", "montage", self.werk, stamm, kosten=stamm * tagessatz * satz,
                              kategorie="montage")
                    if f_ka < 1.0:
                        self._log("kurzarbeit", f"K{tag}", "montage", self.werk, stamm * (1.0 - f_ka))
                if au:
                    tagesbedarf = max(self.basis_rate * self.nachfragefaktor(self.env.now), 1e-9)
                    # Aushilfe nur, wenn die Montage der Engpass ist: keine laufende Versorgungsstörung und Teile
                    # für den Mindesteinsatz im Lager. Sonst verbraucht sie nur schneller, was ohnehin knapp ist.
                    kap_mit = kap_stamm + au_ma * pro_ma
                    teile_da = (min(self.bestand.values()) >= kap_mit * float(au.get("mindest_tage", 10))
                                and not any(self._im_stoerfall(self.env.now, k, nachlauf=False)
                                            for k in self.bestand))
                    if au_seit is None and teile_da and bedarf - kap_stamm > float(au["ausloeser_tage"]) * tagesbedarf:
                        au_seit = tag
                        self._log("aushilfe_start", f"A{tag}", "montage", self.werk, au_ma, neu="aushilfe")
                    elif (au_seit is not None and tag - au_seit >= float(au.get("mindest_tage", 10)) * 7 / 5
                          and bedarf <= kap_stamm):
                        au_seit = None
                        self._log("aushilfe_ende", f"A{tag}", "montage", self.werk, au_ma, alt="aushilfe")
                    if au_seit is not None:
                        self._log("aushilfe", f"A{tag}", "montage", self.werk, au_ma, kategorie="montage",
                                  kosten=au_ma * tagessatz * (1.0 + float(au.get("aufschlag", 0.0))))
                kap = kap_stamm + (au_ma * pro_ma if au_seit is not None else 0.0)
                soll = int(min(kap, max(0.0, math.ceil(bedarf))))
                if soll > 0:
                    moeglich = int(min(self.bestand.values()))
                    n = min(soll, moeglich)
                    if n < soll:
                        fehlend = [k for k, b in self.bestand.items() if b < soll]
                        # Stillstand mit Kundenwirkung: nur wenn Bestellungen warten oder der
                        # normale Fertigwarenvorrat unterschritten ist. Fehlt ein Teil, während genug fertige Uhren
                        # da sind (z. B. beim Vorproduzieren), merkt das kein Kunde: eigenes Ereignis, zählt nicht.
                        normal = (float(mo["fertigwaren_ziel_tage"]) * self.basis_rate
                                  * self.nachfragefaktor(self.env.now))
                        wirkung = len(self.rueckstand) > 0 or self.fg < normal
                        self._log("fehlteil" if wirkung else "fehlteil_ohne_wirkung", f"M{tag}", "montage",
                                  self.werk, soll - n, komponente=";".join(fehlend),
                                  neu="stillstand" if wirkung else "vorrat_reicht")
                    if n > 0:
                        for k in self.bestand:
                            self.bestand[k] -= n
                        self.fg_wip += n
                        self._log("montage_start", f"M{tag}", "montage", self.werk, n, alt="komponenten",
                                  neu="in_montage", kosten=0.0 if personal else n * float(ko["montage_eur_uhr"]),
                                  kategorie="" if personal else "montage")
                        self.env.process(self._fertigmeldung(n, tag))
                    self._pruefe_nachschub()
            yield self.env.timeout(1.0)

    def _fertigmeldung(self, n: int, tag: int):
        yield self.env.timeout(float(self.p["montage"]["durchlauf_tage"]))
        # Gutquote fest (kein Zufall): Ausschuss-Bruchteile werden übertragen, damit z. B. bei 25 Uhren am Tag
        # und 99 % Gutquote im Mittel genau 1 % Ausschuss entsteht und nicht durch Runden 0 %.
        self.ausschuss_rest += n * (1.0 - self.gut)
        aus = int(math.floor(self.ausschuss_rest + 1e-9))
        self.ausschuss_rest -= aus
        gut = n - aus
        self.fg_wip -= n
        self.fg += gut
        self._log("montage_ende", f"M{tag}", "montage", self.werk, gut, alt="in_montage", neu="fertigware")
        if n - gut:
            self._log("ausschuss", f"M{tag}", "montage", self.werk, n - gut, neu="ausschuss")
        self._bediene_rueckstand()

    # ================================================================ Nachschub
    def _saison_zuschlag(self, komp: str) -> float:
        """Maßnahme saison_bestand: in den genannten Monaten wird der Meldebestand um extra_tage Bedarf angehoben.

        Weil die Wiederbeschaffung bis zu rund zwei Monate dauert, beginnt der Aufbau vor der Spitze (z. B. ab
        August für November und Dezember). Danach fällt der Meldebestand zurück, der Vorrat wird abgebaut.
        """
        sb = self.p["massnahmen"].get("saison_bestand") or {}
        if not sb or self.env.now < self.W or self.monat(self.env.now) not in set(sb.get("monate", [])):
            return 0.0
        return float(sb.get("extra_tage", 0)) * self.politik[komp].bedarf_tag

    def _erkannter_faktor(self, reaktion_tage: float) -> float:
        """Nachfragefaktor, wie ihn das Unternehmen mit Verzögerung wahrnimmt (nur Rückgänge, höchstens 1).

        Der Einbruch wird erst nach 'reaktion_tage' erkannt, und die Erholung ebenso spät. Das ist das Risiko der
        Anpassungsmaßnahmen: Kommt die Nachfrage zurück, ist das Unternehmen noch einige Tage zu knapp aufgestellt.
        """
        return min(1.0, self.nachfragefaktor(self.env.now - float(reaktion_tage)))

    def _anpassungsfaktor(self) -> float:
        """Maßnahme bestand_anpassen: Meldebestand und Bestellmenge folgen der erkannt gesunkenen Nachfrage."""
        ba = self.p["massnahmen"].get("bestand_anpassen") or {}
        if not ba or self.env.now < self.W:
            return 1.0
        return self._erkannter_faktor(float(ba.get("reaktion_tage", 7)))

    # ---- Urheberschaft dieses Abschnitts: Benedict Hufgard (B) (Nachschub, Bestellungen, Fertigung, Reservequelle mit Hochlauf und Kapazität, Lieferant)
    def _pruefe_nachschub(self) -> None:
        f = self._anpassungsfaktor()
        for k, pol in self.politik.items():
            if pol.losgroesse <= 0:
                continue
            position = self.bestand[k] + self.pipeline[k]
            los = max(1.0, math.ceil(pol.losgroesse * f))
            while position <= pol.meldebestand * f + self._saison_zuschlag(k):
                self.bestelle(k, los, self.aktive_quelle[k])
                position += los

    def bestelle(self, komp: str, menge: float, quelle: Bezugsquelle, typ: str = "regulaer") -> dict[str, Any]:
        self._sid += 1
        b = {"id": f"B{self._sid}", "komponente": komp, "menge": menge, "lieferant": quelle.lieferant,
             "typ": typ, "status": "fertigung", "storniert": False, "t_bestellung": self.env.now}
        self.pipeline[komp] += menge
        self.offen[komp].append(b)
        zuschlag = float(self.p["massnahmen"].get("luft_eilzuschlag", 0.0)) if typ == "notfall" else 0.0
        self._log("bestellung", b["id"], "bestellung", quelle.lieferant, menge, komponente=komp, neu="bestellt",
                  kosten=menge * self.preis(quelle, self.env.now) * (1 + zuschlag), kategorie="einkauf")
        self.env.process(self._lieferant(b, quelle))
        return b

    def _fertigungsende(self, q: Bezugsquelle, start: float, dauer: float) -> float:
        """Ende der Fertigung, wenn ein Werksausfall die Produktion unterbricht.

        Während eines Ausfalls (Lieferant oder Vorlieferant) ruht die Fertigung und läuft danach weiter.
        Aufträge, die während des Ausfalls eingehen, beginnen erst nach dem Wiederanlauf.
        """
        fenster = sorted((st.start, st.ende) for st in self.m.stoerungen
                         if st.typ == "knoten_ausfall" and st.ziel in (q.lieferant, *q.tier2))
        t, rest = start, dauer
        for a, e in fenster:
            if e <= t:
                continue
            if a > t:
                arbeit = min(rest, a - t)
                t += arbeit
                rest -= arbeit
                if rest <= 1e-9:
                    return t
            t = max(t, e)
        return t + rest

    def _reserve_ende(self, q: Bezugsquelle, menge: float, fertigung: float) -> float:
        """Fertigungsende bei einer Reservequelle: Sie fertigt erst nach dem Hochlauf und nur mit
        begrenzter Wochenkapazität. Aufträge laufen nacheinander durch die Kapazität."""
        ms = self.p["massnahmen"]
        kap_tag = float(ms.get("notfall_kapazitaet_stk_woche", 0) or 0) / 7.0
        start = max(self.env.now, self.reserve_bereit_ab.get(q.lieferant, self.env.now),
                    self.reserve_frei_ab.get(q.lieferant, 0.0))
        dauer = menge / kap_tag if kap_tag > 0 else 0.0
        self.reserve_frei_ab[q.lieferant] = start + dauer
        return max(self.env.now + fertigung, start + dauer)

    def _lieferant(self, b: dict[str, Any], q: Bezugsquelle):
        fertigung = 2.0 if b["typ"] == "notfall" else q.fertigung_tage + q.wbz_zuschlag_tage
        if q.rolle == "alternativ" and b["typ"] != "notfall":
            fertigung = self._reserve_ende(q, b["menge"], fertigung) - self.env.now
        ende = self._fertigungsende(q, self.env.now, fertigung)
        if ende - self.env.now > fertigung + 1e-9:
            b["status"] = "wartet"
            self._log("warten_lieferant", b["id"], "bestellung", q.lieferant, b["menge"],
                      komponente=b["komponente"], alt="fertigung", neu="wartet")
        yield self.env.timeout(ende - self.env.now)
        while self.lieferant_unten(q, self.env.now) and not b["storniert"]:
            if b["status"] != "wartet":
                b["status"] = "wartet"
                self._log("warten_lieferant", b["id"], "bestellung", q.lieferant, b["menge"],
                          komponente=b["komponente"], alt="fertigung", neu="wartet")
            yield self.env.timeout(max(0.01, self.naechste_freigabe(self.env.now) - self.env.now))
        if b["storniert"]:
            return
        b["status"] = "transport"
        yield from self._transport(b, q)

    # ---- Urheberschaft dieses Abschnitts: Noah Scharly (N) (Routenwahl, Umleitung gesperrter Abschnitte, Umleitung von Schiffen unterwegs)
    # ================================================================ Transport
    def _routen(self, lieferant: str, notfall: bool) -> list[Route]:
        """Routen eines Lieferanten. Ausweichrouten nur mit Maßnahme 'ausweichroute' (sonst wartet die Ware)."""
        if notfall:
            typen: tuple[str, ...] = ("notfall",)
        elif self.p["massnahmen"].get("ausweichroute", False):
            typen = ("regulaer", "ausweich")
        else:
            typen = ("regulaer",)
        return [r for r in self.m.routen.get(lieferant, []) if r.typ in typen]

    def _waehle_route(self, lieferant: str, notfall: bool) -> Route:
        t = self.env.now
        kandidaten = self._routen(lieferant, notfall)
        for r in kandidaten:
            if all(self.kante_nutzbar(k, t) for k in r.kanten):
                return r
        for r in kandidaten:  # alles gestört: Route mit offenem Korridor nehmen und warten
            if all(self.kante_aktiv(k, t) for k in r.kanten):
                return r
        return kandidaten[-1]

    def _umleitung(self, lieferant: str, position: str, notfall: bool) -> list[str] | None:
        t = self.env.now
        for r in self._routen(lieferant, notfall):
            for i, kid in enumerate(r.kanten):
                if self.m.kanten[kid].von == position:
                    rest = list(r.kanten[i:])
                    if all(self.kante_nutzbar(k, t) for k in rest):
                        return rest
        return None

    def _umleitung_unterwegs(self, lieferant: str, k: Any, notfall: bool) -> list[str] | None:
        """Schiff ist unterwegs, der Zielhafen ist gesperrt: gleiches Verkehrsmittel, anderer Zielhafen einer
        erlaubten Route (z. B. Bremerhaven statt Hamburg), Rest der Route muss frei sein. Nur mit Ausweichroute."""
        t = self.env.now
        for r in self._routen(lieferant, notfall):
            for j, kid2 in enumerate(r.kanten):
                k2 = self.m.kanten[kid2]
                if k2.von == k.von and k2.nach != k.nach and k2.modus == k.modus:
                    rest = list(r.kanten[j:])
                    if not self.knoten_unten(k2.nach, t) and all(self.kante_nutzbar(x, t) for x in rest[1:]):
                        return rest
        return None

    # ---- Urheberschaft dieses Abschnitts: Benedict Hufgard (B) (Transportprozess)
    def _transport(self, b: dict[str, Any], q: Bezugsquelle):
        komp, menge = b["komponente"], b["menge"]
        masse_t = menge * self.m.komponenten[komp].gewicht_g * self.vf / 1e6
        notfall = b["typ"] == "notfall"
        route = self._waehle_route(q.lieferant, notfall)
        kanten = list(route.kanten)
        kosten = self.p["kosten"]
        sendung = {"sendung_id": b["id"], "komponente": komp, "menge": menge, "lieferant": q.lieferant,
                   "route": route.id, "typ": b["typ"], "abschnitte": []}
        zoll_offen = self.m.knoten[q.lieferant].land not in EU_LAENDER
        pos, i = q.lieferant, 0
        while pos != self.werk:
            kid = kanten[i]
            k = self.m.kanten[kid]
            iv = k.abfahrt_intervall_tage
            abfahrt = math.ceil(self.env.now / iv - 1e-9) * iv
            yield self.env.timeout(max(0.0, abfahrt - self.env.now))
            if not self.kante_nutzbar(kid, self.env.now):
                neu = self._umleitung(q.lieferant, pos, notfall)
                if neu is not None and neu != kanten[i:]:
                    self._log("umleitung", b["id"], "sendung", pos, menge, komponente=komp, alt=kid, neu=neu[0])
                    kanten, i = neu, 0
                else:
                    self._log("warten_sperre", b["id"], "sendung", pos, menge, komponente=komp, kante=kid)
                    yield self.env.timeout(max(0.01, self.naechste_freigabe(self.env.now) - self.env.now))
                continue
            dauer = k.km / self.speed[k.modus]
            self._log("abfahrt", b["id"], "sendung", pos, menge, komponente=komp, alt="bereit", neu="unterwegs",
                      kante=kid, kategorie="transport",
                      kosten=kosten["transport_fix_eur_sendung"][k.modus] + masse_t * k.km * kosten["transport_eur_tkm"][k.modus],
                      co2e_g=masse_t * k.km * k.co2e_g_tkm)
            t0 = self.env.now
            yield self.env.timeout(dauer)
            umleitung = (self._umleitung_unterwegs(q.lieferant, k, notfall)
                         if self.knoten_unten(k.nach, self.env.now) and self.p["massnahmen"].get("ausweichroute")
                         else None)
            if umleitung:
                # Umleitung auf See: zusätzliche Strecke zum Ausweichhafen, Kosten und CO2e dafür [A]
                k_neu = self.m.kanten[umleitung[0]]
                extra_km = float(self.p["transport"].get("umleitung_km", 150))
                self._log("umleitung", b["id"], "sendung", k.von, menge, komponente=komp, alt=kid, neu=k_neu.nach,
                          kante=umleitung[0], kategorie="transport",
                          kosten=masse_t * extra_km * kosten["transport_eur_tkm"][k.modus],
                          co2e_g=masse_t * extra_km * k.co2e_g_tkm)
                yield self.env.timeout(extra_km / self.speed[k.modus] + float(
                    self.p["transport"].get("umleitung_tage", 0.5)))
                kid, k = umleitung[0], k_neu
                kanten, i = umleitung, 0
            while self.knoten_unten(k.nach, self.env.now):
                self._log("warten_knoten", b["id"], "sendung", k.nach, menge, komponente=komp)
                yield self.env.timeout(max(0.01, self.naechste_freigabe(self.env.now) - self.env.now))
            sendung["abschnitte"].append({"kante": kid, "t_ab": t0, "t_an": self.env.now})
            pos, i = k.nach, i + 1
            self._log("ankunft", b["id"], "sendung", pos, menge, komponente=komp, alt="unterwegs", neu="angekommen",
                      kante=kid)
            if zoll_offen and (self.m.knoten[pos].zollgrenze or pos == self.werk):
                yield self.env.timeout(self._dreieck("zoll", self.p["transport"]["zoll_tage"]))
                wert = menge * self.preis(q, self.env.now)
                self._log("zollfreigabe", b["id"], "sendung", pos, menge, komponente=komp, alt="angekommen",
                          neu="verzollt", kosten=wert * self.zollsatz(komp, self.env.now), kategorie="zoll")
                zoll_offen = False
            if pos != self.werk:
                yield self.env.timeout(self._dreieck("umschlag", self.p["transport"]["umschlag_tage"][k.modus]))
                self._log("umschlag", b["id"], "sendung", pos, menge, komponente=komp, kategorie="umschlag",
                          kosten=masse_t * kosten["umschlag_eur_t"],
                          co2e_g=masse_t * self.p["emissionen"]["umschlag_kg_t"] * 1000)
        self.bestand[komp] += menge
        self.pipeline[komp] -= menge
        if b in self.offen[komp]:
            self.offen[komp].remove(b)
        if notfall:
            self.luft_offen[komp] = False
        sendung["t_ankunft"] = self.env.now
        self.sendungen.append(sendung)
        self._log("wareneingang", b["id"], "sendung", self.werk, menge, komponente=komp, alt="angekommen",
                  neu="verfuegbar")

    # ---- Urheberschaft dieses Abschnitts: Noah Scharly (N) (Eilbezug per Luftfracht, Brücke nach Lieferantenausfall, Notfallplan Ersatzlieferant Sellita)
    # ================================================================ Eilbezug
    def _im_stoerfall(self, t: float, komp: str, nachlauf: bool = True) -> bool:
        """Notfallplan Luftfracht gilt nur bei einer Versorgungsstörung dieser Komponente und im Nachlauf danach.

        Versorgungsstörung = Ausfall des Lieferanten, eines Vorlieferanten oder eines Knotens auf der regulären
        Route, oder Sperrung einer Kante dieser Route. Nachfragespitzen oder Normalbetrieb lösen keinen Flug aus.
        """
        if nachlauf and not self.p["massnahmen"].get("luft_nur_im_stoerfall", True):
            return True
        nach = float(self.p["massnahmen"].get("luft_nachlauf_tage", 90)) if nachlauf else 0.0
        q = self.aktive_quelle[komp]
        kanten = {kid for r in self._routen(q.lieferant, False) for kid in r.kanten}
        knoten = {q.lieferant, *q.tier2} | {self.m.kanten[kid].nach for kid in kanten}
        for st in self.m.stoerungen:
            if not st.start <= t <= st.ende + nach:
                continue
            if (st.typ == "knoten_ausfall" and st.ziel in knoten) or (st.typ == "kanten_sperre" and st.ziel in kanten):
                return True
        return False

    def _luftfracht(self):
        mass = self.p["massnahmen"]
        while self.env.now < self.H:
            for k, pol in self.politik.items():
                if self.luft_offen[k] or pol.bedarf_tag <= 0 or not self._im_stoerfall(self.env.now, k):
                    continue
                q = self.aktive_quelle[k]
                if not self._routen(q.lieferant, notfall=True) or self.lieferant_unten(q, self.env.now):
                    continue
                if self.bestand[k] < mass["luft_ausloeser_tage"] * pol.bedarf_tag:
                    self.luft_offen[k] = True
                    self.bestelle(k, float(math.ceil(mass["luft_menge_tage"] * pol.bedarf_tag)), q, typ="notfall")
            yield self.env.timeout(1.0)

    def _luft_wiederanlauf(self, s):
        """Brückenlieferung per Luftfracht, sobald ein ausgefallener Lieferant wieder produziert.

        Während des Ausfalls kann nichts geflogen werden (es wird nichts gefertigt). Beim Wiederanlauf wird der
        Bedarf bis zur Ankunft der ersten regulären Seesendung (plus Sicherheitsbestand) abzüglich Lager und
        bereits versandter Ware als Eilauftrag per Flugzeug bestellt. Danach übernimmt wieder der Seeweg.
        """
        yield self.env.timeout(max(0.0, s.ende - self.env.now))
        for k, pol in self.politik.items():
            q = self.aktive_quelle[k]
            if s.ziel not in (q.lieferant, *q.tier2) or pol.bedarf_tag <= 0 or self.luft_offen[k]:
                continue
            if not self._routen(q.lieferant, notfall=True):
                continue
            transport_plan = max(q.wbz_min_tage - q.fertigung_tage, 0.0)
            ankunft = [self._fertigungsende(q, b["t_bestellung"], q.fertigung_tage + q.wbz_zuschlag_tage)
                       + transport_plan for b in self.offen[k]
                       if b["typ"] == "regulaer" and not b["storniert"] and b["status"] in ("fertigung", "wartet")
                       and b["lieferant"] == q.lieferant]
            if not ankunft:
                continue
            versandt = sum(b["menge"] for b in self.offen[k] if b["status"] == "transport" and not b["storniert"])
            bedarf = pol.bedarf_tag * (min(ankunft) - self.env.now) + pol.sicherheitsbestand
            menge = math.ceil(max(0.0, bedarf - self.bestand[k] - versandt))
            if menge <= 0:
                continue
            self.luft_offen[k] = True
            self._log("luft_wiederanlauf", k, "notfallplan", q.lieferant, float(menge), komponente=k,
                      alt="ausfall_beendet", neu="brueckenlieferung")
            self.bestelle(k, float(menge), q, typ="notfall")

    # ================================================================ Notfallplan Ersatzlieferant
    def _notfallplan(self, s, komp: str, primaer: Bezugsquelle, ersatz: Bezugsquelle):
        """Lieferantenwechsel bei Ausfall des Primärlieferanten (Maßnahme sellita_notfall).

        Nach der Reaktionszeit wird beim Ersatzlieferanten bestellt. Aufträge beim ausgefallenen Lieferanten,
        die noch nicht versandt sind, werden storniert (Einkaufskosten gutgeschrieben) und neu disponiert.
        Endet der Ausfall, wird wieder beim Primärlieferanten bestellt; laufende Ersatzaufträge werden geliefert.
        """
        reaktion = float(self.p["massnahmen"].get("notfall_reaktion_tage", 3))
        start = s.start + reaktion
        if start >= s.ende:
            return
        yield self.env.timeout(max(0.0, start - self.env.now))
        self.aktive_quelle[komp] = ersatz
        self.reserve_bereit_ab[ersatz.lieferant] = self.env.now + float(
            self.p["massnahmen"].get("notfall_hochlauf_tage", 0) or 0)
        self._log("lieferantenwechsel", komp, "notfallplan", ersatz.lieferant, komponente=komp,
                  alt=primaer.lieferant, neu=ersatz.lieferant)
        for b in list(self.offen[komp]):
            if b["lieferant"] == primaer.lieferant and b["status"] in ("fertigung", "wartet") and not b["storniert"]:
                b["storniert"] = True
                self.pipeline[komp] -= b["menge"]
                self.offen[komp].remove(b)
                self._log("storno", b["id"], "bestellung", primaer.lieferant, b["menge"], komponente=komp,
                          alt=b["status"], neu="storniert", kategorie="einkauf",
                          kosten=-b["menge"] * self.preis(primaer, self.env.now))
        self._pruefe_nachschub()
        if math.isfinite(s.ende) and s.ende < self.H:
            yield self.env.timeout(s.ende - self.env.now)
            self.aktive_quelle[komp] = primaer
            self._log("lieferantenwechsel", komp, "notfallplan", primaer.lieferant, komponente=komp,
                      alt=ersatz.lieferant, neu=primaer.lieferant)
            self._pruefe_nachschub()

    # ---- Urheberschaft dieses Abschnitts: Benedict Hufgard (B) (Störungen)
    # ================================================================ Störungen und Tagesabschluss
    def _stoerung(self, s):
        yield self.env.timeout(max(0.0, s.start - self.env.now))
        self._log("stoerung_beginn", s.ziel or s.typ, "stoerung", s.ziel, neu=s.typ)
        if math.isfinite(s.ende) and s.ende < self.H:
            yield self.env.timeout(s.ende - self.env.now)
            self._log("stoerung_ende", s.ziel or s.typ, "stoerung", s.ziel, alt=s.typ)

    # ---- Urheberschaft dieses Abschnitts: Philipp Maywald (P) (Tagesabschluss, Bestandswerte je Tag)
    def _tagesabschluss(self):
        yield self.env.timeout(0.999)
        while self.env.now < self.H:
            zeile = {"tag": int(math.floor(self.env.now)), "fertigware": self.fg, "in_montage": self.fg_wip,
                     "rueckstand": len(self.rueckstand)}
            wert = 0.0
            for k in self.m.komponenten:
                zeile[f"bestand_{k}"] = self.bestand[k]
                zeile[f"unterwegs_{k}"] = self.pipeline[k]
                wert += self.bestand[k] * self.m.primaerquelle(k).preis_eur
            stueckpreis = sum(self.m.primaerquelle(k).preis_eur for k in self.m.komponenten)
            zeile["bestandswert_eur"] = wert + (self.fg + self.fg_wip) * stueckpreis
            zeile["komponenten_stueck"] = sum(self.bestand.values())
            self.tage.append(zeile)
            yield self.env.timeout(1.0)

    # ---- Urheberschaft dieses Abschnitts: Benedict Hufgard (B) (Simulationslauf, Laufstart mit Konfigurations-Hash)
    # ================================================================ Lauf
    def lauf(self) -> Ergebnis:
        from . import __version__
        self._log("laufstart", self.m.konfig_hash, "lauf", self.werk, alt=f"version {__version__}",
                  neu=f"seed {self.p['simulation']['seed']} replikation {self.rep}")
        self.env.process(self._nachfrage())
        self.env.process(self._montage())
        self.env.process(self._tagesabschluss())
        if self.p["massnahmen"].get("luftfracht_notfall"):
            self.env.process(self._luftfracht())
        for s in self.m.stoerungen:
            self.env.process(self._stoerung(s))
            if (s.typ == "knoten_ausfall" and self.p["massnahmen"].get("luftfracht_notfall")
                    and self.p["massnahmen"].get("luft_bei_wiederanlauf", True)
                    and math.isfinite(s.ende) and s.ende < self.H):
                self.env.process(self._luft_wiederanlauf(s))
            if s.typ == "knoten_ausfall" and self.p["massnahmen"].get("sellita_notfall"):
                for k in self.m.komponenten:
                    q, ersatz = self.m.primaerquelle(k), self.m.alternativquelle(k)
                    if ersatz is not None and s.ziel in (q.lieferant, *q.tier2):
                        self.env.process(self._notfallplan(s, k, q, ersatz))
        self._pruefe_nachschub()
        self.env.run(until=self.H)
        fenster = [{"typ": s.typ, "ziel": s.ziel, "start": s.start, "ende": s.ende} for s in self.m.stoerungen]
        return Ergebnis(
            modell=self.m, replikation=self.rep, deterministisch=self.det,
            log=pd.DataFrame(self.log), auftraege=pd.DataFrame(self.auftraege,
                                                               columns=["auftrag_id", "t_eingang", "t_versand", "t_zustellung",
                                                                        "kunde", "lat", "lon", "km"]),
            tage=pd.DataFrame(self.tage), sendungen=self.sendungen, politik=self.politik, stoerungsfenster=fenster,
        )


def simuliere(modell: Modell, replikation: int = 0, deterministisch: bool = False) -> Ergebnis:
    """Schnittstelle run(modell, szenario, seed) aus dem Konzept: ein Lauf, ein Ereignisprotokoll."""
    return Simulation(modell, replikation, deterministisch).lauf()
