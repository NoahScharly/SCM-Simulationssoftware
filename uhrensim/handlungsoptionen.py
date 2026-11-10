# Urheberschaft: Ferdinand Lautenbach (F); KI-gestützt (Claude), siehe KI_NACHWEIS.md
"""Handlungsoptionen je Szenario automatisch vergleichen und eine Empfehlung ableiten (Modul handlungsoptionen).


Ablauf in der Oberfläche:
    1. Normalbetrieb (S0) und die gewählten Störungsszenarien werden festgelegt.
    2. Für jedes Szenario wird jede Handlungsoption aus base.yaml (Block handlungsoptionen) simuliert,
       alle mit denselben Zufallszahlen (Common Random Numbers).
    3. Je Szenario: Vergleich gegen "Ohne Maßnahme", Vor- und Nachteile, Empfehlung nach Priorität.
    4. Je Option: Break-even-Wahrscheinlichkeit p* (lohnt sich die Vorsorge, wenn die Störung mit p eintritt?).
    5. Bandbreite: Stärke der Störung variieren und prüfen, ab wann die Empfehlung kippt.

Empfehlungslogik (transparent, keine Blackbox):
    Stufe 1: Mindestanforderungen (Standard: Lieferfähigkeit >= 95 %, Lieferzeit 95 % <= 3 Tage, einstellbar).
    Stufe 2: Priorisierung durch Reihenfolge der Zielkriterien, daraus Rank-Order-Centroid-Gewichte.
    Erfüllt keine Option alle Anforderungen, wird die mit dem kleinsten Abstand zu den Anforderungen empfohlen.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .data import KONFIG_STANDARD, lade_modell
from .evaluation import RICHTUNG, pareto_front
from .experiment import _setze_pfad, fuehre_aus

REFERENZ = "keine"
KENNZAHLEN = ("beta", "kosten_pro_uhr", "co2e_pro_uhr_kg", "lieferzeit_p95", "fehlteil_tage",
              "luftfracht_sendungen", "kapital_gebunden_eur", "kapitalbindung_pro_uhr", "bereitschaft_pro_uhr",
              "aushilfe_tage")
# Schwellen, ab denen eine Differenz als Vor- oder Nachteil gilt (darunter: Rauschen bzw. keine Wirkung)
SCHWELLEN = {"beta": 0.005, "kosten_pro_uhr": 0.10, "co2e_pro_uhr_kg": 0.0005, "lieferzeit_p95": 0.5,
             "fehlteil_tage": 0.5, "kapital_gebunden_eur": 1000.0}


def lade_entscheidungsregel(konfig_ordner: str | Path = KONFIG_STANDARD) -> tuple[dict[str, float], list[str]]:
    """Standard-Anforderungen und -Priorisierung aus base.yaml (Block entscheidung)."""
    e = lade_modell(None, None, konfig_ordner).param.get("entscheidung") or {}
    anf = {k: float(v) for k, v in (e.get("anforderungen") or {"beta": 0.95, "lieferzeit_p95": 3}).items()}
    prio = list(e.get("prioritaet") or ["beta", "kosten_pro_uhr", "co2e_pro_uhr_kg", "lieferzeit_p95"])
    return anf, [k for k in prio if k in KRITERIEN]


def lade_optionen(konfig_ordner: str | Path = KONFIG_STANDARD) -> list[dict[str, Any]]:
    """Handlungsoptionen aus base.yaml. Die Referenz "keine" steht immer zuerst."""
    roh = lade_modell(None, None, konfig_ordner).param.get("handlungsoptionen") or []
    optionen = [dict(o) for o in roh]
    if not any(o.get("id") == REFERENZ for o in optionen):
        optionen.insert(0, {"id": REFERENZ, "name": "Ohne Maßnahme", "kurz": "Ohne Maßnahme", "massnahmen": {}})
    optionen.sort(key=lambda o: 0 if o["id"] == REFERENZ else 1)
    for o in optionen:
        o.setdefault("kurz", o.get("name", o["id"]))
        o.setdefault("massnahmen", {})
    return optionen


def _overrides_mit(overrides: dict[str, Any] | None, option: dict[str, Any]) -> dict[str, Any]:
    ov = copy.deepcopy(overrides or {})
    ov["massnahmen"] = copy.deepcopy(option.get("massnahmen") or {})
    return ov


def relevante_ids(szenario: str | Path | dict) -> list[str] | None:
    """Relevante Optionen laut Szenariodatei; None = alle (Normalbetrieb ohne Störung: dort braucht jede Option
    ihre Mehrkosten im Normalbetrieb für die Break-even-Wahrscheinlichkeit p*)."""
    import yaml
    if isinstance(szenario, dict):
        roh = szenario
    else:
        with open(szenario, encoding="utf-8") as f:
            roh = yaml.safe_load(f) or {}
    if not roh.get("stoerungen") or not roh.get("relevante_optionen"):
        return None
    ids = list(roh["relevante_optionen"])
    return ids if REFERENZ in ids else [REFERENZ, *ids]


def _job(pfad: str | Path, overrides: dict[str, Any], replikationen: int | None) -> dict[str, Any]:
    """Ein Szenario mit einer Option (auf Modulebene, damit es in einem eigenen Prozess laufen kann)."""
    return fuehre_aus(pfad, overrides, replikationen)


class _OhneHauptskript:
    """Neue Rechenprozesse starten unter Windows, indem sie das Hauptskript neu laden. Unter Streamlit ist das die
    ganze App: Jeder Prozess würde die Oberfläche ohne Browser ausführen ("missing ScriptRunContext") und Zeit
    verlieren. Solange der Prozesspool startet, wird das Hauptskript deshalb vor den Kindprozessen verborgen; sie
    brauchen nur das Paket uhrensim."""

    def __enter__(self):
        import sys
        self.main = sys.modules.get("__main__")
        self.alt = {k: getattr(self.main, k) for k in ("__file__", "__spec__") if hasattr(self.main, k)}
        for k in self.alt:
            setattr(self.main, k, None)
        return self

    def __exit__(self, *args):
        for k, v in self.alt.items():
            setattr(self.main, k, v)
        return False


def _prozesse(anzahl_jobs: int, prozesse: int | None) -> int:
    import os
    if prozesse is not None:
        return max(1, int(prozesse))
    return max(1, min(anzahl_jobs, os.cpu_count() or 1, 8))


def vergleiche_optionen(
    szenarien: list[str | Path],
    optionen: list[dict[str, Any]] | None = None,
    overrides: dict[str, Any] | None = None,
    replikationen: int | None = None,
    fortschritt: Callable[[int, int], None] | None = None,
    nur_relevante: bool = False,
    prozesse: int | None = 1,
) -> dict[str, Any]:
    """Simuliert die Optionen auf den Szenarien.

    nur_relevante: je Szenario nur die Optionen aus 'relevante_optionen' (Normalbetrieb immer alle).
    prozesse: Anzahl paralleler Prozesse (None = automatisch, alle Kerne, höchstens 8; 1 = nacheinander).
        Jeder Lauf hat seinen festen Seed, parallel und nacheinander liefern deshalb dieselben Zahlen. Klappt die
        Parallelisierung nicht (z. B. eingeschränkte Umgebung), wird automatisch nacheinander gerechnet.

    Liefert:
        detail:      eine Zeile je (Szenario, Option) mit Mittelwerten, KI und gepaarten Differenzen zu "Ohne Maßnahme"
        ergebnisse:  {(szenario_id, option_id): Rückgabe von fuehre_aus} für Karte und Diagramme
        optionen:    die verwendeten Optionen
    """
    optionen = optionen or lade_optionen()
    jobs: list[tuple[str | Path, dict[str, Any]]] = []
    for pfad in szenarien:
        erlaubt = relevante_ids(pfad) if nur_relevante else None
        jobs += [(pfad, opt) for opt in optionen if erlaubt is None or opt["id"] in erlaubt]
    gesamt = len(jobs)
    roh: list[dict[str, Any] | None] = [None] * gesamt
    n_proz = _prozesse(gesamt, prozesse)
    fertig = 0
    if n_proz > 1 and gesamt > 1:
        try:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            # "spawn" auf allen Systemen wie unter Windows: saubere Kindprozesse, gleiches Verhalten überall
            with ProcessPoolExecutor(max_workers=n_proz, mp_context=multiprocessing.get_context("spawn")) as pool:
                with _OhneHauptskript():        # nur solange die Prozesse starten (beim Einreichen der Aufgaben)
                    zukunft = {pool.submit(_job, pfad, _overrides_mit(overrides, opt), replikationen): i
                               for i, (pfad, opt) in enumerate(jobs)}
                for z in as_completed(zukunft):
                    roh[zukunft[z]] = z.result()
                    fertig += 1
                    if fortschritt:
                        fortschritt(fertig, gesamt)
        except Exception:                                   # Rückfall: nacheinander rechnen
            roh, fertig = [None] * gesamt, 0
    for i, (pfad, opt) in enumerate(jobs):
        if roh[i] is None:
            roh[i] = _job(pfad, _overrides_mit(overrides, opt), replikationen)
            fertig += 1
            if fortschritt:
                fortschritt(fertig, gesamt)
    ergebnisse: dict[tuple[str, str], dict[str, Any]] = {}
    for (pfad, opt), e in zip(jobs, roh):
        ergebnisse[(e["szenario_id"], opt["id"])] = e

    zeilen = []
    for (sid, oid), e in ergebnisse.items():
        z = e["zusammenfassung"]
        ref = ergebnisse.get((sid, REFERENZ))
        zeile: dict[str, Any] = {"szenario": sid, "szenario_name": e["szenario_name"], "option": oid}
        for kz in z.index:
            zeile[kz] = float(z.loc[kz, "mittel"])
            zeile[f"{kz}_ki95"] = float(z.loc[kz, "ki95"])
        for kz in KENNZAHLEN:
            if ref is None or kz not in e["laeufe"]:
                zeile[f"delta_{kz}"] = math.nan
                continue
            d = (e["laeufe"][kz] - ref["laeufe"][kz]).dropna()   # gepaart: gleiche Replikationsnummern
            zeile[f"delta_{kz}"] = float(d.mean()) if not d.empty else math.nan
        zeilen.append(zeile)
    return {"detail": pd.DataFrame(zeilen), "ergebnisse": ergebnisse, "optionen": optionen}


# Zielkriterien, aus denen in der Oberfläche gewählt und priorisiert wird: Name, Einheit, Richtung (+1 = mehr ist besser)
KRITERIEN: dict[str, tuple[str, str, int]] = {
    "beta": ("Lieferfähigkeit", "%", +1),
    "kosten_pro_uhr": ("Kosten je Uhr", "€", -1),
    "co2e_pro_uhr_kg": ("CO2e je Uhr", "g", -1),
    "lieferzeit_p95": ("Lieferzeit (95 %)", "Tage", -1),
    "kapital_gebunden_eur": ("Gebundenes Kapital", "€", -1),   # optional, Standard aus (Liquidität)
}
# Montagestillstand ist kein Zielkriterium mehr : Seine Folgen stecken schon in
# Lieferfähigkeit und Kosten, als eigenes Kriterium würde er doppelt zählen. Er bleibt als Kennzahl sichtbar.

# Relevanzschwellen der Rangfolge: Unterschiede darunter gelten als gleichwertig (Rauschen oder praktisch ohne
# Bedeutung) und entscheiden die Rangfolge nicht.
RELEVANZ = {"beta": 0.005, "kosten_pro_uhr": 1.0, "co2e_pro_uhr_kg": 0.005, "lieferzeit_p95": 0.5,
            "kapital_gebunden_eur": 5000.0}
# Reihenfolge, nach der Gleichstände aufgelöst werden, wenn alle gewählten Kriterien gleichwertig sind
GLEICHSTAND_FOLGE = ["kosten_pro_uhr", "beta", "co2e_pro_uhr_kg", "lieferzeit_p95", "kapital_gebunden_eur"]


def richtung(kz: str) -> int:
    return KRITERIEN[kz][2] if kz in KRITERIEN else RICHTUNG.get(kz, -1)


def roc_gewichte(n: int) -> list[float]:
    """Rank-Order-Centroid-Gewichte (Barron und Barrett 1996): aus einer reinen Reihenfolge abgeleitete Gewichte.

    w_i = 1/n * Summe_{k=i..n} 1/k. Beispiel n = 4: 52 %, 27 %, 15 %, 6 %.
    """
    return [sum(1 / k for k in range(i, n + 1)) / n for i in range(1, n + 1)]


def rangiere(df: pd.DataFrame, anforderungen: dict[str, float], prioritaet: list[str],
             spalten: dict[str, str] | None = None) -> pd.DataFrame:
    """Zweistufige, offengelegte Rangfolge. Rang 1 = Empfehlung.

    Stufe 1 (Mindestanforderungen): zulässig ist eine Option, wenn sie jede Anforderung erfüllt
        (z. B. Lieferfähigkeit >= 95 %, Lieferzeit 95 % <= 3 Tage). spalten erlaubt, eine Anforderung gegen eine
        andere Spalte zu prüfen (robust: schlechtestes Szenario statt Mittelwert).
    Stufe 2 (Priorisierung): die Reihenfolge der Kriterien ergibt ROC-Gewichte. Jedes Kriterium wird über die
        zulässigen Optionen auf 0 (bester Wert) bis 1 (schlechtester Wert) normiert, der gewichtete Abstand zum
        Besten ist die Punktzahl (kleiner = besser).
    Unzulässige Optionen stehen dahinter, sortiert nach ihrem relativen Abstand zu den Anforderungen.
    """
    spalten = spalten or {}
    out = df.copy().reset_index(drop=True)
    verletzt = [[] for _ in range(len(out))]
    abstand = pd.Series(0.0, index=out.index)
    for kz, grenze in (anforderungen or {}).items():
        sp = spalten.get(kz, kz)
        werte = out[sp].astype(float)
        fehl = (grenze - werte) if richtung(kz) > 0 else (werte - grenze)
        fehl = fehl.clip(lower=0).where(fehl > 1e-9, 0.0)
        abstand += fehl / (abs(grenze) or 1.0)
        for i in out.index[fehl > 0]:
            verletzt[i].append(KRITERIEN.get(kz, (kz,))[0])
    out["zulaessig"] = abstand <= 0
    out["verletzt"] = [", ".join(v) for v in verletzt]
    out["abstand_anforderung"] = abstand

    gewichte = dict(zip(prioritaet, roc_gewichte(len(prioritaet))))
    basis = out[out.zulaessig] if out.zulaessig.any() else out
    punkte = pd.Series(0.0, index=out.index)
    for kz, w in gewichte.items():
        x = out[kz].astype(float)
        bx = basis[kz].astype(float)
        bester, schlechtester = (bx.max(), bx.min()) if richtung(kz) > 0 else (bx.min(), bx.max())
        schwelle = RELEVANZ.get(kz, SCHWELLEN.get(kz, 0.0))
        # Wer weniger als die Relevanzschwelle vom Besten entfernt ist, gilt als gleich gut (0). Sonst Abstand zum
        # Besten relativ zur Spanne der zulässigen Optionen (mindestens die Schwelle).
        abst = (x - bester).abs()
        spanne = max(abs(schlechtester - bester), schwelle)
        norm = (abst / spanne).where(abst >= schwelle, 0.0) if spanne > 1e-12 else pd.Series(0.0, index=out.index)
        out[f"norm_{kz}"] = norm
        punkte += w * norm
    out["punkte"] = punkte.round(9)
    out["pareto"] = pareto_front(out, list(gewichte)) if gewichte else False
    # Gleichstand: gleiche Punkte (alle gewählten Kriterien gleichwertig). Dann entscheiden weitere Kriterien in
    # fester Folge (zuerst Kosten), damit nie die Reihenfolge der Liste den Ausschlag gibt.
    folge = [k for k in GLEICHSTAND_FOLGE if k in out]
    hilfs = []
    for k in folge:
        out[f"_gs_{k}"] = out[k].astype(float) * -richtung(k)    # kleiner = besser
        hilfs.append(f"_gs_{k}")
    out = out.sort_values(["zulaessig", "abstand_anforderung", "punkte", *hilfs],
                          ascending=[False, True, True, *[True] * len(hilfs)])
    out = out.drop(columns=hilfs)
    out["rang"] = range(1, len(out) + 1)
    erster = out.iloc[0] if len(out) else None
    out["gleichauf"] = False
    if erster is not None and bool(erster.zulaessig) and prioritaet:
        out["gleichauf"] = (out.zulaessig & (out.punkte - erster.punkte).abs().lt(1e-9)
                            & out.rang.gt(1)) | False
        if out.gleichauf.any():
            out.loc[out.rang == 1, "gleichauf"] = True
    out.attrs["gewichte"] = gewichte
    return out


PROFILE: dict[str, list[str]] = {
    "Lieferfähigkeit zuerst": ["beta", "kosten_pro_uhr", "co2e_pro_uhr_kg", "lieferzeit_p95"],
    "Kosten zuerst": ["kosten_pro_uhr", "beta", "co2e_pro_uhr_kg", "lieferzeit_p95"],
    "CO2e zuerst": ["co2e_pro_uhr_kg", "beta", "kosten_pro_uhr", "lieferzeit_p95"],
    "Kapital zuerst": ["kapital_gebunden_eur", "beta", "kosten_pro_uhr", "co2e_pro_uhr_kg"],
}


def empfehlung_je_profil(detail: pd.DataFrame, anforderungen: dict[str, float],
                         profile: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Empfehlung je Szenario unter mehreren Prioritätsprofilen: zeigt, ob und wie die Priorität entscheidet."""
    profile = profile or PROFILE
    zeilen = []
    for sid, g in detail.groupby("szenario", sort=False):
        zeile: dict[str, Any] = {"szenario": sid}
        for name, prio in profile.items():
            prio = [k for k in prio if k in g]              # Profile mit fehlender Kennzahl rechnen ohne sie
            r = rangiere(g, anforderungen, prio)
            zeile[name] = r.iloc[0].option
            zeile[f"{name}_gleichauf"] = bool(r.iloc[0].gleichauf)
        zeile["kippt"] = len({zeile[n] for n in profile}) > 1
        zeilen.append(zeile)
    return pd.DataFrame(zeilen)


def empfehlung_je_szenario(detail: pd.DataFrame, anforderungen: dict[str, float],
                           prioritaet: list[str]) -> dict[str, pd.DataFrame]:
    """Rangfolge der Optionen getrennt für jedes Szenario."""
    return {sid: rangiere(d, anforderungen, prioritaet) for sid, d in detail.groupby("szenario", sort=False)}


def robuste_option(detail: pd.DataFrame, anforderungen: dict[str, float], prioritaet: list[str]) -> pd.DataFrame:
    """Eine Option für alle Szenarien zugleich.

    Anforderungen gelten im schlechtesten Szenario (z. B. kleinste Lieferfähigkeit, längste Lieferzeit),
    priorisiert wird nach dem Mittel über die Szenarien.
    """
    g = detail.groupby("option", sort=False)
    df = g[[k for k in KENNZAHLEN if k in detail]].mean()
    for kz in KRITERIEN:
        if kz in detail:
            df[f"{kz}_schlecht"] = g[kz].min() if richtung(kz) > 0 else g[kz].max()
    df = df.reset_index()
    df["beta_min"], df["kosten_max"] = df["beta_schlecht"], df["kosten_pro_uhr_schlecht"]
    return rangiere(df, anforderungen, prioritaet, spalten={kz: f"{kz}_schlecht" for kz in anforderungen})


def _de(x: float, n: int = 1) -> str:
    return f"{x:,.{n}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def vor_und_nachteile(zeile: pd.Series | dict[str, Any]) -> tuple[list[str], list[str]]:
    """Vor- und Nachteile einer Option gegenüber "Ohne Maßnahme" im selben Szenario (gepaarte Differenzen)."""
    z = dict(zeile)
    vor, nach = [], []

    def d(kz: str) -> float:
        v = z.get(f"delta_{kz}", math.nan)
        return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

    regeln = [  # Kennzahl, Text für Verbesserung, Text für Verschlechterung, Formatierung
        ("beta", "Lieferfähigkeit +{} Pp", "Lieferfähigkeit {} Pp", lambda v: _de(v * 100)),
        ("fehlteil_tage", "{} Tage weniger Montagestillstand", "{} Tage mehr Montagestillstand", lambda v: _de(abs(v))),
        ("lieferzeit_p95", "Lieferzeit (95 %) {} Tage kürzer", "Lieferzeit (95 %) {} Tage länger", lambda v: _de(abs(v))),
        ("kosten_pro_uhr", "{} € je Uhr günstiger", "+{} € je Uhr", lambda v: _de(abs(v), 2)),
        ("co2e_pro_uhr_kg", "{} g CO2e je Uhr weniger", "+{} g CO2e je Uhr", lambda v: _de(abs(v) * 1000)),
        ("kapital_gebunden_eur", "{} € weniger Kapital gebunden", "bindet {} € mehr Kapital", lambda v: _de(abs(v), 0)),
    ]
    for kz, gut, schlecht, fmt in regeln:
        v = d(kz)
        if abs(v) < SCHWELLEN[kz]:
            continue
        besser = (v > 0) if RICHTUNG.get(kz, -1) > 0 else (v < 0)
        (vor if besser else nach).append((gut if besser else schlecht).format(fmt(v)))
    b = z.get("bereitschaft_pro_uhr", 0.0)
    if b and not (isinstance(b, float) and math.isnan(b)) and float(b) >= 0.005:
        nach.append(f"Bereitschaftskosten {_de(float(b), 2)} € je Uhr, auch ohne Ausfall")
    a = z.get("aushilfe_tage", 0.0)
    if a and not (isinstance(a, float) and math.isnan(a)) and float(a) >= 0.5:
        nach.append(f"{_de(float(a))} Arbeitstage Aushilfe")
    if z.get("luftfracht_sendungen", 0) and float(z["luftfracht_sendungen"]) >= 0.5:
        nach.append(f"{_de(float(z['luftfracht_sendungen']))} Eilsendungen per Flugzeug")
    return vor, nach


def wirkt(zeile: pd.Series | dict[str, Any]) -> bool:
    """False, wenn sich keine Kennzahl spürbar von "Ohne Maßnahme" unterscheidet (Notfallplan nicht ausgelöst).

    Feste Bereitschaftskosten einer Reservequelle zählen dabei nicht: Sie fallen auch an, wenn der Notfallplan
    nie ausgelöst wird, und werden getrennt als Nachteil genannt.
    """
    z = dict(zeile)
    b = z.get("bereitschaft_pro_uhr", 0.0)
    if b is not None and not (isinstance(b, float) and math.isnan(b)) and "delta_kosten_pro_uhr" in z:
        dk = z["delta_kosten_pro_uhr"]
        if dk is not None and not (isinstance(dk, float) and math.isnan(dk)):
            z["delta_kosten_pro_uhr"] = float(dk) - float(b)
    z["bereitschaft_pro_uhr"] = 0.0
    v, n = vor_und_nachteile(z)
    return bool(v or n)


# --------------------------------------------------------------------------- Break-even p*
def break_even(detail: pd.DataFrame, basis: str = "S0") -> pd.DataFrame:
    """Break-even-Wahrscheinlichkeit p* je (Szenario, Option), Entscheidungsregel p · vermiedener Schaden > Mehrkosten.

    Eine Vorsorge-Option kostet im Normalbetrieb M (Mehrkosten je Jahr gegenüber "Ohne Maßnahme" in S0) und spart
    im Störungsjahr E (Kosten "Ohne Maßnahme" minus Kosten mit Option im Szenario, inkl. Verspätungskosten).
    Kosten werden je Uhr verglichen und auf die Jahresnachfrage hochgerechnet: Ohne Maßnahme werden im
    Störungsjahr weniger Uhren ausgeliefert, die Gesamtkosten des Jahres wären deshalb scheinbar niedriger.
    Tritt die Störung in einem Jahr mit Wahrscheinlichkeit p ein, lohnt sich die Option, wenn
        p · E  >  (1 − p) · M     also     p  >  p* = M / (M + E).
    Die Eintrittswahrscheinlichkeit wird nicht geschätzt, die Software zeigt nur die Schwelle p*.
    Bewertung:
        "ohne"    M und E praktisch null: Option wirkt in diesem Szenario nicht und kostet nichts
        "immer"   M <= 0 und E >= 0: günstiger im Normalbetrieb und nicht schlechter im Szenario
        "nie"     E <= 0 und M >= 0: hilft im Szenario nicht und kostet etwas
        "ab"      lohnt sich ab p > p*
        "bis"     M < 0, E < 0: spart im Normalbetrieb, verliert im Szenario, lohnt bis p < p*
    """
    zeilen = []
    if "kosten_pro_uhr" not in detail:
        return pd.DataFrame(columns=["szenario", "option", "mehrkosten_normal", "ersparnis_stoerung", "p_stern",
                                     "p_art"])
    idx = detail.set_index(["szenario", "option"])
    k = idx["kosten_pro_uhr"]
    menge = idx["auftraege"] if "auftraege" in idx else pd.Series(1.0, index=idx.index)
    for (sid, oid), _ in k.items():
        if sid == basis or oid == REFERENZ or (basis, oid) not in k or (basis, REFERENZ) not in k \
                or (sid, REFERENZ) not in k:
            continue
        M = float((k[(basis, oid)] - k[(basis, REFERENZ)]) * menge[(basis, REFERENZ)])
        E = float((k[(sid, REFERENZ)] - k[(sid, oid)]) * menge[(sid, REFERENZ)])
        # Differenzen unter 0,05 % der Jahreskosten gelten als null (Rauschen der Simulation)
        tol = max(1.0, 0.0005 * float(k[(basis, REFERENZ)] * menge[(basis, REFERENZ)]))
        M0, E0 = (0.0 if abs(M) < tol else M), (0.0 if abs(E) < tol else E)
        if M0 == 0 and E0 == 0:
            art, p = "ohne", math.nan
        elif M0 <= 0 and E0 >= 0:
            art, p = "immer", 0.0
        elif E0 <= 0 and M0 >= 0:
            art, p = "nie", math.nan
        elif M0 > 0 and E0 > 0:
            art, p = "ab", M0 / (M0 + E0)
        else:
            art, p = "bis", M0 / (M0 + E0)
        zeilen.append({"szenario": sid, "option": oid, "mehrkosten_normal": M, "ersparnis_stoerung": E,
                       "p_stern": p, "p_art": art})
    return pd.DataFrame(zeilen)


def p_text(art: str, p: float) -> str:
    """Lesbare Fassung von p* für die Oberfläche."""
    if art == "ohne":
        return "ohne Wirkung (wird nicht ausgelöst, kostet nichts)"
    if art == "immer":
        return "lohnt sich immer (keine Mehrkosten im Normalbetrieb)"
    if art == "nie":
        return "lohnt sich nie"
    pct = _de(p * 100, 1 if p < 0.1 else 0)
    if art == "ab":
        return f"lohnt sich ab {pct} % Eintrittswahrscheinlichkeit pro Jahr"
    return f"lohnt sich nur bis {pct} % Eintrittswahrscheinlichkeit pro Jahr"


# --------------------------------------------------------------------------- Bandbreite
def bandbreite_definition(szenario: str | Path | dict) -> tuple[str, list[Any]] | None:
    """Bandbreite aus der Szenariodatei (Block sensitivitaet), z. B. ('stoerungen.0.dauer_tage', [15, 30, 60])."""
    import yaml
    if isinstance(szenario, dict):
        roh = szenario
    else:
        with open(szenario, encoding="utf-8") as f:
            roh = yaml.safe_load(f) or {}
    sens = roh.get("sensitivitaet") or {}
    if not sens:
        return None
    pfad, werte = next(iter(sens.items()))
    return str(pfad), list(werte)


def bandbreite(
    szenario: str | Path,
    optionen: list[dict[str, Any]],
    basis_detail: pd.DataFrame,
    anforderungen: dict[str, float],
    prioritaet: list[str],
    overrides: dict[str, Any] | None = None,
    replikationen: int | None = None,
    fortschritt: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Rechnet ein Szenario für jeden Wert der Bandbreite durch und zeigt, ab wann die Empfehlung kippt.

    basis_detail: Zeilen des Normalbetriebs S0 (für p*). Liefert je (Wert, Option) Kennzahlen, Rang und p*.
    """
    import yaml
    defi = bandbreite_definition(szenario)
    if defi is None:
        return pd.DataFrame()
    pfad, werte = defi
    with open(szenario, encoding="utf-8") as f:
        roh = yaml.safe_load(f) or {}
    gesamt, i = len(werte) * len(optionen), 0
    teile = []
    for w in werte:
        sz = _setze_pfad(roh, pfad, w)
        zeilen = []
        for opt in optionen:
            e = fuehre_aus(sz, _overrides_mit(overrides, opt), replikationen)
            z = e["zusammenfassung"]
            zeile = {"szenario": e["szenario_id"], "option": opt["id"], "wert": w}
            for kz in z.index:
                zeile[kz] = float(z.loc[kz, "mittel"])
            zeilen.append(zeile)
            i += 1
            if fortschritt:
                fortschritt(i, gesamt)
        d = pd.DataFrame(zeilen)
        r = rangiere(d, anforderungen, prioritaet)
        spalten = ["szenario", "option", "kosten_pro_uhr", "auftraege"]
        be = break_even(pd.concat([basis_detail[spalten], d[spalten]], ignore_index=True))
        r = r.merge(be[["option", "p_stern", "p_art"]], on="option", how="left")
        teile.append(r)
    out = pd.concat(teile, ignore_index=True)
    out.attrs["parameter"] = pfad
    return out
