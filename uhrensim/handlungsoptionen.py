"""Handlungsoptionen je Szenario automatisch vergleichen und eine Empfehlung ableiten (Modul handlungsoptionen).

Kennzeichnung: G (Ablauf und Fachlogik N. Scharly, Code mit Claude Opus 5.5 umgesetzt), siehe KI_NACHWEIS.md.

Ablauf in der Oberfläche:
    1. Normalbetrieb (S0) und die gewählten Störungsszenarien werden festgelegt.
    2. Für jedes Szenario wird jede Handlungsoption aus base.yaml (Block handlungsoptionen) simuliert,
       alle mit denselben Zufallszahlen (Common Random Numbers).
    3. Je Szenario: Vergleich gegen "Ohne Maßnahme", Vor- und Nachteile, Empfehlung nach Priorität.
    4. Über alle Szenarien: robuste Empfehlung (Mindest-Lieferfähigkeit im schlechtesten Szenario).

Empfehlungslogik (transparent, keine Blackbox):
    Stufe 1: zulässig ist eine Option, wenn die Lieferfähigkeit beta den Mindestwert erreicht.
    Stufe 2: zulässige Optionen werden nach der gewählten Priorität sortiert (z. B. Kosten, dann CO2e).
    Erreicht keine Option den Mindestwert, wird die Option mit der höchsten Lieferfähigkeit empfohlen.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .data import KONFIG_STANDARD, lade_modell
from .evaluation import RICHTUNG, pareto_front
from .experiment import fuehre_aus

REFERENZ = "keine"
KENNZAHLEN = ("beta", "kosten_pro_uhr", "co2e_pro_uhr_kg", "lieferzeit_p95", "fehlteil_tage",
              "luftfracht_sendungen", "kapital_gebunden_eur", "kapitalbindung_pro_uhr")
# Schwellen, ab denen eine Differenz als Vor- oder Nachteil gilt (darunter: Rauschen bzw. keine Wirkung)
SCHWELLEN = {"beta": 0.005, "kosten_pro_uhr": 0.10, "co2e_pro_uhr_kg": 0.0005, "lieferzeit_p95": 0.5,
             "fehlteil_tage": 0.5, "kapital_gebunden_eur": 1000.0}


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


def vergleiche_optionen(
    szenarien: list[str | Path],
    optionen: list[dict[str, Any]] | None = None,
    overrides: dict[str, Any] | None = None,
    replikationen: int | None = None,
    fortschritt: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Simuliert jede Option auf jedem Szenario.

    Liefert:
        detail:      eine Zeile je (Szenario, Option) mit Mittelwerten, KI und gepaarten Differenzen zu "Ohne Maßnahme"
        ergebnisse:  {(szenario_id, option_id): Rückgabe von fuehre_aus} für Karte und Diagramme
        optionen:    die verwendeten Optionen
    """
    optionen = optionen or lade_optionen()
    gesamt, i = len(szenarien) * len(optionen), 0
    ergebnisse: dict[tuple[str, str], dict[str, Any]] = {}
    for pfad in szenarien:
        for opt in optionen:
            e = fuehre_aus(pfad, _overrides_mit(overrides, opt), replikationen)
            ergebnisse[(e["szenario_id"], opt["id"])] = e
            i += 1
            if fortschritt:
                fortschritt(i, gesamt)

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


def rangiere(df: pd.DataFrame, ziel_beta: float, prioritaet: list[str], beta_spalte: str = "beta") -> pd.DataFrame:
    """Zweistufige Rangfolge: erst Mindest-Lieferfähigkeit, dann Priorität. Rang 1 = Empfehlung."""
    out = df.copy()
    out["zulaessig"] = out[beta_spalte] >= ziel_beta - 1e-9
    out["pareto"] = pareto_front(out, ["kosten_pro_uhr", "co2e_pro_uhr_kg", beta_spalte])
    zul = out[out.zulaessig]
    rest = out[~out.zulaessig]
    if not zul.empty:
        schl = [f"_r{i}" for i in range(len(prioritaet))]
        for s, kz in zip(schl, prioritaet):
            zul = zul.assign(**{s: zul[kz].round(3 if kz in ("beta", beta_spalte) else 2)})
        zul = zul.sort_values(schl, ascending=[RICHTUNG.get(kz, -1) < 0 for kz in prioritaet]).drop(columns=schl)
    rest = rest.sort_values([beta_spalte, "kosten_pro_uhr"], ascending=[False, True])
    out = pd.concat([zul, rest])
    out["rang"] = range(1, len(out) + 1)
    return out


def empfehlung_je_szenario(detail: pd.DataFrame, ziel_beta: float, prioritaet: list[str]) -> dict[str, pd.DataFrame]:
    """Rangfolge der Optionen getrennt für jedes Szenario."""
    return {sid: rangiere(d.reset_index(drop=True), ziel_beta, prioritaet)
            for sid, d in detail.groupby("szenario", sort=False)}


def robuste_option(detail: pd.DataFrame, ziel_beta: float, prioritaet: list[str]) -> pd.DataFrame:
    """Eine Option für alle Szenarien zugleich: Mindestwert gilt im schlechtesten Szenario (beta_min)."""
    g = detail.groupby("option", sort=False)
    df = g[[k for k in KENNZAHLEN if k in detail]].mean()
    df["beta_min"] = g["beta"].min()
    df["kosten_max"] = g["kosten_pro_uhr"].max()
    df = df.reset_index()
    prio = ["beta_min" if k == "beta" else k for k in prioritaet]
    return rangiere(df, ziel_beta, prio, beta_spalte="beta_min")


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
    if z.get("luftfracht_sendungen", 0) and float(z["luftfracht_sendungen"]) >= 0.5:
        nach.append(f"{_de(float(z['luftfracht_sendungen']))} Eilsendungen per Flugzeug")
    return vor, nach


def wirkt(zeile: pd.Series | dict[str, Any]) -> bool:
    """False, wenn sich keine Kennzahl spürbar von "Ohne Maßnahme" unterscheidet (Notfallplan nicht ausgelöst)."""
    v, n = vor_und_nachteile(zeile)
    return bool(v or n)
