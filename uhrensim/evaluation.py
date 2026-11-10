"""Kennzahlen aus dem Ereignisprotokoll und Mehrzielbewertung (Modul evaluation).

Kennzeichnung: G (Architektur und Fachlogik N. Scharly, Code teilweise selbst, teilweise mit Claude Opus 5.5 umgesetzt, Fehlerbehebung mit KI), siehe KI_NACHWEIS.md.

Alle Kennzahlen werden ausschließlich aus Ergebnis.log, Ergebnis.auftraege und Ergebnis.tage
berechnet. Ausgewertet wird nur der Zeitraum nach der Einschwingphase.

Kernkriterien:  kosten_pro_uhr [€], beta [-], lieferzeit_mittel / lieferzeit_p95 [Tage],
                co2e_pro_uhr_kg [kg CO2e]
Zusatz:         TTS / TTR je Störung [Tage], reichweite_tage [Tage]
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .data import lagerhaltungssatz
from .simulation import Ergebnis

# Richtung je Kennzahl: +1 größer ist besser, -1 kleiner ist besser
RICHTUNG = {
    "beta": +1, "kosten_pro_uhr": -1, "kosten_gesamt": -1, "lieferzeit_mittel": -1, "lieferzeit_p95": -1,
    "co2e_pro_uhr_kg": -1, "reichweite_tage": -1, "fehlteil_tage": -1, "beta_min": +1, "kosten_max": -1,
    "kapital_gebunden_eur": -1, "kapitalbindung_pro_uhr": -1,
}

BESCHRIFTUNG = {
    "beta": ("Lieferfähigkeit β", "%"),
    "kosten_pro_uhr": ("Kosten je Uhr", "€"),
    "kosten_gesamt": ("Gesamtkosten", "€/Jahr"),
    "lieferzeit_mittel": ("Lieferzeit Mittel", "Tage"),
    "lieferzeit_p95": ("Lieferzeit 95-%-Quantil", "Tage"),
    "co2e_pro_uhr_kg": ("CO2e je Uhr", "kg"),
    "co2e_inbound_g": ("CO2e Inbound je Uhr", "g"),
    "reichweite_tage": ("Bestandsreichweite Komponenten", "Tage"),
    "fehlteil_tage": ("Tage mit Fehlteil", "Tage"),
    "luftfracht_sendungen": ("Luftfracht-Eilsendungen", "Anzahl"),
    "tts_min": ("TTS (kürzeste)", "Tage"),
    "ttr_max": ("TTR (längste)", "Tage"),
}


def ist_arbeitstag(tag: int, einschwingen: float, arbeitstage_pro_woche: int) -> bool:
    return (tag - int(einschwingen) + 4) % 7 < arbeitstage_pro_woche


def lieferzeiten(erg: Ergebnis, t_eingang: np.ndarray, t_ende: np.ndarray) -> np.ndarray:
    """Lieferzeit in der konfigurierten Einheit. Werktage: Wochenendtage werden abgezogen."""
    p = erg.modell.param
    roh = t_ende - t_eingang
    if p["distribution"].get("lieferzeit_einheit", "werktage") != "werktage":
        return roh
    W, n = erg.modell.einschwingen, int(p["montage"]["arbeitstage_pro_woche"])
    abzug = np.zeros_like(roh)
    for i, (te, tz) in enumerate(zip(t_eingang, t_ende)):
        if not np.isfinite(tz):
            continue
        d0, d1 = int(math.floor(te)), int(math.floor(tz))
        if not ist_arbeitstag(d0, W, n):
            abzug[i] += min(math.ceil(te + 1e-12), tz) - te
        abzug[i] += sum(1 for d in range(d0 + 1, d1) if not ist_arbeitstag(d, W, n))
    return roh - abzug


def _betroffene_komponenten(erg: Ergebnis, stoerung: dict[str, Any]) -> list[str]:
    m = erg.modell
    betroffen = []
    for komp in m.komponenten:
        q = m.primaerquelle(komp)
        basis = next(r for r in m.routen[q.lieferant] if r.typ == "regulaer" and all(m.kanten[k].aktiv for k in r.kanten))
        knoten_auf_route = {m.kanten[k].von for k in basis.kanten} | {m.kanten[k].nach for k in basis.kanten}
        if stoerung["typ"] == "knoten_ausfall" and (stoerung["ziel"] in q.tier2 or stoerung["ziel"] in knoten_auf_route):
            betroffen.append(komp)
        if stoerung["typ"] == "kanten_sperre" and stoerung["ziel"] in basis.kanten:
            betroffen.append(komp)
    return betroffen


def resilienz(erg: Ergebnis) -> list[dict[str, Any]]:
    """TTS und TTR je Störung (Simchi-Levi et al. 2014, Vorlesung Kap. 9).

    TTR (Eingabe aus dem Szenario): Zeit, bis die betroffene Versorgung wieder bereitsteht.
         Knoten- oder Kantenausfall: Störungsdauer. Bei einem Lieferanten kommt die Transitzeit
         der ersten neuen Sendung hinzu.
    TTS (Ergebnis der Simulation): Zeit vom Störungsbeginn bis zum ersten Montagestillstand wegen
         einer betroffenen Komponente. Kein Stillstand bis Simulationsende: TTS = unendlich.
    Resilient, wenn TTS >= TTR.
    """
    m, log = erg.modell, erg.log
    if log.empty:
        return []
    fehlteile = log[log.ereignis == "fehlteil"]
    ergebnis = []
    for s in erg.stoerungsfenster:
        if s["typ"] not in ("knoten_ausfall", "kanten_sperre"):
            continue
        komps = _betroffene_komponenten(erg, s)
        if not komps:
            continue
        dauer = min(s["ende"], m.horizont) - s["start"]
        lieferanten_ttr = []
        for komp in komps:
            q = m.primaerquelle(komp)
            if s["typ"] == "knoten_ausfall" and s["ziel"] in (q.lieferant, *q.tier2):
                kandidat = dauer + (q.wbz_min_tage - q.fertigung_tage)  # plus Transit der ersten neuen Sendung
                lieferanten_ttr.append(kandidat)
        ttr = max(lieferanten_ttr) if lieferanten_ttr else dauer
        betrifft = fehlteile.komponente.astype(str).apply(lambda x: any(k in x.split(";") for k in komps))
        ft = fehlteile[(fehlteile.zeit >= s["start"]).to_numpy() & betrifft.to_numpy(dtype=bool)]
        tts = float(ft.zeit.min() - s["start"]) if not ft.empty else math.inf
        ergebnis.append({"stoerung": f"{s['typ']}:{s['ziel']}", "komponenten": ", ".join(komps),
                         "tts": tts, "ttr": float(ttr), "resilient": tts >= ttr})
    return ergebnis


def kennzahlen(erg: Ergebnis) -> dict[str, float]:
    """evaluate(protokoll) aus dem Konzept: alle Zielkriterien eines Laufs."""
    m, p = erg.modell, erg.modell.param
    W, H = m.einschwingen, m.horizont
    zusage = float(p["distribution"]["lieferzusage_tage"])

    # ---- Lieferfähigkeit und Lieferzeit (je Auftrag 1 Uhr: mengen- = auftragsbezogen)
    a = erg.auftraege
    a = a[(a.t_eingang >= W) & (a.t_eingang < H)] if not a.empty else a
    if a.empty:
        beta = lz_mittel = lz_p95 = math.nan
        zugestellt, verspaetet = 0, 0
    else:
        te = a.t_eingang.to_numpy(float)
        tz = a.t_zustellung.to_numpy(float)
        lz = lieferzeiten(erg, te, tz)
        offen_alter = lieferzeiten(erg, te, np.full_like(te, H))
        geliefert = np.isfinite(tz)
        puenktlich = geliefert & (lz <= zusage + 1e-9)
        sicher_spaet = (geliefert & ~puenktlich) | (~geliefert & (offen_alter > zusage))
        bewertbar = puenktlich | sicher_spaet
        beta = float(puenktlich.sum() / bewertbar.sum()) if bewertbar.any() else math.nan
        lz_mittel = float(np.mean(lz[geliefert])) if geliefert.any() else math.nan
        lz_p95 = float(np.quantile(lz[geliefert], 0.95)) if geliefert.any() else math.nan
        zugestellt, verspaetet = int(geliefert.sum()), int(sicher_spaet.sum())

    # ---- Kosten
    log = erg.log
    lf = log[(log.zeit >= W) & (log.zeit < H)] if not log.empty else log
    kosten = lf.groupby("kostenart").kosten_eur.sum().to_dict() if not lf.empty else {}
    kosten.pop("", None)
    # Einkauf als Wareneinsatz: verbaute Komponenten × mittlerer Einkaufspreis je Komponente.
    # Bestellungen nach Bestelldatum zu buchen würde Störungsjahre scheinbar billiger machen,
    # weil Einkauf nur ins Folgejahr verschoben wird.
    best = log[(log.ereignis == "bestellung") & (log.kostenart == "einkauf")] if not log.empty else log
    if not best.empty and not lf.empty:
        summe = best.groupby("komponente").agg(eur=("kosten_eur", "sum"), stk=("menge", "sum"))
        preis_satz = float((summe.eur / summe.stk).sum())
        verbaut = float(lf.loc[lf.ereignis == "montage_start", "menge"].sum())
        kosten["einkauf"] = preis_satz * verbaut
    t = erg.tage
    tf = t[(t.tag >= W) & (t.tag < H)] if not t.empty else t
    k = p["kosten"]
    kosten["kapitalbindung"] = float(tf.bestandswert_eur.sum() * lagerhaltungssatz(p) / 365.0) if not tf.empty else 0.0
    # Bereitschaftskosten einer Reservequelle (Qualifizierung, Musterfreigabe, Rahmenvertrag; Idee B. Hufgard):
    # fallen jedes Jahr an, auch wenn der Notfallplan nie ausgelöst wird
    bereit = float(p["massnahmen"].get("notfall_bereitschaft_eur_jahr", 0.0) or 0.0)
    kosten["bereitschaft"] = bereit * (H - W) / 365.0 if p["massnahmen"].get("sellita_notfall") else 0.0
    kosten["lagerplatz"] = float(tf.komponenten_stueck.sum() * k["lager_eur_stueck_tag"]) if not tf.empty else 0.0
    kosten["verspaetung"] = verspaetet * float(k["verspaetung_eur_auftrag"])
    kosten_gesamt = float(sum(kosten.values()))

    # ---- Emissionen
    if lf.empty:
        inbound_g = letzte_meile_g = 0.0
    else:
        inbound_g = float(lf[lf.kostenart.isin(["transport", "umschlag"])].co2e_g.sum())
        letzte_meile_g = float(lf[lf.kostenart == "letzte_meile"].co2e_g.sum())

    # ---- Bestand
    stueckpreis = sum(m.primaerquelle(kk).preis_eur for kk in m.komponenten)
    bedarf_tag = float(p["nachfrage"]["uhren_pro_jahr"]) / 365.0
    if not tf.empty and bedarf_tag > 0:
        komp_wert = tf.bestandswert_eur - (tf.fertigware + tf.in_montage) * stueckpreis
        reichweite = float(komp_wert.mean() / (bedarf_tag * stueckpreis))
    else:
        reichweite = math.nan

    fehlteil_tage = int(lf[lf.ereignis == "fehlteil"].objekt_id.nunique()) if not lf.empty else 0
    luft = int(((lf.ereignis == "bestellung") & lf.objekt_id.isin(
        [s["sendung_id"] for s in erg.sendungen if s["typ"] == "notfall"])).sum()) if not lf.empty else 0

    res = resilienz(erg)
    n = max(zugestellt, 1)
    ergebnis = {
        "auftraege": int(len(a)), "zugestellt": zugestellt, "verspaetet": verspaetet,
        "beta": beta, "lieferzeit_mittel": lz_mittel, "lieferzeit_p95": lz_p95,
        "kosten_gesamt": kosten_gesamt, "kosten_pro_uhr": kosten_gesamt / n if zugestellt else math.nan,
        "co2e_pro_uhr_kg": (inbound_g + letzte_meile_g) / n / 1000.0 if zugestellt else math.nan,
        "co2e_inbound_g": inbound_g / n if zugestellt else math.nan,
        # Getrennt ausgewiesen (Idee F. Lautenbach): Zustellung zum Kunden gegen Beschaffung, weil die letzte Meile
        # je Uhr ein Vielfaches des Inbound verursacht und nur über Paketdienst und Verpackung beeinflussbar ist
        "co2e_letzte_meile_g": letzte_meile_g / n if zugestellt else math.nan,
        "reichweite_tage": reichweite, "fehlteil_tage": fehlteil_tage, "luftfracht_sendungen": luft,
        "tts_min": min((r["tts"] for r in res), default=math.nan),
        "ttr_max": max((r["ttr"] for r in res), default=math.nan),
        "resilient_anteil": float(np.mean([r["resilient"] for r in res])) if res else math.nan,
    }
    for art, wert in kosten.items():
        ergebnis[f"kosten_{art}"] = float(wert)
    # Kapitalbindung sichtbar machen: Ø gebundener Bestandswert und dessen Zinskosten je Uhr
    ergebnis["kapital_gebunden_eur"] = float(tf.bestandswert_eur.mean()) if not tf.empty else math.nan
    ergebnis["kapitalbindung_pro_uhr"] = kosten["kapitalbindung"] / n if zugestellt else math.nan
    ergebnis["bereitschaft_pro_uhr"] = kosten["bereitschaft"] / n if zugestellt else math.nan
    ergebnis["aushilfe_tage"] = int((lf.ereignis == "aushilfe").sum()) if not lf.empty else 0
    return ergebnis


# --------------------------------------------------------------------------- Mehrzielbewertung
def pareto_front(df: pd.DataFrame, kriterien: list[str]) -> pd.Series:
    """True für nicht dominierte Zeilen (alle Kriterien gemäß RICHTUNG)."""
    werte = np.column_stack([df[k].to_numpy(float) * -RICHTUNG.get(k, -1) for k in kriterien])  # alles minimieren
    n = len(df)
    nicht_dominiert = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.all(np.isfinite(werte[i])):
            nicht_dominiert[i] = False
            continue
        besser_gleich = np.all(werte <= werte[i], axis=1)
        echt_besser = np.any(werte < werte[i], axis=1)
        if np.any(besser_gleich & echt_besser):
            nicht_dominiert[i] = False
    return pd.Series(nicht_dominiert, index=df.index)


def bewerte(df: pd.DataFrame, bewertung: dict[str, Any]) -> pd.DataFrame:
    """compare(kennzahlen) aus dem Konzept: Zulässigkeit, Pareto-Front und Rangfolge.

    methode = mindestwerte:   zulässig, wenn alle Mindestwerte erfüllt; Rang nach Kosten je Uhr
    methode = gewichtet:      Min-Max-normierte, gewichtete Summe (Gewichte offengelegt)
    methode = lexikografisch: Ziele nacheinander nach Rangfolge (gerundet, damit Gleichstand möglich)
    """
    out = df.copy()
    zul = pd.Series(True, index=out.index)
    for kz, grenze in (bewertung.get("mindestwerte") or {}).items():
        richtung = RICHTUNG.get(kz, +1)
        zul &= (out[kz] >= grenze) if richtung > 0 else (out[kz] <= grenze)
    out["zulaessig"] = zul
    out["pareto"] = False
    if zul.any():
        out.loc[zul, "pareto"] = pareto_front(out[zul], ["kosten_pro_uhr", "co2e_pro_uhr_kg", "beta"])

    methode = bewertung.get("methode", "mindestwerte")
    if methode == "gewichtet":
        score = pd.Series(0.0, index=out.index)
        for kz, w in bewertung["gewichte"].items():
            x = out[kz].astype(float)
            spanne = x.max() - x.min()
            norm = (x - x.min()) / spanne if spanne > 0 else x * 0
            score += w * (norm if RICHTUNG.get(kz, -1) < 0 else 1 - norm)
        out["bewertung"] = score
        out = out.sort_values(["zulaessig", "bewertung"], ascending=[False, True])
    elif methode == "lexikografisch":
        schluessel, aufsteigend = ["zulaessig"], [False]
        for kz in bewertung["rangfolge"]:
            out[f"_r_{kz}"] = out[kz].round(3 if kz == "beta" else 2)
            schluessel.append(f"_r_{kz}")
            aufsteigend.append(RICHTUNG.get(kz, -1) < 0)
        out = out.sort_values(schluessel, ascending=aufsteigend).drop(columns=[c for c in out if c.startswith("_r_")])
        out["bewertung"] = range(1, len(out) + 1)
    else:
        out = out.sort_values(["zulaessig", "kosten_pro_uhr"], ascending=[False, True])
        out["bewertung"] = out["kosten_pro_uhr"]
    out["rang"] = range(1, len(out) + 1)
    return out
