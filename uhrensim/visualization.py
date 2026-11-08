"""Diagramme und animierte Karte (Plotly). Wird von der Streamlit-Oberfläche genutzt.

Kennzeichnung: G (Architektur und Fachlogik N. Scharly, Code teilweise selbst, teilweise mit Claude Opus 5.5 umgesetzt, Fehlerbehebung mit KI), siehe KI_NACHWEIS.md.

Alle Funktionen lesen nur aus Ergebnis-Objekten bzw. Kennzahlentabellen: eine Datenquelle
für Karte, Diagramme und Kennzahlen (Folie 10).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .evaluation import BESCHRIFTUNG, lieferzeiten
from .simulation import Ergebnis

# Farben im Stil des Posters
BRAUN_DUNKEL = "#3E2C23"
BRAUN = "#6B4A35"
KUPFER = "#C08A55"
GRAU = "#6D594C"
CREME = "#F8F4EC"
ROT = "#9B3B2A"
MODUS_FARBE = {"see": BRAUN_DUNKEL, "bahn": "#5B7F3A", "binnen": "#2F6F8B", "lkw": KUPFER, "luft": ROT}
MODUS_NAME = {"see": "Seeschiff", "bahn": "Bahn", "binnen": "Binnenschiff", "lkw": "Lkw", "luft": "Luftfracht"}
KOMP_FARBE = ["#3E2C23", "#C08A55", "#5B7F3A", "#2F6F8B", "#9B3B2A", "#8C6D9E", "#B0A69C"]
MONATE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


def _layout(fig: go.Figure, titel: str = "", hoehe: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=titel, font=dict(family="Cambria, Georgia, serif", size=18, color=BRAUN_DUNKEL)),
        paper_bgcolor=CREME, plot_bgcolor="white", height=hoehe,
        font=dict(family="Calibri, Arial, sans-serif", color=BRAUN_DUNKEL, size=12),
        margin=dict(l=40, r=20, t=50 if titel else 20, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#E3D7C7", zeroline=False)
    fig.update_yaxes(gridcolor="#E3D7C7", zeroline=False)
    return fig


def _datum(tag_im_jahr: float) -> str:
    enden = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    t = int(tag_im_jahr) % 365
    m = int(np.searchsorted(enden, t, side="right"))
    tag = t - (enden[m - 1] if m > 0 else 0) + 1
    return f"{tag}. {MONATE[m]}"


# --------------------------------------------------------------------------- Karte
def _polylinie(erg: Ergebnis, kid: str) -> np.ndarray:
    m = erg.modell
    k = m.kanten[kid]
    punkte = [(m.knoten[k.von].lat, m.knoten[k.von].lon), *k.wegpunkte, (m.knoten[k.nach].lat, m.knoten[k.nach].lon)]
    return np.array(punkte, dtype=float)


def _punkt_auf_linie(linie: np.ndarray, anteil: float) -> tuple[float, float]:
    seg = np.linalg.norm(np.diff(linie, axis=0), axis=1)
    kum = np.concatenate([[0.0], np.cumsum(seg)])
    ziel = min(max(anteil, 0.0), 1.0) * kum[-1]
    i = min(int(np.searchsorted(kum, ziel, side="right")) - 1, len(seg) - 1)
    f = (ziel - kum[i]) / seg[i] if seg[i] > 0 else 0.0
    p = linie[i] + f * (linie[i + 1] - linie[i])
    return float(p[0]), float(p[1])


def _sendungspositionen(erg: Ergebnis, t: float, linien: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    m = erg.modell
    pos = []
    for s in erg.sendungen:
        ab = s["abschnitte"]
        if not ab or t < ab[0]["t_ab"] or t >= s["t_ankunft"]:
            continue
        lat = lon = None
        for j, a in enumerate(ab):
            if a["t_ab"] <= t < a["t_an"]:
                lat, lon = _punkt_auf_linie(linien[a["kante"]], (t - a["t_ab"]) / max(a["t_an"] - a["t_ab"], 1e-9))
                break
            if t >= a["t_an"] and (j == len(ab) - 1 or t < ab[j + 1]["t_ab"]):
                kn = m.knoten[m.kanten[a["kante"]].nach]
                lat, lon = kn.lat, kn.lon
                break
        if lat is not None:
            pos.append({"lat": lat, "lon": lon, "komponente": s["komponente"], "menge": s["menge"],
                        "typ": s["typ"], "id": s["sendung_id"]})
    return pos


def netzkarte(erg: Ergebnis, schritt_tage: int = 2) -> go.Figure:
    """Animierte geografische Netzwerksicht: Kanten nach Verkehrsträger, Sendungen als Punkte."""
    m = erg.modell
    W, H = m.einschwingen, m.horizont
    genutzte = {k for r in m.routen.values() for route in r for k in route.kanten}
    linien = {kid: _polylinie(erg, kid) for kid in genutzte}
    fig = go.Figure()

    # statische Kanten, je Verkehrsträger eine Spur (Legende)
    for modus in MODUS_FARBE:
        lats, lons = [], []
        for kid in genutzte:
            k = m.kanten[kid]
            if k.modus != modus:
                continue
            lin = linien[kid]
            lats += list(lin[:, 0]) + [None]
            lons += list(lin[:, 1]) + [None]
        if lats:
            fig.add_trace(go.Scattergeo(lat=lats, lon=lons, mode="lines", name=MODUS_NAME[modus],
                                        line=dict(width=1.6, color=MODUS_FARBE[modus],
                                                  dash="dot" if modus == "luft" else "solid"),
                                        hoverinfo="skip", opacity=0.75))
    # Knoten
    kn = [k for k in m.knoten.values() if k.typ != "kunde"]
    fig.add_trace(go.Scattergeo(
        lat=[k.lat for k in kn], lon=[k.lon for k in kn], mode="markers", name="Standorte",
        marker=dict(size=[11 if k.typ == "werk" else 7 for k in kn],
                    color=[BRAUN if k.typ == "werk" else ("white" if k.typ != "lieferant" else BRAUN_DUNKEL) for k in kn],
                    line=dict(width=1.2, color=BRAUN_DUNKEL)),
        text=[k.name for k in kn], hoverinfo="text"))

    komps = list(m.komponenten)
    farbe = {k: KOMP_FARBE[i % len(KOMP_FARBE)] for i, k in enumerate(komps)}

    def dynamische_spuren(t: float) -> list[go.Scattergeo]:
        p = _sendungspositionen(erg, t, linien)
        gestoert_lat, gestoert_lon, gestoert_txt = [], [], []
        for s in erg.stoerungsfenster:
            if s["start"] <= t < s["ende"]:
                if s["typ"] == "knoten_ausfall":
                    k = m.knoten[s["ziel"]]
                    gestoert_lat.append(k.lat); gestoert_lon.append(k.lon)
                elif s["typ"] == "kanten_sperre":
                    lin = linien.get(s["ziel"], _polylinie(erg, s["ziel"]))
                    mitte = lin[len(lin) // 2]
                    gestoert_lat.append(mitte[0]); gestoert_lon.append(mitte[1])
                else:
                    continue
                gestoert_txt.append(f"Störung: {s['ziel']}")
        return [
            go.Scattergeo(lat=[x["lat"] for x in p], lon=[x["lon"] for x in p], mode="markers", name="Sendungen",
                          marker=dict(size=[8 if x["typ"] != "notfall" else 10 for x in p],
                                      color=[farbe[x["komponente"]] for x in p],
                                      symbol=["circle" if x["typ"] != "notfall" else "diamond" for x in p],
                                      line=dict(width=0.8, color="white")),
                          text=[f"{x['komponente']}: {x['menge']:.0f} Stück ({x['typ']})" for x in p], hoverinfo="text"),
            go.Scattergeo(lat=gestoert_lat, lon=gestoert_lon, mode="markers", name="Störung aktiv",
                          marker=dict(size=18, color="rgba(155,59,42,0.35)", line=dict(width=2, color=ROT)),
                          text=gestoert_txt, hoverinfo="text"),
        ]

    n_statisch = len(fig.data)
    for spur in dynamische_spuren(W):
        fig.add_trace(spur)
    zeiten = np.arange(W, H, schritt_tage)
    fig.frames = [go.Frame(data=dynamische_spuren(t), traces=[n_statisch, n_statisch + 1],
                           name=f"{int(t - W)}") for t in zeiten]
    schritte = [dict(method="animate", label=_datum(t - W),
                     args=[[f"{int(t - W)}"], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                    transition=dict(duration=0))]) for t in zeiten]
    fig.update_layout(
        geo=dict(projection_type="natural earth", showland=True, landcolor="#E9DFD0", showocean=True,
                 oceancolor="#F4EFE6", showcountries=True, countrycolor="#D8CBB8", coastlinecolor="#CBBBA5",
                 lataxis_range=[-50, 70], lonaxis_range=[-95, 150], bgcolor=CREME),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                          buttons=[dict(label="▶ Abspielen", method="animate",
                                        args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(steps=schritte, x=0.18, len=0.8, y=0.0, currentvalue=dict(prefix="Datum: "),
                      pad=dict(t=10))],
    )
    return _layout(fig, "Animierte Netzwerksicht", hoehe=620)


# --------------------------------------------------------------------------- Diagramme
def _stoerungsflaechen(fig: go.Figure, erg: Ergebnis, zeilen: int = 1) -> None:
    W, H = erg.modell.einschwingen, erg.modell.horizont
    for s in erg.stoerungsfenster:
        if s["typ"] in ("knoten_ausfall", "kanten_sperre"):
            x0, x1 = s["start"] - W, min(s["ende"], H) - W
            for z in range(1, zeilen + 1):
                fig.add_vrect(x0=x0, x1=x1, fillcolor=ROT, opacity=0.10, line_width=0, row=z, col=1)


def bestandsverlauf(erg: Ergebnis) -> go.Figure:
    """Bestandsreichweite je Komponente in Tagen, mit Meldebestand und Störungsfenstern."""
    m, t = erg.modell, erg.tage
    W = m.einschwingen
    t = t[t.tag >= W]
    fig = go.Figure()
    for i, (k, pol) in enumerate(erg.politik.items()):
        if pol.bedarf_tag <= 0:
            continue
        fig.add_trace(go.Scatter(x=t.tag - W, y=t[f"bestand_{k}"] / pol.bedarf_tag, mode="lines", name=k,
                                 line=dict(width=1.8, color=KOMP_FARBE[i % len(KOMP_FARBE)]),
                                 hovertemplate=f"{k}: %{{y:.0f}} Tage<extra></extra>"))
    _stoerungsflaechen(fig, erg)
    fig.update_xaxes(title="Tag im Jahr")
    fig.update_yaxes(title="Reichweite [Tage Bedarf]")
    return _layout(fig, "Bestandsreichweite der Komponenten in Heilbronn")


def bestand_einzeln(erg: Ergebnis, komponente: str) -> go.Figure:
    m, t = erg.modell, erg.tage
    W = m.einschwingen
    t = t[t.tag >= W]
    pol = erg.politik[komponente]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t.tag - W, y=t[f"bestand_{komponente}"], mode="lines", name="Bestand",
                             line=dict(color=BRAUN, width=2)))
    fig.add_trace(go.Scatter(x=t.tag - W, y=t[f"bestand_{komponente}"] + t[f"unterwegs_{komponente}"], mode="lines",
                             name="Lagerposition (Bestand + bestellt)", line=dict(color=KUPFER, width=1.5, dash="dot")))
    fig.add_hline(y=pol.meldebestand, line=dict(color=GRAU, dash="dash"),
                  annotation_text=f"Meldebestand s = {pol.meldebestand:.0f} (gilt für die Lagerposition)")
    _stoerungsflaechen(fig, erg)
    fig.update_xaxes(title="Tag im Jahr")
    fig.update_yaxes(title="Stück")
    return _layout(fig, f"Bestand {komponente}: (s, Q) = ({pol.meldebestand:.0f}, {pol.losgroesse:.0f})")


def serviceverlauf(erg: Ergebnis, fenster: int = 14) -> go.Figure:
    """Gleitende Lieferfähigkeit und Lieferzeit je Auftragstag."""
    m = erg.modell
    W, H = m.einschwingen, m.horizont
    a = erg.auftraege
    a = a[(a.t_eingang >= W) & (a.t_eingang < H)].copy()
    a["lz"] = lieferzeiten(erg, a.t_eingang.to_numpy(float), a.t_zustellung.to_numpy(float))
    a["tag"] = np.floor(a.t_eingang - W)
    zusage = m.param["distribution"]["lieferzusage_tage"]
    a["puenktlich"] = a.lz <= zusage + 1e-9
    tag = a.groupby("tag").agg(beta=("puenktlich", "mean"), lz=("lz", "mean"), n=("lz", "size"))
    roll = tag.rolling(fenster, min_periods=1).mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"Lieferfähigkeit β ({fenster}-Tage-Mittel)", "Lieferzeit Mittel [Tage]"))
    fig.add_trace(go.Scatter(x=roll.index, y=roll.beta * 100, mode="lines", line=dict(color=BRAUN, width=2),
                             name="β"), row=1, col=1)
    fig.add_hline(y=m.param["bewertung"]["mindestwerte"].get("beta", 0.95) * 100,
                  line=dict(color=ROT, dash="dash"), row=1, col=1)
    fig.add_trace(go.Scatter(x=roll.index, y=roll.lz, mode="lines", line=dict(color=KUPFER, width=2),
                             name="Lieferzeit"), row=2, col=1)
    fig.add_hline(y=zusage, line=dict(color=ROT, dash="dash"), row=2, col=1)
    _stoerungsflaechen(fig, erg, zeilen=2)
    fig.update_yaxes(title="%", row=1, col=1)
    fig.update_yaxes(title="Tage", row=2, col=1)
    fig.update_xaxes(title="Tag im Jahr", row=2, col=1)
    fig.update_layout(showlegend=False)
    return _layout(fig, "", hoehe=520)


def boxplot_vergleich(ergebnisse: dict[str, dict[str, Any]], kennzahl: str) -> go.Figure:
    """Streuung einer Kennzahl über die Replikationen je Szenario."""
    name, einheit = BESCHRIFTUNG.get(kennzahl, (kennzahl, ""))
    faktor = 100 if kennzahl == "beta" else 1
    fig = go.Figure()
    for i, (sid, e) in enumerate(ergebnisse.items()):
        y = e["laeufe"][kennzahl].replace([np.inf, -np.inf], np.nan).dropna() * faktor
        fig.add_trace(go.Box(y=y, name=sid, marker_color=BRAUN if i == 0 else KUPFER, boxmean=True,
                             hovertext=e["szenario_name"]))
    fig.update_yaxes(title=f"{name} [{einheit}]")
    fig.update_layout(showlegend=False)
    return _layout(fig, f"{name} je Szenario über alle Replikationen")


def delta_balken(tabelle: pd.DataFrame) -> go.Figure:
    """Differenz zur Basis (gepaart, Common Random Numbers) für vier Kernkriterien."""
    kz = [("delta_beta", "β [Prozentpunkte]", 100), ("delta_kosten_pro_uhr", "Kosten je Uhr [€]", 1),
          ("delta_co2e_pro_uhr_kg", "CO2e je Uhr [g]", 1000), ("delta_lieferzeit_p95", "Lieferzeit P95 [Tage]", 1)]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[k[1] for k in kz])
    t = tabelle.iloc[1:]
    for j, (spalte, _, f) in enumerate(kz, 1):
        werte = t[spalte] * f
        fig.add_trace(go.Bar(x=t.index, y=werte, marker_color=[ROT if v < 0 and spalte == "delta_beta" or
                                                               v > 0 and spalte != "delta_beta" else "#5B7F3A"
                                                               for v in werte]), row=1, col=j)
    fig.update_layout(showlegend=False)
    return _layout(fig, "Veränderung gegenüber der Basis S0", hoehe=360)


def pareto_diagramm(raster: pd.DataFrame) -> go.Figure:
    """Zielkonflikt Kosten gegen CO2e; Farbe zeigt Zulässigkeit, Rahmen die Pareto-Front."""
    fig = go.Figure()
    def label(r):
        teile = [f"Puffer {r.puffer_uhrwerk}"]
        if r.dual_sourcing: teile.append("Dual")
        if r.luftfracht_notfall: teile.append("Luft")
        return ", ".join(teile)
    for zul, farbe, name in ((False, "#B0A69C", "unzulässig (Mindestwert verfehlt)"), (True, KUPFER, "zulässig")):
        d = raster[raster.zulaessig == zul]
        fig.add_trace(go.Scatter(
            x=d.kosten_pro_uhr, y=d.co2e_pro_uhr_kg * 1000, mode="markers+text", name=name,
            text=[label(r) for r in d.itertuples()], textposition="top center", textfont=dict(size=10),
            marker=dict(size=14, color=farbe, line=dict(width=[3 if p else 0.5 for p in d.pareto], color=BRAUN_DUNKEL)),
            hovertemplate="%{text}<br>Kosten %{x:.2f} €/Uhr<br>CO2e %{y:.1f} g/Uhr<extra></extra>"))
    front = raster[raster.pareto].sort_values("kosten_pro_uhr")
    fig.add_trace(go.Scatter(x=front.kosten_pro_uhr, y=front.co2e_pro_uhr_kg * 1000, mode="lines",
                             line=dict(color=BRAUN, dash="dash"), name="Pareto-Front"))
    fig.update_xaxes(title="Kosten je Uhr [€]")
    fig.update_yaxes(title="CO2e je Uhr [g]")
    return _layout(fig, "Maßnahmen im Zielkonflikt", hoehe=480)


# --------------------------------------------------------------------------- Tabellen
def kennzahlentabelle(tabelle: pd.DataFrame) -> pd.DataFrame:
    """Numerische Sicht: Mittelwert ± 95-%-KI, formatiert."""
    spalten = ["beta", "lieferzeit_mittel", "lieferzeit_p95", "kosten_pro_uhr", "co2e_pro_uhr_kg",
               "reichweite_tage", "fehlteil_tage", "tts_min", "ttr_max"]
    out = pd.DataFrame(index=tabelle.index)
    out["Szenario"] = tabelle["name"]
    for s in spalten:
        if s not in tabelle:
            continue
        name, einheit = BESCHRIFTUNG[s]
        f = 100 if s == "beta" else (1000 if s == "co2e_pro_uhr_kg" else 1)
        einheit = "g" if s == "co2e_pro_uhr_kg" else einheit
        werte = []
        for sid in tabelle.index:
            mw, ki = tabelle.loc[sid, s], tabelle.loc[sid, f"{s}_ki95"] if f"{s}_ki95" in tabelle else math.nan
            if s == "tts_min" and (pd.isna(mw)):
                werte.append("kein Stillstand")
            elif pd.isna(mw):
                werte.append("")
            elif pd.isna(ki) or ki == 0:
                werte.append(f"{mw * f:.1f}")
            else:
                werte.append(f"{mw * f:.1f} ± {ki * f:.1f}")
        out[f"{name} [{einheit}]"] = werte
    return out
