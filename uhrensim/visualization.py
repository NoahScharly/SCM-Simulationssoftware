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
BRAUN_DUNKEL = "#1E1E22"  # Asphalt
BRAUN = "#3A3D44"
KUPFER = "#E0A100"  # Rennsport-Gelb (Akzent, Empfehlung)
GRAU = "#5E6168"
CREME = "#F6F6F4"
ROT = "#C8102E"  # Rennrot (Störung, Luftfracht)
MODUS_FARBE = {"see": BRAUN_DUNKEL, "bahn": "#5B7F3A", "binnen": "#2F6F8B", "lkw": KUPFER, "luft": ROT}
MODUS_NAME = {"see": "Seeschiff", "bahn": "Bahn", "binnen": "Binnenschiff", "lkw": "Lkw", "luft": "Luftfracht"}
KOMP_FARBE = ["#1E1E22", "#E0A100", "#3B7D3A", "#2F6F8B", "#C8102E", "#8C6D9E", "#9A9CA2"]
MONATE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


def _layout(fig: go.Figure, titel: str = "", hoehe: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=titel, font=dict(family="Cambria, Georgia, serif", size=18, color=BRAUN_DUNKEL)),
        paper_bgcolor=CREME, plot_bgcolor="white", height=hoehe,
        font=dict(family="Calibri, Arial, sans-serif", color=BRAUN_DUNKEL, size=12),
        margin=dict(l=40, r=20, t=50 if titel else 20, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#DCDCD8", zeroline=False)
    fig.update_yaxes(gridcolor="#DCDCD8", zeroline=False)
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


def _sendungspositionen(erg: Ergebnis, t: float, linien: dict[str, np.ndarray],
                        schritt: float = 0.0) -> list[dict[str, Any]]:
    """Position jeder Sendung zum Zeitpunkt t.

    Kurze Lkw-Fahrten (z. B. Sellita -> Heilbronn, rund 1 Tag) wären bei 3 Tagen je Bild nie zu sehen.
    Deshalb wird eine Sendung, die im Intervall (t - schritt, t] unterwegs war, in der Mitte dieses
    Abschnitts gezeigt.
    """
    m = erg.modell
    pos = []
    for s in erg.sendungen:
        ab = s["abschnitte"]
        if not ab:
            continue
        beginn, ende = ab[0]["t_ab"], s["t_ankunft"]
        if beginn <= t < ende:
            tt = t
        elif schritt > 0 and beginn < t and ende > t - schritt:
            tt = (max(beginn, t - schritt) + min(ende, t)) / 2
        else:
            continue
        lat = lon = None
        for j, a in enumerate(ab):
            if a["t_ab"] <= tt < a["t_an"]:
                lat, lon = _punkt_auf_linie(linien[a["kante"]], (tt - a["t_ab"]) / max(a["t_an"] - a["t_ab"], 1e-9))
                break
            if tt >= a["t_an"] and (j == len(ab) - 1 or tt < ab[j + 1]["t_ab"]):
                kn = m.knoten[m.kanten[a["kante"]].nach]
                lat, lon = kn.lat, kn.lon
                break
        if lat is not None:
            pos.append({"lat": lat, "lon": lon, "komponente": s["komponente"], "menge": s["menge"],
                        "typ": s["typ"], "id": s["sendung_id"], "lieferant": s["lieferant"]})
    return pos


def _bestandsdaten(erg: Ergebnis) -> dict[str, Any]:
    """Tagesbestände, Meldebestände und Bestellzeitpunkte für die animierten Bestandsbalken.

    Alles in Reichweite (Tage Bedarf), damit 1.600 Uhrwerke und 160 Zifferblätter vergleichbar sind.
    Liest nur aus dem Ergebnis desselben Laufs wie die Karte (eine Datenquelle).
    """
    m, t = erg.modell, erg.tage.set_index("tag")
    komps = [k for k, p in erg.politik.items() if p.bedarf_tag > 0]
    namen = [komp_name(m, k) for k in komps]
    bed = {k: erg.politik[k].bedarf_tag for k in komps}
    log = erg.log
    best = log[log.ereignis == "bestellung"][["zeit", "komponente", "menge", "ort"]]
    W = m.einschwingen
    fenster = t[t.index >= W]
    xmax = max(((fenster[f"bestand_{k}"] + fenster[f"unterwegs_{k}"]) / bed[k]).max() for k in komps) if komps else 1
    return {"komps": komps, "namen": namen, "bedarf": bed, "tage": t, "bestellungen": best,
            "melde": [erg.politik[k].meldebestand / bed[k] for k in komps], "xmax": float(xmax) * 1.42,
            "xstatus": float(xmax) * 1.40}


def _bestand_spuren(bd: dict[str, Any], t: float, schritt: float) -> list[go.Bar | go.Scatter]:
    """Bestandsbalken zum Zeitpunkt t (Tagesabschluss unmittelbar vor t).

    Statt eines kurz aufblinkenden Hinweises steht rechts je Komponente ein ruhiger Status:
    "↻ in 12 T" = in etwa 12 Tagen erreicht der helle Balken den Meldebestand, dann wird nachbestellt;
    "✓ vor 3 T" = vor 3 Tagen wurde nachbestellt (wird 10 Tage lang angezeigt).
    """
    tage = bd["tage"]
    tag = int(min(max(math.ceil(t) - 1, tage.index.min()), tage.index.max()))
    z = tage.loc[tag]
    t_stand = tag + 1.0                         # Tagesabschluss wird kurz vor Tagesende geschrieben
    bestand = [z[f"bestand_{k}"] / bd["bedarf"][k] for k in bd["komps"]]
    position = [(z[f"bestand_{k}"] + z[f"unterwegs_{k}"]) / bd["bedarf"][k] for k in bd["komps"]]
    stueck = [z[f"bestand_{k}"] for k in bd["komps"]]
    b = bd["bestellungen"]
    b = b[b.zeit <= t_stand]
    letzte = b.groupby("komponente").zeit.max().to_dict() if not b.empty else {}
    status, status_farbe = [], []
    for k, p, s in zip(bd["komps"], position, bd["melde"]):
        seit = t_stand - letzte.get(k, -math.inf)
        if seit < 10:
            status.append(f"✓ vor {max(seit, 0):.0f} T")
            status_farbe.append("#5B7F3A")
        else:
            status.append(f"↻ in {max(p - s, 0):.0f} T")
            status_farbe.append(GRAU)
    txt = [f"{n:.0f} Stk · {r:.0f} T" for n, r in zip(stueck, bestand)]
    return [
        go.Bar(y=bd["namen"], x=position, orientation="h", name="Lager + unterwegs",
               marker=dict(color="rgba(192,138,85,0.30)", line=dict(width=0)), showlegend=False,
               hovertemplate="%{y}: Lager + bestellt/unterwegs = %{x:.0f} Tage<extra></extra>"),
        go.Bar(y=bd["namen"], x=bestand, orientation="h", name="Im Lager", text=txt,
               textposition="outside", cliponaxis=False, textfont=dict(size=10),
               marker=dict(color=[ROT if r < 5 else BRAUN for r in bestand]), showlegend=False,
               hovertemplate="%{y}: im Lager %{x:.0f} Tage<extra></extra>"),
        go.Scatter(y=bd["namen"], x=[bd["xstatus"]] * len(bd["komps"]), mode="text", text=status,
                   textposition="middle left", textfont=dict(size=10, color=status_farbe), showlegend=False,
                   hovertemplate="%{y}: %{text}<extra>↻ nächste Nachbestellung in etwa · ✓ zuletzt bestellt vor</extra>"),
    ]


def _quellen_verlauf(erg: Ergebnis) -> dict[str, list[tuple[float, str]]]:
    """Aktiver Lieferant je Komponente über die Zeit (aus den Ereignissen 'lieferantenwechsel')."""
    m = erg.modell
    verlauf = {k: [(-math.inf, m.primaerquelle(k).lieferant)] for k in m.komponenten if m.alternativquelle(k)}
    if not erg.log.empty:
        w = erg.log[erg.log.ereignis == "lieferantenwechsel"]
        for z in w.itertuples():
            verlauf.setdefault(z.komponente, []).append((z.zeit, z.zustand_neu))
    return verlauf


def _aktiver_lieferant(verlauf: list[tuple[float, str]], t: float) -> str:
    aktiv = verlauf[0][1]
    for zeit, lief in verlauf:
        if zeit <= t:
            aktiv = lief
    return aktiv


def _luftsendungen(erg: Ergebnis, t: float, schritt: float) -> list[dict[str, Any]]:
    """Luftfracht-Sendungen, die um t herum unterwegs sind (ein Bild Vor- und Nachlauf, damit kurze Flüge sichtbar
    bleiben, auch wenn der Flug selbst zwischen zwei Bildern liegt)."""
    aus = []
    for s in erg.sendungen:
        if s["typ"] != "notfall" or not s["abschnitte"]:
            continue
        if s["abschnitte"][0]["t_ab"] - schritt <= t <= s["t_ankunft"] + schritt:
            aus.append(s)
    return aus


def _status_text(erg: Ergebnis, verlauf: dict[str, list[tuple[float, str]]], t: float, schritt: float = 3.0) -> str:
    m = erg.modell
    teile = []
    for k, v in verlauf.items():
        aktiv = _aktiver_lieferant(v, t)
        primaer = m.primaerquelle(k).lieferant
        name = m.knoten[aktiv].name.split(" (")[0]
        land = m.knoten[aktiv].land
        if aktiv != primaer:
            seit = max(z for z, l in v if z <= t and l == aktiv)
            teile.append(f"<b>{komp_name(m, k)}</b> kommt aus: <b>{name} ({land})</b> · "
                         f"<span style='color:{ROT}'>Notfallplan aktiv seit {_datum(seit - m.einschwingen)}</span>")
        else:
            ausfall = any(s["typ"] == "knoten_ausfall" and s["start"] <= t < s["ende"]
                          and s["ziel"] in (primaer, *m.primaerquelle(k).tier2) for s in erg.stoerungsfenster)
            zusatz = f" · <span style='color:{ROT}'>Lieferant ausgefallen</span>" if ausfall else ""
            teile.append(f"<b>{komp_name(m, k)}</b> kommt aus: <b>{name} ({land})</b>{zusatz}")
    for sd in _luftsendungen(erg, t, schritt):
        stationen = [m.knoten[sd["lieferant"]].name.split(" (")[0]] + [
            ("Heilbronn" if m.kanten[a["kante"]].nach == erg.modell.werk else
             m.knoten[m.kanten[a["kante"]].nach].name.split(" (")[0].replace("Flughafen ", ""))
            for a in sd["abschnitte"]]
        status = "angekommen" if t >= sd["t_ankunft"] else f"Ankunft {_datum(sd['t_ankunft'] - m.einschwingen)}"
        teile.append(f"<span style='color:{ROT}'>✈ <b>Luftfracht</b></span>: {sd['menge']:.0f} × "
                     f"{komp_name(m, sd['komponente'])} · {' → '.join(stationen)} · {status}")
    if not erg.log.empty:
        f = erg.log[(erg.log.ereignis == "fehlteil") & (erg.log.zeit > t - schritt) & (erg.log.zeit <= t)]
        if not f.empty:
            fehlt = sorted({k for x in f.komponente.astype(str) for k in x.split(";") if k and k != "nan"})
            teile.append(f"<span style='color:{ROT}'>⚠ <b>Montage steht still</b></span>: es fehlt "
                         + ", ".join(komp_name(m, k) for k in fehlt))
    return "<br>".join(teile)


def netzkarte(erg: Ergebnis, schritt_tage: int = 3, hoehe: int = 760, titel: str = "Animierte Netzwerksicht",
              mit_bestand: bool = True, tempo_ms: int = 240) -> go.Figure:
    """Animierte geografische Netzwerksicht mit Europa-Ausschnitt und Bestandsbalken.

    Links die Weltkarte, rechts oben ein Ausschnitt Europa (Häfen, Rhein, Bahn, Sellita, Heilbronn),
    rechts unten die Bestandsbalken je Komponente, alles synchron zum Zeitregler und aus demselben Lauf.
    Ein goldener Ring markiert den Lieferanten, der das Uhrwerk gerade liefert (Miyota oder im Notfall Sellita).
    tempo_ms: Dauer eines Bildes; 240 ms = halbe Geschwindigkeit gegenüber früher (120 ms).
    """
    m = erg.modell
    W, H = m.einschwingen, m.horizont
    genutzte = {k for r in m.routen.values() for route in r for k in route.kanten}
    linien = {kid: _polylinie(erg, kid) for kid in genutzte}
    verlauf = _quellen_verlauf(erg)
    if mit_bestand:
        fig = make_subplots(
            rows=2, cols=2, column_widths=[0.60, 0.40], row_heights=[0.55, 0.45],
            horizontal_spacing=0.09, vertical_spacing=0.10,
            specs=[[{"type": "geo", "rowspan": 2}, {"type": "geo"}], [None, {"type": "xy"}]],
            subplot_titles=("", "Ausschnitt Europa", "Bestand in Heilbronn (Reichweite in Tagen)"))
        geos = [("geo", dict(row=1, col=1)), ("geo2", dict(row=1, col=2))]
        bd = _bestandsdaten(erg)
    else:
        fig = go.Figure()
        geos = [("geo", {})]
        bd = None

    komps = list(m.komponenten)
    farbe = {k: KOMP_FARBE[i % len(KOMP_FARBE)] for i, k in enumerate(komps)}
    kn = [k for k in m.knoten.values() if k.typ != "kunde"]
    for g_nr, (geo, pos) in enumerate(geos):
        zeige_legende = g_nr == 0
        for modus in MODUS_FARBE:
            lats, lons = [], []
            for kid in genutzte:
                if m.kanten[kid].modus != modus:
                    continue
                lin = linien[kid]
                lats += list(lin[:, 0]) + [None]
                lons += list(lin[:, 1]) + [None]
            if lats:
                fig.add_trace(go.Scattergeo(lat=lats, lon=lons, mode="lines", name=MODUS_NAME[modus],
                                            line=dict(width=1.6, color=MODUS_FARBE[modus],
                                                      dash="dot" if modus == "luft" else "solid"),
                                            hoverinfo="skip", opacity=0.75, showlegend=zeige_legende,
                                            legendgroup=modus), **pos)
        fig.add_trace(go.Scattergeo(
            lat=[k.lat for k in kn], lon=[k.lon for k in kn], mode="markers", name="Standorte",
            marker=dict(size=[11 if k.typ == "werk" else 7 for k in kn],
                        color=[BRAUN if k.typ == "werk" else ("white" if k.typ != "lieferant" else BRAUN_DUNKEL)
                               for k in kn],
                        line=dict(width=1.2, color=BRAUN_DUNKEL)),
            text=[k.name for k in kn], hoverinfo="text", showlegend=zeige_legende, legendgroup="standorte"), **pos)
        if m.kunden:
            gsum = sum(k["gewicht"] for k in m.kunden)
            fig.add_trace(go.Scattergeo(
                lat=[k["lat"] for k in m.kunden], lon=[k["lon"] for k in m.kunden], mode="markers", name="Kundenorte",
                marker=dict(size=[4 + 14 * math.sqrt(k["gewicht"] / gsum) * 2 for k in m.kunden],
                            color="rgba(110,140,170,0.35)", line=dict(width=0.6, color="#6E8CAA")),
                text=[f"{k['name']}: {k['gewicht'] / gsum * 100:.1f} % der Bestellungen, {k['km']:.0f} km Luftlinie"
                      for k in m.kunden],
                hoverinfo="text", showlegend=zeige_legende, legendgroup="kunden"), **pos)
    if bd is not None:
        fig.add_trace(go.Scatter(y=bd["namen"], x=bd["melde"], mode="markers", name="Meldebestand",
                                 marker=dict(symbol="line-ns", size=22, line=dict(width=3, color=KUPFER)),
                                 hovertemplate="%{y}: Meldebestand s = %{x:.0f} Tage (erreicht der helle Balken "
                                               "den Strich, wird nachbestellt)<extra></extra>",
                                 showlegend=False), row=2, col=2)

    werk = m.knoten[m.werk]
    pakete = erg.auftraege.dropna(subset=["t_versand", "t_zustellung", "lat"]) if "lat" in erg.auftraege else None

    def paketpositionen(t: float) -> tuple[list, list, list]:
        if pakete is None or pakete.empty:
            return [], [], []
        ab, an = pakete.t_versand.to_numpy(), pakete.t_zustellung.to_numpy()
        sel = (ab < t) & (an > t - schritt_tage)
        d = pakete[sel]
        if d.empty:
            return [], [], []
        mitte = (np.maximum(d.t_versand, t - schritt_tage) + np.minimum(d.t_zustellung, t)) / 2
        f = ((mitte - d.t_versand) / (d.t_zustellung - d.t_versand).clip(lower=1e-9)).clip(0, 1)
        lat = werk.lat + f * (d.lat - werk.lat)
        lon = werk.lon + f * (d.lon - werk.lon)
        namen = {k["id"]: k["name"] for k in m.kunden}
        txt = [f"Paket an {namen.get(k, k)}" for k in d.kunde]
        return list(lat), list(lon), txt

    def geo_spuren(t: float, legende: bool) -> list[go.Scattergeo]:
        p = _sendungspositionen(erg, t, linien, schritt_tage)
        pk_lat, pk_lon, pk_txt = paketpositionen(t)
        g_lat, g_lon, g_txt = [], [], []
        for s in erg.stoerungsfenster:
            if s["start"] <= t < s["ende"]:
                if s["typ"] == "knoten_ausfall":
                    k = m.knoten[s["ziel"]]
                    g_lat.append(k.lat); g_lon.append(k.lon)
                elif s["typ"] == "kanten_sperre":
                    lin = linien.get(s["ziel"], _polylinie(erg, s["ziel"]))
                    mitte = lin[len(lin) // 2]
                    g_lat.append(mitte[0]); g_lon.append(mitte[1])
                else:
                    continue
                ziel = m.knoten[s["ziel"]].name if s["ziel"] in m.knoten else s["ziel"]
                g_txt.append(f"Störung: {ziel}")
        aktiv = [(k, _aktiver_lieferant(v, t)) for k, v in verlauf.items()]
        l_lat, l_lon = [], []
        for sd in _luftsendungen(erg, t, schritt_tage):
            for a in sd["abschnitte"]:
                lin = linien.get(a["kante"], _polylinie(erg, a["kante"]))
                l_lat += list(lin[:, 0]) + [None]
                l_lon += list(lin[:, 1]) + [None]
        return [
            go.Scattergeo(lat=pk_lat, lon=pk_lon, mode="markers", name="Pakete an Kunden",
                          marker=dict(size=5, color="#6E8CAA", symbol="square", line=dict(width=0.5, color="white")),
                          text=pk_txt, hoverinfo="text", showlegend=legende, legendgroup="pakete"),
            go.Scattergeo(lat=l_lat, lon=l_lon, mode="lines", name="Luftfracht unterwegs",
                          line=dict(width=3.5, color=ROT, dash="dash"), hoverinfo="skip",
                          showlegend=legende, legendgroup="luft_aktiv"),
            go.Scattergeo(lat=[x["lat"] for x in p], lon=[x["lon"] for x in p], mode="markers", name="Sendungen",
                          marker=dict(size=[8 if x["typ"] != "notfall" else 10 for x in p],
                                      color=[farbe[x["komponente"]] for x in p],
                                      symbol=["circle" if x["typ"] != "notfall" else "diamond" for x in p],
                                      line=dict(width=0.8, color="white")),
                          text=[f"{komp_name(m, x['komponente'])}: {x['menge']:.0f} Stück von "
                                f"{m.knoten[x['lieferant']].name.split(' (')[0]}"
                                + (" (Luftfracht)" if x["typ"] == "notfall" else "") for x in p],
                          hoverinfo="text", showlegend=legende, legendgroup="sendungen"),
            go.Scattergeo(lat=g_lat, lon=g_lon, mode="markers", name="Störung aktiv",
                          marker=dict(size=18, color="rgba(155,59,42,0.35)", line=dict(width=2, color=ROT)),
                          text=g_txt, hoverinfo="text", showlegend=legende, legendgroup="stoerung"),
            go.Scattergeo(lat=[m.knoten[l].lat for _, l in aktiv], lon=[m.knoten[l].lon for _, l in aktiv],
                          mode="markers+text", name="liefert gerade",
                          marker=dict(size=17, color="rgba(201,162,39,0.15)", line=dict(width=2.5, color="#C9A227")),
                          text=["Uhrwerk" if k == "uhrwerk" else komp_name(m, k) for k, _ in aktiv], textposition="top center",
                          textfont=dict(size=10, color=BRAUN_DUNKEL),
                          hovertext=[f"{komp_name(m, k)} kommt gerade von {m.knoten[l].name}" for k, l in aktiv],
                          hoverinfo="text", showlegend=legende, legendgroup="aktiv"),
        ]

    n_statisch = len(fig.data)
    dyn = []
    for g_nr, (geo, pos) in enumerate(geos):
        for spur in geo_spuren(W, g_nr == 0):
            dyn.append(len(fig.data))
            fig.add_trace(spur, **pos)
    if bd is not None:
        for spur in _bestand_spuren(bd, W, schritt_tage):
            dyn.append(len(fig.data))
            fig.add_trace(spur, row=2, col=2)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(range=[0, bd["xmax"]], title="Tage Reichweite", row=2, col=2, domain=[0.62, 0.885])
        fig.update_yaxes(autorange="reversed", row=2, col=2)
        for an in fig.layout.annotations:
            if an.text and an.text.startswith("Bestand in Heilbronn"):
                an.x = (0.62 + 0.885) / 2
        fig.add_annotation(
            xref="paper", yref="paper", x=1.0, y=0.40, xanchor="right", yanchor="top", showarrow=False, align="left",
            font=dict(size=10, color=BRAUN_DUNKEL), bgcolor="rgba(255,255,255,0.85)", bordercolor="#DCDCD8",
            borderwidth=1, borderpad=5,
            text=(f"<b>Legende</b><br><span style='color:{BRAUN}'>■</span> im Lager"
                  f"<br><span style='color:rgba(192,138,85,0.55)'>■</span> Lager + bestellt<br>   und unterwegs"
                  f"<br><span style='color:{KUPFER}'><b>|</b></span> Meldebestand:<br>   heller Balken erreicht<br>"
                  f"   ihn → Nachbestellung<br><span style='color:{ROT}'>■</span> unter 5 Tagen,<br>   Fehlteil droht"
                  f"<br><span style='color:{GRAU}'>↻ in 12 T</span> nächste<br>   Bestellung in etwa"
                  f"<br><span style='color:#5B7F3A'>✓ vor 3 T</span> zuletzt<br>   nachbestellt"))

    basis_anno = list(fig.layout.annotations)

    def status_anno(t: float) -> list:
        text = _status_text(erg, verlauf, t, schritt_tage)
        if not text:
            return basis_anno
        return basis_anno + [go.layout.Annotation(
            text=text, xref="paper", yref="paper", x=0.0, y=1.0, xanchor="left",
            yanchor="bottom", showarrow=False, align="left", font=dict(size=12, color=BRAUN_DUNKEL),
            bgcolor="rgba(246,246,244,0.92)", bordercolor="#C9A227", borderwidth=1, borderpad=4)]

    def rahmen(t: float) -> list:
        spuren = []
        for g_nr, _ in enumerate(geos):
            spuren += geo_spuren(t, g_nr == 0)
        return spuren + (_bestand_spuren(bd, t, schritt_tage) if bd is not None else [])

    zeiten = np.arange(W, H, schritt_tage)
    fig.frames = [go.Frame(data=rahmen(t), traces=dyn, name=f"{int(t - W)}",
                           layout=go.Layout(annotations=status_anno(t))) for t in zeiten]
    fig.update_layout(annotations=status_anno(W))
    schritte = [dict(method="animate", label=_datum(t - W),
                     args=[[f"{int(t - W)}"], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                    transition=dict(duration=0))]) for t in zeiten]
    geo_basis = dict(showland=True, landcolor="#E7E7E3", showocean=True, oceancolor="#EEF2F5", showcountries=True,
                     countrycolor="#CFCFCA", coastlinecolor="#C2C2BC", bgcolor=CREME)
    fig.update_layout(
        geo=geo_basis | dict(projection_type="natural earth", lataxis_range=[-50, 70], lonaxis_range=[-95, 150]),
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=-0.02, xanchor="left", yanchor="top",
                          direction="right",
                          buttons=[dict(label="▶ Abspielen", method="animate",
                                        args=[None, dict(frame=dict(duration=tempo_ms, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="▶▶ Doppelt", method="animate",
                                        args=[None, dict(frame=dict(duration=tempo_ms // 2, redraw=True),
                                                         fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(steps=schritte, x=0.30, len=0.70, y=-0.02, yanchor="top", currentvalue=dict(prefix="Datum: "),
                      pad=dict(t=0))],
    )
    if bd is not None:
        fig.update_layout(geo2=geo_basis | dict(projection_type="mercator", resolution=50,
                                                lataxis_range=[42.5, 55.5], lonaxis_range=[-1.0, 17.0]))
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.16, x=0))
    fig = _layout(fig, titel, hoehe=hoehe)
    fig.update_layout(margin=dict(l=0, r=10, t=70, b=150))
    return fig


# --------------------------------------------------------------------------- Diagramme
def komp_name(m, komponente: str) -> str:
    """Lesbarer Name einer Komponente (Bezeichnung aus components.csv)."""
    k = m.komponenten.get(komponente)
    return k.bezeichnung if k else komponente


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
        name = komp_name(m, k)
        fig.add_trace(go.Scatter(x=t.tag - W, y=t[f"bestand_{k}"] / pol.bedarf_tag, mode="lines", name=name,
                                 line=dict(width=1.8, color=KOMP_FARBE[i % len(KOMP_FARBE)]),
                                 hovertemplate=f"{name}: %{{y:.0f}} Tage<extra></extra>"))
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
    return _layout(fig, f"{komp_name(m, komponente)}: Meldebestand s = {pol.meldebestand:.0f}, "
                        f"Bestellmenge Q = {pol.losgroesse:.0f}")


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
        fig.add_trace(go.Box(y=y, name=f"{sid}", marker_color=BRAUN if i == 0 else KUPFER, boxmean=True,
                             hovertext=e["szenario_name"]))
    fig.update_yaxes(title=f"{name} [{einheit}]")
    fig.update_layout(showlegend=False)
    return _layout(fig, f"{name} je Szenario über alle Wiederholungen")


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
        if r.sellita_notfall: teile.append("Sellita")
        if getattr(r, "armband_italien", False): teile.append("Armband IT")
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


# Achsen für den Zielkonflikt: Spalte -> (Beschriftung, Faktor für die Anzeige, Einheit, Richtung)
ZIEL_ACHSEN = {
    "beta": ("Lieferfähigkeit", 100.0, "%", +1),
    "kosten_pro_uhr": ("Kosten je Uhr", 1.0, "€", -1),
    "co2e_pro_uhr_kg": ("CO2e je Uhr", 1000.0, "g", -1),
    "lieferzeit_p95": ("Lieferzeit (95 %)", 1.0, "Tage", -1),
    "fehlteil_tage": ("Montagestillstand", 1.0, "Tage", -1),
    "kapital_gebunden_eur": ("Gebundenes Kapital", 1.0, "€", -1),
}


def optionen_diagramm(rangfolge: pd.DataFrame, namen: dict[str, str], x: str = "kosten_pro_uhr", y: str = "beta",
                      anforderungen: dict[str, float] | None = None) -> go.Figure:
    """Zielkonflikt zweier frei wählbarer Zielkriterien.

    ★ = Empfehlung (Rang 1). Grün gefüllt = Pareto-optimal in diesen beiden Zielen (keine andere Option ist in beiden
    zugleich besser), grau = dominiert. Offener Kreis = verfehlt eine Mindestanforderung. Gestrichelte Linien zeigen
    Mindestanforderungen, falls eine Achse eine hat.
    """
    from .evaluation import pareto_front
    anforderungen = anforderungen or {}
    nx, fx, ex, rx = ZIEL_ACHSEN[x]
    ny, fy, ey, ry = ZIEL_ACHSEN[y]
    d = rangfolge.sort_values(x).reset_index(drop=True).copy()
    par = pareto_front(d, [x, y]) if x != y else pd.Series(True, index=d.index)
    lage = ["top center" if i % 2 == 0 else "bottom center" for i in range(len(d))]
    farbe = [KUPFER if r == 1 else ("#6B8F4A" if p else "#B0A69C") for r, p in zip(d.rang, par)]
    symbol = ["circle" if z else "circle-open" for z in d.zulaessig]
    fig = go.Figure(go.Scatter(
        x=d[x] * fx, y=d[y] * fy, mode="markers+text",
        text=[("★ " if r == 1 else "") + namen.get(o, o) for o, r in zip(d.option, d.rang)],
        textposition=lage, textfont=dict(size=11, color=BRAUN_DUNKEL),
        marker=dict(size=18, color=farbe, symbol=symbol,
                    line=dict(width=[3 if r == 1 else 1.5 for r in d.rang], color=[BRAUN_DUNKEL if z else f
                              for z, f in zip(d.zulaessig, farbe)])),
        hovertemplate="%{text}<br>" + f"{nx} %{{x:.2f}} {ex}<br>{ny} %{{y:.2f}} {ey}<extra></extra>", showlegend=False))
    for achse, kz, f, n, e in (("x", x, fx, nx, ex), ("y", y, fy, ny, ey)):
        if kz in anforderungen:
            wert = anforderungen[kz] * f
            text = f"Mindestanforderung {n} {wert:g} {e}".replace(".", ",")
            if achse == "y":
                fig.add_hline(y=wert, line=dict(color=ROT, dash="dash"), annotation_text=text,
                              annotation_position="bottom right")
            else:
                fig.add_vline(x=wert, line=dict(color=ROT, dash="dash"), annotation_text=text,
                              annotation_position="top left")
    fig.update_xaxes(title=f"{nx} [{ex}]  ({'rechts' if rx > 0 else 'links'} = besser)")
    fig.update_yaxes(title=f"{ny} [{ey}]  ({'oben' if ry > 0 else 'unten'} = besser)")
    ywerte = d[y] * fy
    spanne = float(ywerte.max() - ywerte.min()) or max(abs(float(ywerte.max())) * 0.05, 1.0)
    fig.update_yaxes(range=[float(ywerte.min()) - 0.25 * spanne, float(ywerte.max()) + 0.35 * spanne])
    xwerte = d[x] * fx
    xsp = float(xwerte.max() - xwerte.min()) or max(abs(float(xwerte.max())) * 0.05, 1.0)
    fig.update_xaxes(range=[float(xwerte.min()) - 0.18 * xsp, float(xwerte.max()) + 0.18 * xsp])
    return _layout(fig, f"{nx} gegen {ny}", hoehe=400)


KOSTENARTEN = {
    "einkauf": "Einkauf Komponenten", "montage": "Montage", "paket": "Paketversand", "transport": "Transport",
    "umschlag": "Umschlag", "zoll": "Zoll", "kapitalbindung": "Kapitalbindung", "lagerplatz": "Lagerplatz",
    "verspaetung": "Verspätungen", "letzte_meile": "Letzte Meile",
}


def kostenaufteilung(zeile: pd.Series) -> go.Figure:
    """Kosten je Uhr nach Kostenart für ein Szenario (Mittelwert über die Wiederholungen)."""
    n = max(float(zeile.get("zugestellt", 0) or 0), 1.0)
    werte = {name: float(zeile.get(f"kosten_{art}", 0) or 0) / n for art, name in KOSTENARTEN.items()}
    werte = {k: v for k, v in sorted(werte.items(), key=lambda x: x[1]) if v > 0.005}
    fig = go.Figure(go.Bar(x=list(werte.values()), y=list(werte.keys()), orientation="h", marker_color=KUPFER,
                           text=[f"{v:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".") for v in werte.values()],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.2f} €/Uhr<extra></extra>"))
    fig.update_xaxes(title="€ je Uhr")
    fig = _layout(fig, "Woraus sich die Kosten je Uhr zusammensetzen", hoehe=340)
    fig.update_layout(margin=dict(l=10, r=60, t=50, b=40))
    return fig


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
