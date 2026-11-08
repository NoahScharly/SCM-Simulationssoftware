"""Streamlit-Oberfläche: Regler, drei Sichten, Szenariovergleich und Optimierung (Modul ui).

Kennzeichnung: KI (Claude Opus 5.5), Prompts siehe KI_NACHWEIS.md.

Start:  streamlit run app/app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uhrensim import visualization as viz  # noqa: E402
from uhrensim.data import KonfigurationsFehler, lade_modell, szenario_dateien  # noqa: E402
from uhrensim.experiment import fuehre_aus, vergleiche  # noqa: E402
from uhrensim.optimization import rastersuche  # noqa: E402

st.set_page_config(page_title="Simulationsmodell Uhren-Supply-Chain", layout="wide")
st.markdown(
    """<style>
    .stApp {background-color: #F8F4EC;}
    h1, h2, h3 {font-family: Cambria, Georgia, serif; color: #3E2C23;}
    [data-testid="stSidebar"] {background-color: #F1E9DD;}
    </style>""",
    unsafe_allow_html=True,
)

st.title("Simulationsmodell")
st.caption("Ereignisdiskrete Simulation einer resilienten Lieferkette für eine Automatikuhr aus Heilbronn")

# --------------------------------------------------------------------------- Regler
basis = lade_modell()
grenzen = basis.param["grenzen"]

with st.sidebar:
    st.header("Parameter")
    menge = st.slider("Uhren pro Jahr", int(grenzen["uhren_pro_jahr"][0]), int(grenzen["uhren_pro_jahr"][1]),
                      int(basis.param["nachfrage"]["uhren_pro_jahr"]), step=250)
    kap = st.slider("Montagen pro Tag", int(grenzen["kapazitaet_pro_tag"][0]), int(grenzen["kapazitaet_pro_tag"][1]),
                    int(basis.param["montage"]["kapazitaet_pro_tag"]))
    noetig = math.ceil(menge / (grenzen["arbeitstage_pro_jahr"] * basis.param["montage"]["gutquote"])) if menge else 0
    st.caption(f"Nötig für diese Menge: mindestens {noetig} Montagen pro Tag")

    wbz = {}
    with st.expander("Wiederbeschaffungszeit je Komponente"):
        st.caption("Nicht kleiner als Fertigung + Transit + Zoll der Route")
        for komp in basis.komponenten:
            q = basis.primaerquelle(komp)
            minimum = math.ceil(q.wbz_min_tage)
            wert = st.number_input(f"{komp} ({q.lieferant}) [Tage]", min_value=minimum,
                                   max_value=int(grenzen["wbz_max_tage"]), value=minimum, step=1, key=f"wbz_{komp}")
            if wert > minimum:
                wbz[komp] = float(wert)

    st.header("Maßnahmen")
    puffer = st.select_slider("Zusätzlicher Uhrwerk-Puffer [Stück]", options=[0, 250, 500, 750, 1000, 1500, 2000])
    dual = st.checkbox("Dual Sourcing Uhrwerk (Notfallwerk)")
    luft = st.checkbox("Luftfracht bei drohendem Fehlteil")

    st.header("Experiment")
    reps = st.slider("Replikationen", 1, 50, 10)
    seed = st.number_input("Seed", value=int(basis.param["simulation"]["seed"]), step=1)
    methode = st.selectbox("Bewertung", ["mindestwerte", "gewichtet", "lexikografisch"],
                           format_func={"mindestwerte": "Mindestwerte, dann Kosten", "gewichtet": "Gewichtete Summe",
                                        "lexikografisch": "Rangfolge"}.get)

overrides = {
    "nachfrage": {"uhren_pro_jahr": menge},
    "montage": {"kapazitaet_pro_tag": kap},
    "simulation": {"seed": int(seed)},
    "massnahmen": {"puffer_extra": {"uhrwerk": puffer}, "dual_sourcing": dual, "luftfracht_notfall": luft},
    "bewertung": {"methode": methode},
    "wbz": wbz,
}

try:
    pruefung = lade_modell(None, overrides)
except KonfigurationsFehler as fehler:
    st.error(f"Unplausible Eingabe: {fehler}")
    st.stop()
for w in pruefung.warnungen:
    st.warning(w)

schluessel = repr(sorted((k, repr(v)) for k, v in overrides.items())) + f"|{reps}"

# --------------------------------------------------------------------------- Ausführen
dateien = szenario_dateien()
namen = {}
for p in dateien:
    m = lade_modell(p)
    namen[m.szenario_id] = (p, m.szenario_name)

spalte1, spalte2 = st.columns([3, 1])
with spalte1:
    auswahl = st.multiselect("Szenarien", list(namen), default=list(namen),
                             format_func=lambda s: f"{s}: {namen[s][1]}")
with spalte2:
    st.write("")
    starten = st.button("Simulation starten", type="primary", width="stretch")

if starten:
    if "S0" not in auswahl:
        auswahl = ["S0", *auswahl]
    balken = st.progress(0.0, text="Simuliere ...")
    fertig = []

    def melde(sid):
        fertig.append(sid)
        balken.progress(len(fertig) / len(auswahl), text=f"{sid} fertig")

    st.session_state["vergleich"] = vergleiche([namen[s][0] for s in auswahl], overrides, reps, melde)
    st.session_state["schluessel"] = schluessel
    balken.empty()

if "vergleich" not in st.session_state:
    st.info("Parameter links einstellen und **Simulation starten** drücken.")
    st.stop()
if st.session_state.get("schluessel") != schluessel:
    st.warning("Parameter wurden geändert. Die angezeigten Ergebnisse gehören noch zum letzten Lauf.")

vgl = st.session_state["vergleich"]
tabelle, ergebnisse = vgl["tabelle"], vgl["ergebnisse"]

tab_karte, tab_diagramm, tab_zahlen, tab_opt, tab_modell = st.tabs(
    ["Netzwerkkarte", "Diagramme", "Kennzahlen", "Optimierung", "Modell und Daten"])

# --------------------------------------------------------------------------- Sichten
with tab_karte:
    sid = st.selectbox("Szenario für die Karte", list(ergebnisse),
                       format_func=lambda s: f"{s}: {ergebnisse[s]['szenario_name']}", key="karte_sz")
    st.plotly_chart(viz.netzkarte(ergebnisse[sid]["beispiellauf"], schritt_tage=3), width="stretch")
    st.caption("Punkte = Sendungen (Farbe je Komponente, Raute = Luftfracht). Roter Kreis = aktive Störung. "
               "Gezeigt wird Replikation 0.")

with tab_diagramm:
    sid = st.selectbox("Szenario", list(ergebnisse), format_func=lambda s: f"{s}: {ergebnisse[s]['szenario_name']}",
                       key="diag_sz")
    erg = ergebnisse[sid]["beispiellauf"]
    st.plotly_chart(viz.serviceverlauf(erg), width="stretch")
    st.plotly_chart(viz.bestandsverlauf(erg), width="stretch")
    komp = st.selectbox("Bestand im Detail", list(erg.politik))
    st.plotly_chart(viz.bestand_einzeln(erg, komp), width="stretch")
    kz = st.selectbox("Streuung über Replikationen", ["beta", "kosten_pro_uhr", "co2e_pro_uhr_kg", "lieferzeit_p95"],
                      format_func=lambda k: viz.BESCHRIFTUNG[k][0])
    st.plotly_chart(viz.boxplot_vergleich(ergebnisse, kz), width="stretch")
    if len(tabelle) > 1:
        st.plotly_chart(viz.delta_balken(tabelle), width="stretch")

with tab_zahlen:
    st.subheader("Kennzahlen je Szenario (Mittelwert ± 95-%-Konfidenzintervall)")
    st.dataframe(viz.kennzahlentabelle(tabelle), width="stretch")
    mind = pruefung.param["bewertung"]["mindestwerte"].get("beta", 0.95)
    ampel = pd.DataFrame({
        "Szenario": tabelle["name"],
        "β erfüllt": ["✅" if b >= mind else "❌" for b in tabelle["beta"]],
        "Δ β [Pp]": (tabelle["delta_beta"] * 100).round(1),
        "Δ Kosten [€/Uhr]": tabelle["delta_kosten_pro_uhr"].round(2),
        "Δ CO2e [g/Uhr]": (tabelle["delta_co2e_pro_uhr_kg"] * 1000).round(1),
    })
    st.dataframe(ampel, width="stretch")
    res = pd.concat([e["resilienz"].assign(szenario=s) for s, e in ergebnisse.items() if not e["resilienz"].empty],
                    ignore_index=True) if any(not e["resilienz"].empty for e in ergebnisse.values()) else pd.DataFrame()
    if not res.empty:
        st.subheader("Resilienz je Störung (TTS ≥ TTR?)")
        agg = res.replace(math.inf, float("nan")).groupby(["szenario", "stoerung", "komponenten"]).agg(
            TTS_mittel=("tts", "mean"), TTR=("ttr", "mean"), resilient_anteil=("resilient", "mean")).round(1)
        st.dataframe(agg, width="stretch")
        st.caption("TTS leer = kein Montagestillstand bis Jahresende.")
    st.download_button("Kennzahlen als CSV", tabelle.to_csv().encode("utf-8"), "kennzahlen.csv", "text/csv")

with tab_opt:
    st.write("Rastersuche über Maßnahmen: Uhrwerk-Puffer, Dual Sourcing, Luftfracht. "
             "Bewertet wird der Mittelwert über die ausgewählten Szenarien.")
    opt_sz = st.multiselect("Szenarien für die Optimierung", list(namen), default=["S0", "S1", "S2"],
                            format_func=lambda s: f"{s}: {namen[s][1]}")
    opt_reps = st.slider("Replikationen je Kombination", 1, 20, 3)
    if st.button("Optimierung starten"):
        bal = st.progress(0.0)
        ov = {k: v for k, v in overrides.items() if k != "massnahmen"}
        st.session_state["raster"] = rastersuche([namen[s][0] for s in opt_sz], overrides=ov,
                                                 replikationen=opt_reps, bewertung=pruefung.param["bewertung"],
                                                 fortschritt=lambda i, n: bal.progress(i / n, text=f"{i}/{n}"))
        bal.empty()
    if "raster" in st.session_state:
        r = st.session_state["raster"]
        st.plotly_chart(viz.pareto_diagramm(r), width="stretch")
        zeige = r[["rang", "puffer_uhrwerk", "dual_sourcing", "luftfracht_notfall", "beta", "beta_min",
                   "kosten_pro_uhr", "co2e_pro_uhr_kg", "zulaessig", "pareto"]].copy()
        zeige["beta"] = (zeige.beta * 100).round(1)
        zeige["beta_min"] = (zeige.beta_min * 100).round(1)
        zeige["co2e_pro_uhr_kg"] = (zeige.co2e_pro_uhr_kg * 1000).round(1)
        st.dataframe(zeige.rename(columns={"beta": "β Mittel [%]", "beta_min": "β schlechtestes Szenario [%]",
                                           "co2e_pro_uhr_kg": "CO2e [g/Uhr]", "kosten_pro_uhr": "Kosten [€/Uhr]"}),
                     width="stretch", hide_index=True)

with tab_modell:
    st.subheader("Bestellpolitik (s, Q) je Komponente")
    erg0 = ergebnisse["S0"]["beispiellauf"] if "S0" in ergebnisse else next(iter(ergebnisse.values()))["beispiellauf"]
    st.dataframe(pd.DataFrame([vars(p) for p in erg0.politik.values()]).round(1), width="stretch",
                 hide_index=True)
    st.subheader("Bezugsquellen und Wiederbeschaffungszeit")
    st.dataframe(pd.DataFrame([{**{k: v for k, v in vars(q).items() if k != "tier2"}, "tier2": ", ".join(q.tier2)}
                               for l in pruefung.quellen.values() for q in l]).round(1),
                 width="stretch", hide_index=True)
    st.subheader("Ereignisprotokoll (Replikation 0, Basis)")
    st.dataframe(erg0.log.head(2000), width="stretch", hide_index=True)
    st.download_button("Ereignisprotokoll als CSV", erg0.log.to_csv(index=False).encode("utf-8"),
                       "ereignisprotokoll.csv", "text/csv")
