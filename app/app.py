"""Streamlit-Oberfläche als Entscheidungsunterstützung (Modul ui).

Ablauf: 1 Normalbetrieb und Störungen wählen, 2 Simulation rechnet je Szenario alle Handlungsoptionen,
3 Fall mit Karte und Kennzahlen, Schalter je Option, 4 Vergleich mit Vor- und Nachteilen und Empfehlung.

Kennzeichnung: G (Architektur und Fachlogik N. Scharly, Code teilweise selbst, teilweise mit Claude Opus 5.5 umgesetzt, Fehlerbehebung mit KI), siehe KI_NACHWEIS.md.

Start:  streamlit run app/app.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uhrensim import visualization as viz  # noqa: E402
from uhrensim.data import KonfigurationsFehler, lade_modell, montage_bedarf, szenario_dateien  # noqa: E402
from uhrensim.handlungsoptionen import (KRITERIEN, empfehlung_je_szenario, lade_entscheidungsregel,  # noqa: E402
                                        lade_optionen, robuste_option, roc_gewichte, vergleiche_optionen,
                                        vor_und_nachteile, wirkt)

try:  # Drag-and-drop-Liste für die Priorisierung; ohne das Paket greift eine einfache Auswahlliste
    from streamlit_sortables import sort_items
except ImportError:  # pragma: no cover
    sort_items = None
from uhrensim.optimization import kombinationen, optimum_je_szenario, raster_detail, robuste_empfehlung  # noqa: E402

st.set_page_config(page_title="Uhren-Supply-Chain Heilbronn", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """<style>
    .stApp {background-color: #F8F4EC;}
    [data-testid="stSidebar"] {background-color: #F1E9DD;}
    h1, h2, h3, h4 {font-family: Cambria, Georgia, serif; color: #3E2C23;}
    h1 {font-size: 2.0rem !important; margin-bottom: 0 !important;}
    .block-container {padding-top: 2.2rem; max-width: 1500px;}
    .untertitel {color: #6D594C; font-size: 0.98rem; margin-bottom: 0.6rem;}
    .kasten {background: #FFFFFF; border: 1px solid #E3D7C7; border-radius: 10px; padding: 12px 14px;
             margin: 4px 0 10px 0; font-size: 0.92rem; color: #3E2C23;}
    .kasten b {color: #3E2C23;}
    .hinweis {border-left: 4px solid #B0A69C; background: #FFFFFF; border-radius: 6px; padding: 9px 12px;
              margin-bottom: 7px; color: #3E2C23; font-size: 0.95rem;}
    .hinweis.gut {border-left-color: #5B7F3A;}
    .hinweis.warn {border-left-color: #C08A55;}
    .hinweis.schlecht {border-left-color: #9B3B2A;}
    .hinweis .titel {font-weight: 600;}
    [data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E3D7C7; border-radius: 10px;
                              padding: 10px 14px 6px 14px;}
    [data-testid="stMetricLabel"] p {font-size: 0.9rem; color: #6D594C;}
    .erkl {color: #6D594C; font-size: 0.82rem; line-height: 1.25; margin: 4px 2px 0 2px;}
    </style>""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- Hilfen
def de(x: float, nachkomma: int = 0) -> str:
    """Zahl im deutschen Format."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "k. A."
    return f"{x:,.{nachkomma}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def kurzname(knoten_name: str) -> str:
    return knoten_name.split(" (")[0]


@st.cache_data(show_spinner=False)
def szenarien() -> dict[str, dict]:
    ergebnis = {}
    for p in szenario_dateien():
        m = lade_modell(p)
        with open(p, encoding="utf-8") as f:
            roh = yaml.safe_load(f) or {}
        ergebnis[m.szenario_id] = {"pfad": str(p), "name": m.szenario_name,
                                   "kurz": str(roh.get("kurz", m.szenario_name)),
                                   "beschreibung": str(roh.get("beschreibung", "")),
                                   "optionen": list(roh.get("relevante_optionen") or [])}
    # Reihenfolge: Basis, Störungen S1, S2, ..., Chancen (C...) zuletzt
    return dict(sorted(ergebnis.items(), key=lambda kv: (kv[0] != "S0", kv[0].startswith("C"), kv[0])))


@st.cache_resource(show_spinner=False)
def _ergebnisspeicher() -> dict:
    """Gemeinsamer Speicher für gerechnete Läufe (gleiche Eingaben werden nicht neu simuliert)."""
    return {}


def rechne(overrides_json: str, ids: tuple[str, ...], reps: int, fortschritt=None) -> dict:
    """Alle gewählten Szenarien × alle Handlungsoptionen."""
    speicher = _ergebnisspeicher()
    schl = json.dumps([overrides_json, ids, reps])
    if schl not in speicher:
        sz = szenarien()
        speicher[schl] = vergleiche_optionen([sz[s]["pfad"] for s in ids], OPTIONEN, json.loads(overrides_json),
                                             reps, fortschritt)
        while len(speicher) > 4:
            speicher.pop(next(iter(speicher)))
    return speicher[schl]


def hinweis(art: str, titel: str, text: str) -> None:
    st.markdown(f'<div class="hinweis {art}"><span class="titel">{titel}</span> {text}</div>',
                unsafe_allow_html=True)


SZ = szenarien()
OPTIONEN = lade_optionen()
OPT = {o["id"]: o for o in OPTIONEN}
OPT_KURZ = {o["id"]: o["kurz"] for o in OPTIONEN}
basis = lade_modell()
g, mo = basis.param["grenzen"], basis.param["montage"]
PRO_MA = float(mo.get("uhren_pro_mitarbeiter_tag", 7))
SEED_STANDARD = int(basis.param["simulation"]["seed"])
KNAME = {k: v[0] for k, v in KRITERIEN.items()}
ANF_STANDARD, PRIO_STANDARD = lade_entscheidungsregel()
SORT_STIL = """
.sortable-component {background-color: transparent; padding: 0; border: none;}
.sortable-container {background-color: transparent; counter-reset: item; padding: 0; margin: 0;}
.sortable-container-header {display: none;}
.sortable-container-body {background-color: transparent; padding: 0;}
.sortable-item, .sortable-item:hover {background-color: #FFFFFF; color: #3E2C23; border: 1px solid #E3D7C7;
    border-left: 4px solid #8B5E3C; border-radius: 8px; font-size: 0.9rem; margin: 0 0 6px 0; padding: 7px 10px;
    cursor: grab; text-align: left; font-family: 'Source Sans Pro', 'Source Sans 3', sans-serif;}
.sortable-item::before {counter-increment: item; content: counter(item) ". "; font-weight: 600; color: #8B5E3C;}
"""
SEK_PER_LAUF = 0.25

# --------------------------------------------------------------------------- Seitenleiste
with st.sidebar:
    st.markdown("## Einstellungen")

    st.markdown("#### 1 · Normalbetrieb")
    menge = st.slider("Uhren pro Jahr", int(g["uhren_pro_jahr"][0]), int(g["uhren_pro_jahr"][1]),
                      int(basis.param["nachfrage"]["uhren_pro_jahr"]), step=250,
                      help="Geplante Jahresmenge. Montagekapazität und Mitarbeiterzahl werden daraus abgeleitet. "
                           "Lieferanten, Verkehrsmittel, Kosten und CO2e kommen aus der Datenbasis (config/).")
    mb = montage_bedarf(menge, g["arbeitstage_pro_jahr"], float(mo["gutquote"]), PRO_MA)
    auslastung = mb["montagen_noetig"] / mb["kapazitaet_pro_tag"] if mb["kapazitaet_pro_tag"] else 0
    st.markdown(
        f'<div class="kasten">Nötig: <b>{de(mb["montagen_noetig"], 1)} Montagen pro Tag</b><br>'
        f'Dafür: <b>{mb["mitarbeiter"]} Mitarbeiter</b> × {de(PRO_MA)} Uhren = '
        f'<b>{de(mb["kapazitaet_pro_tag"])} Montagen/Tag</b><br>'
        f'<span style="color:#6D594C">Auslastung {de(auslastung * 100)} % · {g["arbeitstage_pro_jahr"]} '
        f'Arbeitstage · {de(float(mo["gutquote"]) * 100)} % Gutquote</span></div>',
        unsafe_allow_html=True)

    st.markdown("#### 2 · Welche Störungen simulieren?")
    stoer_ids = [s for s in SZ if s != "S0"]
    auswahl = st.pills("Störungen", stoer_ids, default=stoer_ids, selection_mode="multi",
                       format_func=lambda s: SZ[s]["kurz"], label_visibility="collapsed",
                       help="Der Normalbetrieb (Basis) wird immer mitgerechnet, er ist der Vergleichsmaßstab.") or []
    st.caption(f"Basis wird immer mitgerechnet. Für jede Störung rechnet die Software automatisch alle "
               f"{len(OPTIONEN)} Handlungsoptionen durch: " + ", ".join(OPT_KURZ.values()) + ".")

    st.markdown("#### 3 · Zielkriterien")
    st.markdown("**Mindestanforderungen**")
    anforderungen: dict[str, float] = {}
    if st.toggle("Lieferfähigkeit mindestens", value="beta" in ANF_STANDARD, key="anf_beta_an"):
        anforderungen["beta"] = st.slider(
            "Lieferfähigkeit mindestens", 80, 100, int(round(ANF_STANDARD.get("beta", 0.95) * 100)), 1,
            format="%d %%", label_visibility="collapsed",
            help="Anteil der Bestellungen, die pünktlich ankommen. Optionen darunter sind nicht zulässig.") / 100
    if st.toggle("Lieferzeit (95 %) höchstens", value="lieferzeit_p95" in ANF_STANDARD, key="anf_lz_an"):
        anforderungen["lieferzeit_p95"] = st.slider(
            "Lieferzeit höchstens", 1.0, 10.0, float(ANF_STANDARD.get("lieferzeit_p95", 3.0)), 0.5,
            format="%.1f Tage", label_visibility="collapsed",
            help="95 % der Kunden haben ihre Uhr spätestens nach so vielen Werktagen. Standard 3 Tage = Lieferzusage.")

    st.markdown("**Priorisierung** (oben = am wichtigsten)")
    gewaehlt = st.pills("Kriterien", list(KRITERIEN), default=PRIO_STANDARD, selection_mode="multi",
                        format_func=KNAME.get, label_visibility="collapsed", key="prio_auswahl",
                        help="Welche Zielkriterien sollen in die Empfehlung eingehen?") or []
    alt = [k for k in st.session_state.get("prio_reihenfolge", PRIO_STANDARD) if k in gewaehlt]
    start = alt + [k for k in gewaehlt if k not in alt]
    if sort_items is not None and start:
        sortiert = sort_items([KNAME[k] for k in start], direction="vertical", custom_style=SORT_STIL,
                              key="prio_sort_" + "_".join(sorted(start)))
        zurueck = {v: k for k, v in KNAME.items()}
        prioritaet = [zurueck[x] for x in sortiert if x in zurueck]
    else:
        prioritaet = start
        if start:
            st.caption("Reihenfolge = Reihenfolge der Auswahl. Zum Umsortieren abwählen und neu anklicken.")
    st.session_state["prio_reihenfolge"] = prioritaet
    if prioritaet:
        st.caption("Gewichte aus der Reihenfolge (Rank-Order-Centroid): " + " · ".join(
            f"{KNAME[k]} {de(w * 100)} %" for k, w in zip(prioritaet, roc_gewichte(len(prioritaet)))))
    else:
        st.caption("Keine Priorisierung gewählt: Es zählen nur die Mindestanforderungen.")
    ziel_beta = anforderungen.get("beta")
    st.caption("Wirkt sofort, ohne neu zu rechnen. Ziehen zum Umsortieren.")

    with st.expander("Experteneinstellungen"):
        reps = st.slider("Wiederholungen je Szenario und Option", 2, 30, 5,
                         help="Nachfrage, Zollzeiten und Ausschuss sind zufällig. Mehr Wiederholungen machen den "
                              "Mittelwert genauer, die Rechnung dauert aber länger.")
        eigener_seed = st.toggle("Eigenen Seed festlegen", value=False,
                                 help="Standard ist ein fester Seed, damit jeder Start dieselben Zahlen liefert "
                                      "(nachprüfbar). Ein anderer Seed zeigt, wie stabil das Ergebnis ist.")
        seed = (st.number_input("Seed", value=SEED_STANDARD, step=1) if eigener_seed else SEED_STANDARD)

    n_laeufe = (1 + len(auswahl)) * len(OPTIONEN) * reps
    st.caption(f"{de(n_laeufe)} Simulationsläufe, etwa {de(max(n_laeufe * SEK_PER_LAUF / 60, 0.1), 1)} Minuten.")
    starten = st.button("Simulation starten", type="primary", width="stretch")

overrides = {
    "nachfrage": {"uhren_pro_jahr": menge},
    "montage": {"kapazitaet_pro_tag": "auto"},
    "simulation": {"seed": int(seed)},
}
ids = tuple(["S0", *[s for s in auswahl if s != "S0"]])
schluessel = json.dumps([overrides, ids, reps], sort_keys=True)

# --------------------------------------------------------------------------- Kopf
st.title("Uhren-Supply-Chain Heilbronn")
st.markdown(f'<div class="untertitel">Simulationsgestützte Entscheidungsunterstützung · {de(menge)} Uhren pro Jahr · '
            f'{mb["mitarbeiter"]} Mitarbeiter in der Montage · Basis + {len(ids) - 1} Störungen × '
            f'{len(OPTIONEN)} Handlungsoptionen</div>', unsafe_allow_html=True)

if menge == 0:
    st.info("Bei 0 Uhren pro Jahr gibt es nichts zu simulieren. Links eine Jahresmenge einstellen.")
    st.stop()

try:
    pruefung = lade_modell(None, overrides)
except KonfigurationsFehler as fehler:
    st.error(f"Diese Einstellung ist nicht möglich: {fehler}")
    st.stop()

with st.expander("Datenbasis des Normalbetriebs: Lieferanten und Verkehrsmittel"):
    p = pruefung.param
    modi = {}
    for k in pruefung.kanten.values():
        modi.setdefault(k.modus, []).append(k.co2e_g_tkm)
    st.markdown("**Verkehrsmittel** (Kosten, Dauer und CO2e je Tonnenkilometer; Werte mit [A] in der Konfiguration "
                "sind Annahmen)")
    st.dataframe(pd.DataFrame([{
        "Verkehrsmittel": viz.MODUS_NAME.get(mo_, mo_),
        "Geschwindigkeit [km/Tag]": p["transport"]["geschwindigkeit_km_tag"].get(mo_),
        "Kosten [€/tkm]": p["kosten"]["transport_eur_tkm"].get(mo_),
        "Fixkosten [€/Sendung]": p["kosten"]["transport_fix_eur_sendung"].get(mo_),
        "Umschlag [Tage, wahrscheinlichster Wert]": (p["transport"]["umschlag_tage"].get(mo_) or [None] * 3)[1],
        "CO2e [g/tkm, GLEC, Mittel der Kanten]": round(sum(v) / len(v), 1),
        "Kanten im Netz": len(v),
    } for mo_, v in modi.items()]), hide_index=True, width="stretch")
    st.markdown("**Lieferanten und Bezugsquellen**")
    rolle_txt = {"primaer": "regulär", "alternativ": "Notfallplan", "option": "Bezugsoption"}
    st.dataframe(pd.DataFrame([{
        "Komponente": viz.komp_name(pruefung, k), "Lieferant": kurzname(pruefung.knoten[q.lieferant].name),
        "Land": pruefung.knoten[q.lieferant].land, "Rolle": rolle_txt.get(q.rolle, q.rolle), "Preis [€]": q.preis_eur,
        "Fertigung [Tage]": q.fertigung_tage, "Tier-2": ", ".join(kurzname(pruefung.knoten[t].name) for t in q.tier2
                                                                  if t in pruefung.knoten),
    } for k, qs in pruefung.quellen.items() for q in qs]), hide_index=True, width="stretch")

if starten or "lauf" not in st.session_state:
    st.session_state["lauf"] = schluessel
lauf_ov, lauf_ids, lauf_reps = json.loads(st.session_state["lauf"])
bal = st.progress(0.0, text="Simulation läuft …")
vgl = rechne(json.dumps(lauf_ov, sort_keys=True), tuple(lauf_ids), int(lauf_reps),
             fortschritt=lambda i, n: bal.progress(i / n, text=f"Simulation {i} von {n} (Szenario × Option × "
                                                                 f"{lauf_reps} Wiederholungen gebündelt)"))
bal.empty()
if st.session_state["lauf"] != schluessel:
    st.info("Die Einstellungen wurden geändert. Die Ergebnisse unten gehören noch zum letzten Lauf. "
            "Links auf **Simulation starten** drücken, um neu zu rechnen.")

detail, erg_alle = vgl["detail"], vgl["ergebnisse"]
def relevant(s: str) -> list[str]:
    """Für das Szenario relevante Handlungsoptionen (Szenariodatei: relevante_optionen), "keine" immer zuerst."""
    liste = [o for o in (SZ[s]["optionen"] or list(OPT)) if o in OPT]
    return ["keine"] + [o for o in liste if o != "keine"]


# Empfehlung je Szenario nur unter den relevanten Optionen; die robuste Empfehlung nutzt alle Läufe
detail_rel = detail[[o in relevant(s) for s, o in zip(detail.szenario, detail.option)]]
rang_je_sz = empfehlung_je_szenario(detail_rel, anforderungen, prioritaet)


def anf_text() -> str:
    teile = []
    if "beta" in anforderungen:
        teile.append(f"Lieferfähigkeit ≥ {de(anforderungen['beta'] * 100)} %")
    if "lieferzeit_p95" in anforderungen:
        teile.append(f"Lieferzeit (95 %) ≤ {de(anforderungen['lieferzeit_p95'], 1)} Tage")
    return ", ".join(teile) or "keine Mindestanforderungen"


def prio_text() -> str:
    return " → ".join(f"{i + 1}. {KNAME[k]}" for i, k in enumerate(prioritaet)) or "keine Priorisierung"


def kurzwerte(r) -> str:
    return (f"{de(r.beta * 100, 1)} % Lieferfähigkeit, {de(r.kosten_pro_uhr, 2)} € und "
            f"{de(r.co2e_pro_uhr_kg * 1000, 1)} g CO2e je Uhr, Lieferzeit (95 %) {de(r.lieferzeit_p95, 1)} Tage")
zusage = int(pruefung.param["distribution"]["lieferzusage_tage"])
lauf_seed = int(lauf_ov["simulation"]["seed"])

# --------------------------------------------------------------------------- 1 · Fall
st.markdown("### 1 · Der Fall")
sz_ids = [s for s in SZ if s in set(detail.szenario)]
sid = st.segmented_control("Szenario", sz_ids, default=sz_ids[1] if len(sz_ids) > 1 else sz_ids[0],
                           selection_mode="single", format_func=lambda s: f"{s} · {SZ[s]['kurz']}",
                           key="szenario") or sz_ids[0]
rang = rang_je_sz[sid]
empf = rang.iloc[0]
st.markdown(f'<div class="untertitel"><b>{sid} · {SZ[sid]["name"]}:</b> {SZ[sid]["beschreibung"]}</div>',
            unsafe_allow_html=True)

oid = st.segmented_control(
    "Handlungsoption anzeigen (Karte und Kennzahlen wechseln mit)", relevant(sid), default="keine",
    selection_mode="single", key=f"option_{sid}",
    format_func=lambda o: ("★ " if o == empf.option else "") + OPT_KURZ[o]) or "keine"
zeilen_sz = detail[detail.szenario == sid].set_index("option")
z = zeilen_sz.loc[oid]
e = erg_alle[(sid, oid)]
erg = e["beispiellauf"]
m = erg.modell
mass = OPT[oid].get("massnahmen") or {}
sellita, luft = bool(mass.get("sellita_notfall")), bool(mass.get("luftfracht_notfall"))
puffer = int((mass.get("puffer_extra") or {}).get("uhrwerk", 0) or 0)
st.caption(f"**{OPT[oid]['name']}:** {OPT[oid].get('beschreibung', '')}  ·  ★ = Empfehlung für dieses Szenario")

st.plotly_chart(viz.netzkarte(erg, schritt_tage=3, hoehe=760, titel=""), width="stretch",
                config={"displayModeBar": False})
st.caption("▶ Abspielen zeigt ein simuliertes Jahr in halber Geschwindigkeit (▶▶ doppelt so schnell). Punkte sind "
           "Sendungen (Farbe je Komponente, Raute = Luftfracht), ein roter Kreis eine aktive Störung, der goldene "
           "Ring den Lieferanten, der das Uhrwerk gerade liefert. Der Kasten oben links zeigt, ob ein Notfallplan "
           "aktiv ist.")

pol_u = erg.politik.get("uhrwerk")
with st.expander("So liest du die Bestandsbalken", expanded=False):
    st.markdown(f"""
Jede Zeile ist eine Komponente im Lager Heilbronn. Gemessen wird in **Tagen Reichweite**: Wie viele Tage kann die
Montage mit dieser Menge weiterarbeiten? So sind 1.600 Uhrwerke und 160 Zifferblätter vergleichbar.

| Element | Bedeutung |
|---|---|
| **Dunkler Balken** | Was physisch im Lager liegt. Daneben steht die Stückzahl und die Reichweite (T = Tage). |
| **Heller Balken** | Lager **plus** alles, was schon bestellt, in Fertigung oder unterwegs ist (Lagerposition). |
| **Kupferfarbener Strich** | Meldebestand s, gilt für den **hellen** Balken. Sinkt der **helle** Balken auf den Strich, wird automatisch die Bestellmenge Q nachbestellt („⟵ bestellt“), der helle Balken springt nach rechts. |
| Farbe | braun = normal · rot = weniger als 5 Tage im Lager, Fehlteil droht |
| **Rechte Spalte** | „↻ in 12 T“: in etwa 12 Tagen erreicht der helle Balken den Strich, dann wird nachbestellt. „✓ vor 3 T“: vor 3 Tagen wurde nachbestellt, der helle Balken ist dabei nach rechts gesprungen. |

**Beispiel Uhrwerk in diesem Lauf:** Bedarf {de(pol_u.bedarf_tag, 1) if pol_u else "-"} Stück pro Tag,
Wiederbeschaffungszeit {de(pol_u.wbz_tage) if pol_u else "-"} Tage. Meldebestand s = Bedarf in der
Wiederbeschaffungszeit + Sicherheitsbestand = {de(pol_u.meldebestand) if pol_u else "-"} Stück. Bestellmenge
Q = {de(pol_u.losgroesse) if pol_u else "-"} Stück. Der größte Teil steckt in Fertigung und auf dem Schiff, deshalb
liegt der dunkle Balken meist weit unter dem Strich.
""")

# --------------------------------------------------------------------------- Kennzahlen
st.markdown(f"#### Kennzahlen: {sid} mit „{OPT[oid]['name']}“")
ref_txt = "ggü. ohne Maßnahme" if oid != "keine" else "ggü. Basis"


def delta(spalte: str, faktor: float = 1.0, einheit: str = "", nachkomma: int = 1) -> str | None:
    """Gepaarte Differenz: Option gegen 'Ohne Maßnahme' im selben Szenario, sonst Szenario gegen Basis."""
    if oid != "keine":
        d = z.get(f"delta_{spalte}", math.nan)
    elif sid != "S0" and ("S0", "keine") in erg_alle:
        diff = (e["laeufe"][spalte] - erg_alle[("S0", "keine")]["laeufe"][spalte]).dropna()
        d = float(diff.mean()) if not diff.empty else math.nan
    else:
        return None
    if d is None or pd.isna(d):
        return None
    d *= faktor
    return f"{'+' if d >= 0 else ''}{de(d, nachkomma)} {einheit} {ref_txt}"


karten = [
    ("Lieferfähigkeit", f"{de(z['beta'] * 100, 1)} %", delta("beta", 100, "Pp"), "normal",
     f"Anteil der Bestellungen, die in höchstens {zusage} Werktagen ankommen."
     + (f" Anforderung: mindestens {de(ziel_beta * 100)} %." if ziel_beta else "")),
    ("Kosten je Uhr", f"{de(z['kosten_pro_uhr'], 2)} €", delta("kosten_pro_uhr", 1, "€", 2), "inverse",
     "Einkauf, Montage, Transport, Lager, Kapitalbindung und Verspätungen, geteilt durch die gelieferten Uhren."),
    ("CO2e je Uhr", f"{de(z['co2e_pro_uhr_kg'] * 1000, 1)} g", delta("co2e_pro_uhr_kg", 1000, "g"), "inverse",
     "Emissionen von Transport, Umschlag und Zustellung je Uhr nach GLEC."),
    ("Lieferzeit (95 %)", f"{de(z['lieferzeit_p95'], 1)} Tage", delta("lieferzeit_p95", 1, "Tage"), "inverse",
     "95 % der Kunden haben ihre Uhr spätestens nach so vielen Werktagen."),
    ("Montagestillstand", f"{de(z['fehlteil_tage'], 1)} Tage", delta("fehlteil_tage", 1, "Tage"), "inverse",
     "Arbeitstage, an denen mindestens eine Komponente fehlte und nicht montiert werden konnte."),
]
for spalte, (titel, wert, d, farbe, erkl) in zip(st.columns(len(karten)), karten):
    with spalte:
        st.metric(titel, wert, d, delta_color=farbe)
        st.markdown(f'<div class="erkl">{erkl}</div>', unsafe_allow_html=True)

with st.expander("Wie werden diese Zahlen berechnet?"):
    st.markdown(f"""
Alle Werte stammen aus dem **Ereignisprotokoll** der Simulation. Jede Bestellung, Fertigung, Abfahrt, Ankunft,
Montage und Zustellung ist dort eine Zeile. Ausgewertet wird ein Jahr ({de(m.param['simulation']['auswertung_tage'])}
Tage), davor läuft eine Einschwingphase von {de(m.einschwingen)} Tagen, damit die Lager realistisch gefüllt sind.
Jede Kombination aus Szenario und Option wurde {lauf_reps}-mal mit anderen Zufallszahlen simuliert, angezeigt ist
der Mittelwert. Seed {lauf_seed}: Alle Szenarien und Optionen nutzen dieselben Zufallszahlen (Common Random Numbers),
deshalb zeigen die Differenzen nur die Wirkung der Störung oder der Option, und jeder Start ist reproduzierbar.

**Lieferfähigkeit β** = pünktliche Bestellungen / alle Bestellungen. Pünktlich heißt: zugestellt nach höchstens
{zusage} Werktagen. Wert hier: {de(z['beta'] * 100, 1)} % ± {de(z['beta_ki95'] * 100, 1)} Pp (95-%-Konfidenzintervall).

**Kosten je Uhr** = (Einkauf + Montage {de(m.param['kosten']['montage_eur_uhr'])} €/Uhr + Paket + Transport je
Tonnenkilometer und Sendung + Umschlag + Zoll + Kapitalbindung {de(m.param['kosten']['kapitalzins_pa'] * 100)} % p. a.
auf den Lagerwert + Lagerplatz + {de(m.param['kosten']['verspaetung_eur_auftrag'])} € je verspäteter Bestellung)
/ gelieferte Uhren.

**CO2e je Uhr** = Σ (Sendungsgewicht × Distanz × GLEC-Faktor des Verkehrsträgers) + Umschlag + Zustellung,
geteilt durch die gelieferten Uhren.

**Lieferzeit (95 %)** = 95-%-Quantil der Werktage zwischen Bestellung und Zustellung.

**Montagestillstand** = Anzahl Arbeitstage mit mindestens einem Fehlteil.
""")

# --------------------------------------------------------------------------- Einschätzung
st.markdown("#### Was bedeutet das?")


def stoerung_name(text: str) -> str:
    typ, _, ziel = text.partition(":")
    if typ == "knoten_ausfall" and ziel in m.knoten:
        return f"Ausfall {kurzname(m.knoten[ziel].name)}"
    if typ == "kanten_sperre" and ziel in m.kanten:
        k = m.kanten[ziel]
        return (f"Sperrung {viz.MODUS_NAME.get(k.modus, k.modus)} {kurzname(m.knoten[k.von].name)} → "
                f"{kurzname(m.knoten[k.nach].name)}")
    return text


links, rechts = st.columns([3, 2], gap="large")
with links:
    jahr, spaet = float(z["auftraege"]), float(z["verspaetet"])
    if z["beta"] >= (ziel_beta or 0.95):
        hinweis("gut", "Lieferversprechen gehalten.",
                f"{de(z['beta'] * 100, 1)} % der Bestellungen kommen pünktlich an"
                + (f" (Anforderung {de(ziel_beta * 100)} %)." if ziel_beta else "."))
    else:
        hinweis("schlecht", "Lieferversprechen verfehlt.",
                f"Nur {de(z['beta'] * 100, 1)} % pünktlich. Rund {de(spaet)} von {de(jahr)} Bestellungen im Jahr "
                f"kommen zu spät.")

    fehl = erg.log[erg.log.ereignis == "fehlteil"]
    komps = sorted({k for x in fehl.komponente.astype(str) for k in x.split(";") if k and k != "nan"})
    if z["fehlteil_tage"] >= 0.5:
        namen = ", ".join(viz.komp_name(m, k) for k in komps) or "Komponenten"
        hinweis("schlecht", "Ursache: Fehlteile.",
                f"An durchschnittlich {de(z['fehlteil_tage'], 1)} Arbeitstagen fehlte {namen}, die Montage stand "
                f"still.")
    stau = erg.log[erg.log.ereignis.isin(["warten_sperre", "warten_knoten"])]
    if not stau.empty:
        verz = []
        for sid_, g_ in stau.groupby("objekt_id"):
            nach_ = erg.log[(erg.log.objekt_id == sid_) & (erg.log.zeit > g_.zeit.max())
                            & ~erg.log.ereignis.isin(["warten_sperre", "warten_knoten"])]
            ende_ = nach_.zeit.min() if not nach_.empty else m.horizont
            verz.append((g_.komponente.iloc[0], ende_ - g_.zeit.min(), g_.ort.iloc[0]))
        teile_ = ", ".join(sorted({viz.komp_name(m, k) for k, _, _ in verz}))
        orte_ = ", ".join(sorted({kurzname(m.knoten[o].name) for _, _, o in verz if o in m.knoten}))
        hinweis("warn" if z["fehlteil_tage"] < 0.5 else "schlecht", "Ware hängt fest:",
                f"{len(verz)} Sendung(en) ({teile_}) warteten vor der Störung in {orte_}, im Mittel "
                f"{de(sum(d for _, d, _ in verz) / len(verz))} Tage, längstens {de(max(d for _, d, _ in verz))} Tage."
                + (" Das Lager hat die Lücke überbrückt." if z["fehlteil_tage"] < 0.5 else ""))
    if z["fehlteil_tage"] < 0.5 and spaet >= 1:
        kap = m.param["montage"]["kapazitaet_pro_tag"]
        hinweis("warn", "Ursache: Montagekapazität.",
                f"Alle Teile waren da, aber {de(kap)} Montagen/Tag ({m.param['montage']['mitarbeiter']} "
                f"Mitarbeiter) reichen in Nachfragespitzen nicht.")

    res = e["resilienz"]
    if not res.empty:
        for name_roh, d in res.groupby("stoerung"):
            name = stoerung_name(name_roh)
            tts = d.tts.replace(math.inf, float("nan"))
            ttr = float(d.ttr.mean())
            if tts.isna().mean() > 0.5:
                hinweis("gut", f"{name}: überbrückt.",
                        f"Das Lager reicht bis die Versorgung wieder läuft ({de(ttr)} Tage), die Montage läuft durch.")
            else:
                tts_m = float(tts.mean())
                luecke = max(ttr - tts_m, 0.0)
                hinweis("schlecht" if oid == "keine" else "warn", f"{name}: Lager allein reicht nicht.",
                        f"Der Bestand hält {de(tts_m)} Tage (TTS), bis wieder Ware ankommt, vergehen {de(ttr)} Tage "
                        f"(TTR). Lücke rund {de(luecke)} Tage.")

    if oid != "keine":
        vor, nach = vor_und_nachteile(z)
        if not wirkt(z):
            hinweis("", f"{OPT[oid]['kurz']} wird in diesem Szenario nicht ausgelöst.",
                    "Keine Wirkung, aber auch keine Kosten. Die Option ist eine Versicherung für andere Fälle.")
        else:
            if vor:
                hinweis("gut", "Vorteile ggü. ohne Maßnahme:", " · ".join(vor))
            if nach:
                hinweis("warn", "Nachteile ggü. ohne Maßnahme:", " · ".join(nach))
    if sellita:
        wechsel = erg.log[erg.log.ereignis == "lieferantenwechsel"]
        zu = wechsel[wechsel.zustand_neu != "miyota_nagano"]
        if not zu.empty:
            zurueck = wechsel[wechsel.zustand_neu == "miyota_nagano"]
            txt = f"ab {viz._datum(zu.zeit.min() - m.einschwingen)} wird bei Sellita bestellt"
            if not zurueck.empty:
                txt += f", ab {viz._datum(zurueck.zeit.min() - m.einschwingen)} wieder bei Miyota"
            hinweis("warn", "Notfallplan Sellita ausgelöst:", txt + ".")

with rechts:
    st.markdown(f"**Empfehlung für {sid}**")
    if empf.zulaessig:
        n_zul = int(rang.zulaessig.sum())
        hinweis("gut", f"★ {OPT[empf.option]['name']}",
                f"erfüllt die Mindestanforderungen ({anf_text()})"
                + (f" und liegt unter {n_zul} zulässigen Optionen vorn nach eurer Priorisierung ({prio_text()})"
                   if prioritaet and n_zul > 1 else "") + f": {kurzwerte(empf)}.")
    else:
        hinweis("schlecht", f"★ {OPT[empf.option]['name']}",
                f"Keine Option erfüllt alle Mindestanforderungen ({anf_text()}). Am nächsten kommt diese, sie "
                f"verfehlt: {empf.verletzt}. Werte: {kurzwerte(empf)}. Kombinationen prüft Abschnitt 4.")
    probleme = list(dict.fromkeys(pruefung.warnungen + list(e.get("warnungen", []))))
    if auslastung > 0.95:
        probleme.append(f"Die Montage ist zu {de(auslastung * 100)} % ausgelastet. Schwankungen und Saisonspitzen "
                        f"führen dann schnell zu Verspätungen.")
    for p_ in probleme:
        hinweis("warn", "", p_)

# --------------------------------------------------------------------------- 2 · Optionen im Szenario
st.markdown(f"### 2 · Handlungsoptionen für {sid} im Vergleich")
st.markdown("Alle Optionen wurden automatisch mit denselben Zufallszahlen simuliert, Vergleichsmaßstab ist "
            "„Ohne Maßnahme“ im selben Szenario. Rangfolge: **Stufe 1** Mindestanforderungen "
            f"({anf_text()}), **Stufe 2** Priorisierung ({prio_text()}). Punkte = gewichteter Abstand zur jeweils "
            "besten zulässigen Option, 0 = überall die beste.")
tab_opt = []
for r in rang.itertuples():
    vor, nach = vor_und_nachteile(rang.loc[r.Index])
    nicht = r.option != "keine" and not wirkt(rang.loc[r.Index])
    tab_opt.append({
        "Rang": r.rang, "Option": ("★ " if r.rang == 1 else "") + OPT[r.option]["name"],
        "Anforderungen": "✅" if r.zulaessig else f"❌ {r.verletzt}",
        "Punkte": round(r.punkte, 2) if r.zulaessig and prioritaet else None,
        "Lieferfähigkeit [%]": round(r.beta * 100, 1), "Kosten [€/Uhr]": round(r.kosten_pro_uhr, 2),
        "CO2e [g/Uhr]": round(r.co2e_pro_uhr_kg * 1000, 1), "Lieferzeit 95 % [Tage]": round(r.lieferzeit_p95, 1),
        "Stillstand [Tage]": round(r.fehlteil_tage, 1),
        "Vorteile": "nicht ausgelöst" if nicht else (" · ".join(vor) or "-" if r.option != "keine" else "Referenz"),
        "Nachteile": "" if nicht else " · ".join(nach),
    })
st.dataframe(pd.DataFrame(tab_opt), hide_index=True, width="stretch", column_config={
    "Lieferfähigkeit [%]": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
    "Vorteile": st.column_config.TextColumn(width="large"), "Nachteile": st.column_config.TextColumn(width="large")})
st.plotly_chart(viz.optionen_diagramm(rang, OPT_KURZ, ziel_beta), width="stretch")

# --------------------------------------------------------------------------- 3 · Gesamtbild
st.markdown("### 3 · Gesamtbild: alle Szenarien × alle Optionen")
st.markdown("Jede Zelle: Mindestanforderungen erfüllt (✅/❌), Lieferfähigkeit und Kosten je Uhr. Zeigt, welche Option "
            "in welchem Fall greift und was sie dort kostet.")
matrix = []
for s in sz_ids:
    zs = detail[detail.szenario == s].set_index("option")
    zeile = {"Szenario": f"{s} · {SZ[s]['kurz']}"}
    for o in OPT:
        if o not in relevant(s):
            zeile[OPT_KURZ[o]] = "·"
            continue
        r = zs.loc[o]
        zeichen = "✅" if rang_je_sz[s].set_index("option").loc[o, "zulaessig"] else "❌"
        zusatz = " (nicht ausgelöst)" if o in ("sellita", "luftfracht") and not wirkt(r) else ""
        zeile[OPT_KURZ[o]] = f"{zeichen} {de(r.beta * 100, 1)} % · {de(r.kosten_pro_uhr, 2)} €{zusatz}"
    zeile["Empfehlung"] = "★ " + OPT_KURZ[rang_je_sz[s].iloc[0].option]
    matrix.append(zeile)
st.dataframe(pd.DataFrame(matrix), hide_index=True, width="stretch")

robust = robuste_option(detail, anforderungen, prioritaet)
rb = robust.iloc[0]
if rb.zulaessig:
    st.success(f"**Robuste Empfehlung, wenn man vorher nicht weiß, welche Störung kommt: {OPT[rb.option]['name']}.** "
               f"Erfüllt {anf_text()} auch im schlechtesten Szenario (Lieferfähigkeit mind. "
               f"{de(rb.beta_min * 100, 1)} %, Lieferzeit höchstens {de(rb.lieferzeit_p95_schlecht, 1)} Tage). Im "
               f"Mittel {de(rb.kosten_pro_uhr, 2)} € und {de(rb.co2e_pro_uhr_kg * 1000, 1)} g CO2e je Uhr, bindet im "
               f"Mittel {de(rb.kapital_gebunden_eur)} € Kapital. Priorisierung: {prio_text()}.")
else:
    st.warning(f"Keine einzelne Option erfüllt {anf_text()} in allen gewählten Szenarien. Am nächsten kommt "
               f"**{OPT[rb.option]['name']}**, sie verfehlt im schlechtesten Szenario: {rb.verletzt}. "
               f"Kombinationen prüft Abschnitt 4.")
with st.expander("Robuste Rangfolge und Zielkonflikt über alle Szenarien"):
    st.dataframe(pd.DataFrame({
        "Rang": robust.rang.values, "Option": [OPT[o]["name"] for o in robust.option],
        "Lieferfähigkeit schlechtestes Szenario [%]": (robust.beta_min * 100).round(1).values,
        "Kosten Ø [€/Uhr]": robust.kosten_pro_uhr.round(2).values,
        "Kosten schlechtestes Szenario [€/Uhr]": robust.kosten_max.round(2).values,
        "CO2e Ø [g/Uhr]": (robust.co2e_pro_uhr_kg * 1000).round(1).values,
        "Gebundenes Kapital Ø [€]": robust.kapital_gebunden_eur.round(-2).astype(int).values,
        "Lieferzeit schlechtestes Szenario [Tage]": robust.lieferzeit_p95_schlecht.round(1).values,
        "Anforderungen": ["✅" if z_ else f"❌ {v}" for z_, v in zip(robust.zulaessig, robust.verletzt)],
        "Punkte": [round(p_, 2) if z_ and prioritaet else None for p_, z_ in zip(robust.punkte, robust.zulaessig)],
        "Pareto-optimal": robust.pareto.map({True: "ja", False: ""}).values,
    }), hide_index=True, width="stretch")
    st.plotly_chart(viz.optionen_diagramm(robust, OPT_KURZ, ziel_beta, beta_spalte="beta_min"), width="stretch")

# --------------------------------------------------------------------------- Details
st.markdown("### Details")
with st.expander(f"Verlauf über das Jahr ({sid}, {OPT_KURZ[oid]})"):
    st.plotly_chart(viz.serviceverlauf(erg), width="stretch")
    st.plotly_chart(viz.bestandsverlauf(erg), width="stretch")
    komp = st.selectbox("Bestand einer Komponente im Detail", list(erg.politik),
                        format_func=lambda k: viz.komp_name(m, k))
    st.plotly_chart(viz.bestand_einzeln(erg, komp), width="stretch")
    st.caption("Rote Flächen markieren Störungen. Gezeigt ist eine der Wiederholungen.")

with st.expander(f"Kosten im Detail ({sid}, {OPT_KURZ[oid]})"):
    st.plotly_chart(viz.kostenaufteilung(z), width="stretch")

with st.expander("Wie sicher sind die Ergebnisse?"):
    st.markdown(f"Jeder Punkt ist eine Wiederholung mit anderen Zufallszahlen (Option „{OPT_KURZ[oid]}“). Liegen die "
                "Kästen verschiedener Szenarien weit auseinander, ist der Unterschied echt und kein Zufall.")
    kz = st.selectbox("Kennzahl", ["beta", "kosten_pro_uhr", "co2e_pro_uhr_kg", "lieferzeit_p95"],
                      format_func=lambda k: viz.BESCHRIFTUNG[k][0])
    st.plotly_chart(viz.boxplot_vergleich({s: erg_alle[(s, oid)] for s in sz_ids}, kz), width="stretch")
    st.download_button("Alle Ergebnisse (Szenario × Option) als CSV", detail.to_csv(index=False).encode("utf-8"),
                       "szenarien_optionen.csv", "text/csv")

# --------------------------------------------------------------------------- Optimierung
st.markdown("### 4 · Für Fortgeschrittene: Maßnahmen kombinieren")
st.markdown("Abschnitt 2 und 3 vergleichen einzelne Optionen. Hier werden **Kombinationen** getestet: alle "
            f"{len(kombinationen())} Kombinationen aus Uhrwerk-Puffer (0, 500, 1.000 Stück), Notfallplan Sellita, "
            "Armband Italien und Notfallplan Luftfracht. Ergebnis: das Optimum je Szenario und eine "
            "**robuste Empfehlung**, die in allen gewählten Szenarien die Mindestanforderung erfüllt und nach deinen "
            "Zielkriterien am besten abschneidet. Da man vorher nicht weiß, welche Störung eintritt, ist die "
            "robuste Empfehlung die eigentliche Entscheidung.")

KRIT = {"beta": "Lieferfähigkeit", "kosten_pro_uhr": "Kosten je Uhr", "co2e_pro_uhr_kg": "CO2e je Uhr",
        "lieferzeit_p95": "Lieferzeit (95 %)", "fehlteil_tage": "Montagestillstand"}
o1, o2 = st.columns([3, 2])
with o1:
    st.markdown("**1 · Zielkriterien und Priorität**")
    ziele = st.multiselect(
        "Kriterien in der Reihenfolge ihrer Priorität auswählen (zuerst gewählt = wichtigstes)",
        list(KRIT), default=["beta", "kosten_pro_uhr", "co2e_pro_uhr_kg"], format_func=KRIT.get, key="opt_ziele")
    verfahren = st.radio(
        "Wie sollen die Kriterien verrechnet werden?",
        ["lexikografisch", "gewichtet"], horizontal=True, key="opt_verfahren",
        format_func={"lexikografisch": "Streng nach Priorität",
                     "gewichtet": "Gewichtet"}.get,
        help="Streng nach Priorität: Das erste Kriterium entscheidet; nur bei Gleichstand zählt das nächste. "
             "Gewichtet: Jedes Kriterium wird auf 0 bis 1 normiert und mit seinem Gewicht addiert, ein guter Wert "
             "bei einem Kriterium kann einen schlechten bei einem anderen ausgleichen.")
    gewichte = {}
    if verfahren == "gewichtet" and ziele:
        n = len(ziele)
        start = [round(100 * (n - i) / (n * (n + 1) / 2)) for i in range(n)]
        gcols = st.columns(n)
        for c, kz, w0 in zip(gcols, ziele, start):
            gewichte[kz] = c.slider(f"Gewicht {KRIT[kz]}", 0, 100, w0, 5, format="%d %%", key=f"opt_w_{kz}")
        summe = sum(gewichte.values()) or 1
        gewichte = {k: v / summe for k, v in gewichte.items()}
        st.caption("Gewichte werden auf 100 % normiert: " +
                   " · ".join(f"{KRIT[k]} {de(v * 100)} %" for k, v in gewichte.items()))
    elif ziele:
        st.caption("Priorität: " + " → ".join(f"{i + 1}. {KRIT[k]}" for i, k in enumerate(ziele)))
with o2:
    st.markdown("**2 · Mindestanforderung**")
    mit_min = st.checkbox("Lieferfähigkeit muss erreicht werden", value=True, key="opt_min_an")
    min_beta = st.slider("Mindest-Lieferfähigkeit (im schlechtesten Szenario)", 80, 100,
                         int(round(pruefung.param["bewertung"]["mindestwerte"].get("beta", 0.95) * 100)), 1,
                         format="%d %%", disabled=not mit_min, key="opt_min")
    st.markdown("**3 · Umfang**")
    opt_sz = st.multiselect("Szenarien", list(SZ), default=list(SZ),
                            format_func=lambda s: f"{s} · {SZ[s]['name']}", key="opt_sz")
    opt_reps = st.slider("Wiederholungen je Kombination", 1, 20, 5, key="opt_reps")
    st.caption(f"{len(opt_sz) * len(kombinationen()) * opt_reps} Simulationsläufe, etwa "
               f"{de(len(opt_sz) * len(kombinationen()) * opt_reps * 0.25 / 60, 1)} Minuten.")

bewertung = {"methode": verfahren, "rangfolge": ziele or ["kosten_pro_uhr"],
             "gewichte": gewichte or {"kosten_pro_uhr": 1.0},
             "mindestwerte": {"beta": min_beta / 100} if mit_min else {}}
if st.button("Optimierung starten", type="primary") and opt_sz:
    bal = st.progress(0.0)
    ov = {k: v for k, v in overrides.items() if k != "massnahmen"}
    st.session_state["opt_detail"] = raster_detail(
        [SZ[s]["pfad"] for s in opt_sz], overrides=ov, replikationen=opt_reps,
        fortschritt=lambda i, n: bal.progress(i / n, text=f"Lauf {i} von {n}"))
    bal.empty()


def massnahme_txt(r) -> str:
    teile = [f"Puffer {de(r.puffer_uhrwerk)} Stück" if r.puffer_uhrwerk else "kein Puffer"]
    if r.sellita_notfall:
        teile.append("Notfallplan Sellita")
    if getattr(r, "armband_italien", False):
        teile.append("Armband Italien")
    if r.luftfracht_notfall:
        teile.append("Notfallplan Luftfracht")
    return ", ".join(teile)


if "opt_detail" in st.session_state:
    detail = st.session_state["opt_detail"]
    st.caption("Die Simulationsergebnisse bleiben gespeichert. Änderst du Kriterien, Priorität oder Mindestwert, "
               "wird die Rangfolge sofort neu berechnet, ohne neu zu simulieren.")
    js = optimum_je_szenario(detail, bewertung)
    r = robuste_empfehlung(detail, bewertung).sort_values("rang")
    beste = r.iloc[0]
    if beste.zulaessig:
        st.success(f"Robuste Empfehlung: **{massnahme_txt(beste)}**. Lieferfähigkeit im schlechtesten Szenario "
                   f"{de(beste.beta_min * 100, 1)} %, im Mittel {de(beste.beta * 100, 1)} %, "
                   f"{de(beste.kosten_pro_uhr, 2)} € und {de(beste.co2e_pro_uhr_kg * 1000, 1)} g CO2e je Uhr, "
                   f"{de(beste.fehlteil_tage, 1)} Tage Montagestillstand im Mittel. Bindet im Mittel "
                   f"{de(beste.kapital_gebunden_eur)} € Kapital, die Zinsen darauf "
                   f"({de(beste.kapitalbindung_pro_uhr, 2)} € je Uhr) sind in den Kosten enthalten.")
    else:
        st.warning(f"Keine Kombination erfüllt die Mindestanforderung in allen Szenarien. Am nächsten kommt "
                   f"**{massnahme_txt(beste)}** mit {de(beste.beta_min * 100, 1)} % im schlechtesten Szenario.")

    def spalten(d: pd.DataFrame) -> dict:
        return {"Lieferfähigkeit [%]": (d.beta * 100).round(1).values,
                "Kosten [€/Uhr]": d.kosten_pro_uhr.round(2).values,
                "CO2e [g/Uhr]": (d.co2e_pro_uhr_kg * 1000).round(1).values,
                "Lieferzeit 95 % [Tage]": d.lieferzeit_p95.round(1).values,
                "Stillstand [Tage]": d.fehlteil_tage.round(1).values,
                "davon Kapitalbindung [€/Uhr]": d.kapitalbindung_pro_uhr.round(2).values,
                "Gebundenes Kapital Ø [€]": d.kapital_gebunden_eur.round(-2).astype(int).values}

    st.markdown("**Optimum je Szenario**")
    st.dataframe(pd.DataFrame({
        "Szenario": [f"{z} · {SZ[z]['name']}" if z in SZ else z for z in js.szenario],
        "Beste Maßnahmen": [massnahme_txt(x) for x in js.itertuples()]} | spalten(js) | {
        "Ziel erreicht": js.zulaessig.map({True: "✅", False: "❌"}).values,
        "Gleich wie robuste Empfehlung": ["ja" if massnahme_txt(x) == massnahme_txt(beste) else "nein"
                                          for x in js.itertuples()],
    }), width="stretch", hide_index=True)
    st.plotly_chart(viz.pareto_diagramm(r), width="stretch")
    st.markdown("**Alle Kombinationen über alle Szenarien** (Rang nach deinen Kriterien)")
    st.dataframe(pd.DataFrame({
        "Rang": r.rang.values, "Maßnahmen": [massnahme_txt(x) for x in r.itertuples()],
        "Lieferfähigkeit schlechtestes Szenario [%]": (r.beta_min * 100).round(1).values} | spalten(r) | {
        "Mindestanforderung überall erfüllt": r.zulaessig.map({True: "✅", False: "❌"}).values,
    }), width="stretch", hide_index=True)
    st.download_button("Alle Läufe als CSV", detail.to_csv(index=False).encode("utf-8"),
                       "optimierung_detail.csv", "text/csv")
else:
    st.info("Noch keine Optimierung gerechnet. Kriterien wählen und „Optimierung starten“ drücken.")

with st.expander("Modell und Daten"):
    st.markdown("**Nachschub je Komponente (Bestellpunktverfahren)**  \n"
                "Die Wiederbeschaffungszeit (WBZ) wird automatisch berechnet: Fertigung beim Lieferanten + "
                "Transport auf der aktuell besten Route + Umschlag + Zoll. Ändert ein Szenario die Route dauerhaft "
                "(z. B. Suez wieder offen), wird die WBZ angepasst. Sinkt der Bestand inklusive offener "
                "Bestellungen unter den Meldebestand, wird die Bestellmenge nachbestellt.")
    zeilen = []
    for k, pol in erg.politik.items():
        q = m.primaerquelle(k)
        zeilen.append({
            "Komponente": viz.komp_name(m, k), "Lieferant": kurzname(m.knoten[q.lieferant].name),
            "Preis [€]": q.preis_eur, "WBZ [Tage]": round(pol.wbz_tage, 1), "Bedarf [Stück/Tag]": round(pol.bedarf_tag, 1),
            "Sicherheitsbestand": round(pol.sicherheitsbestand + pol.puffer_extra),
            "Meldebestand": round(pol.meldebestand), "Bestellmenge": round(pol.losgroesse),
        })
    st.dataframe(pd.DataFrame(zeilen), width="stretch", hide_index=True)
    st.markdown(f"**Ereignisprotokoll** ({sid}, eine Wiederholung, {de(len(erg.log))} Zeilen)")
    st.dataframe(erg.log.head(1000), width="stretch", hide_index=True)
    st.download_button("Ereignisprotokoll als CSV", erg.log.to_csv(index=False).encode("utf-8"),
                       f"ereignisprotokoll_{sid}.csv", "text/csv")
