"""Streamlit-Oberfläche als Dashboard: Karte, Kennzahlen, Einschätzung, Szenariovergleich (Modul ui).

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
from uhrensim.experiment import vergleiche  # noqa: E402
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
                                   "beschreibung": str(roh.get("beschreibung", ""))}
    return ergebnis


@st.cache_resource(show_spinner=False, max_entries=6)
def rechne(overrides_json: str, ids: tuple[str, ...], reps: int) -> dict:
    sz = szenarien()
    return vergleiche([sz[s]["pfad"] for s in ids], json.loads(overrides_json), reps)


def hinweis(art: str, titel: str, text: str) -> None:
    st.markdown(f'<div class="hinweis {art}"><span class="titel">{titel}</span> {text}</div>',
                unsafe_allow_html=True)


SZ = szenarien()
basis = lade_modell()
g, mo = basis.param["grenzen"], basis.param["montage"]
PRO_MA = float(mo.get("uhren_pro_mitarbeiter_tag", 7))

# --------------------------------------------------------------------------- Seitenleiste
with st.sidebar:
    st.markdown("## Einstellungen")

    st.markdown("#### 1 · Produktion")
    menge = st.slider("Uhren pro Jahr", int(g["uhren_pro_jahr"][0]), int(g["uhren_pro_jahr"][1]),
                      int(basis.param["nachfrage"]["uhren_pro_jahr"]), step=250,
                      help="Geplante Jahresmenge. Montagekapazität und Mitarbeiterzahl werden daraus abgeleitet.")
    mb = montage_bedarf(menge, g["arbeitstage_pro_jahr"], float(mo["gutquote"]), PRO_MA)
    auslastung = mb["montagen_noetig"] / mb["kapazitaet_pro_tag"] if mb["kapazitaet_pro_tag"] else 0
    st.markdown(
        f'<div class="kasten">Nötig: <b>{de(mb["montagen_noetig"], 1)} Montagen pro Tag</b><br>'
        f'Dafür: <b>{mb["mitarbeiter"]} Mitarbeiter</b> × {de(PRO_MA)} Uhren = '
        f'<b>{de(mb["kapazitaet_pro_tag"])} Montagen/Tag</b><br>'
        f'<span style="color:#6D594C">Auslastung {de(auslastung * 100)} % · {g["arbeitstage_pro_jahr"]} '
        f'Arbeitstage · {de(float(mo["gutquote"]) * 100)} % Gutquote</span></div>',
        unsafe_allow_html=True)

    st.markdown("#### 2 · Vorsorge und Notfallpläne")
    st.caption("Gelten in **allen** gewählten Szenarien gleichzeitig. Der Puffer wirkt immer, ein Notfallplan "
               "greift nur, wenn sein Auslöser eintritt.")
    puffer = st.select_slider(
        "Vorsorge: zusätzlicher Uhrwerk-Puffer", options=[0, 250, 500, 750, 1000, 1500, 2000],
        format_func=lambda v: f"{de(v)} Stück",
        help="Wirkt immer: zusätzlicher Sicherheitsbestand an Uhrwerken in Heilbronn. Überbrückt Lieferausfälle, "
             "bindet aber in jedem Szenario Kapital.")
    sellita = st.toggle(
        "Notfallplan Sellita (Uhrwerk)",
        help="Auslöser: Miyota fällt aus (Szenario S1). Nach "
             f"{de(basis.param['massnahmen'].get('notfall_reaktion_tage', 3))} Tagen Reaktionszeit wird bei Sellita "
             "in La Chaux-de-Fonds (Schweiz) bestellt, noch nicht versandte Miyota-Aufträge werden storniert. "
             "Sellita SW300-1: flacher als das Miyota, kompatibel über Werkhaltering und Zifferblattvariante, doppelter "
             "Preis (140 statt 70 €), per Lkw in rund einem Tag in Heilbronn. Nach dem Wiederanlauf wird wieder "
             "bei Miyota bestellt. Fällt Miyota nicht aus, passiert nichts und es entstehen keine Kosten.")
    luft = st.toggle(
        "Notfallplan Luftfracht",
        help=f"Auslöser: Der Bestand einer Komponente reicht nur noch "
             f"{de(basis.param['massnahmen']['luft_ausloeser_tage'])} Tage. Dann wird der Bedarf von "
             f"{de(basis.param['massnahmen']['luft_menge_tage'])} Tagen per Flugzeug bestellt. Schnell, aber teuer "
             "und mit viel CO2e. Kann in jedem Szenario auslösen.")

    armband_it = st.toggle(
        "Bezugsoption: Lederarmband komplett aus Italien",
        help="Armband und Leder aus Santa Croce sull'Arno (Toskana) statt aus Estancia Velha (Brasilien). Teurer "
             "(28 statt 18 € je Armband, Annahme), aber rund 880 km per Lkw über den Brenner statt Seeweg über Hamburg: "
             "kein Zoll, weniger Ware unterwegs und keine Abhängigkeit von Rio Grande, Hamburg und den Tier-2 in "
             "Brasilien. Wirkt in allen Szenarien.")

    st.markdown("#### 3 · Störungen durchspielen")
    auswahl = st.pills("Welche Szenarien sollen simuliert werden?", list(SZ), default=list(SZ),
                       selection_mode="multi", format_func=lambda s: SZ[s]["kurz"],
                       help="Die Basis wird immer mitgerechnet, sie ist der Vergleichsmaßstab.") or []

    with st.expander("Erweiterte Einstellungen"):
        reps = st.slider("Wiederholungen je Szenario", 3, 30, 10)
        st.caption("Nachfrage, Zollzeiten und Ausschuss sind zufällig. Jedes Szenario wird deshalb mehrfach mit "
                   "anderen Zufallszahlen durchgespielt (Replikationen), angezeigt wird der Mittelwert. Mehr "
                   "Wiederholungen machen das Ergebnis genauer, die Rechnung dauert aber länger.")
        seed = st.number_input("Seed", value=int(basis.param["simulation"]["seed"]), step=1)
        st.caption("Startwert des Zufallsgenerators. Gleicher Seed ergibt beim nächsten Start exakt dieselben "
                   "Zahlen, das macht Ergebnisse nachprüfbar. Alle Szenarien nutzen dieselben Zufallszahlen, "
                   "damit Unterschiede nur von der Störung oder Maßnahme kommen.")

    starten = st.button("Simulation starten", type="primary", width="stretch")

overrides = {
    "nachfrage": {"uhren_pro_jahr": menge},
    "montage": {"kapazitaet_pro_tag": "auto"},
    "simulation": {"seed": int(seed)},
    "massnahmen": {"sellita_notfall": sellita, "puffer_extra": {"uhrwerk": puffer}, "luftfracht_notfall": luft,
                   "armband_italien": armband_it},
}
ids = tuple(["S0", *[s for s in auswahl if s != "S0"]])
schluessel = json.dumps([overrides, ids, reps], sort_keys=True)

# --------------------------------------------------------------------------- Kopf
uhrwerk_txt = "Miyota (JP), Notfallplan Sellita (CH) bereit" if sellita else "Miyota (JP)"
st.title("Uhren-Supply-Chain Heilbronn")
st.markdown(f'<div class="untertitel">Ereignisdiskrete Simulation · {de(menge)} Uhren pro Jahr · '
            f'{mb["mitarbeiter"]} Mitarbeiter in der Montage · Uhrwerk von {uhrwerk_txt} · Lederarmband aus '
            f'{"Italien" if armband_it else "Brasilien"}</div>',
            unsafe_allow_html=True)

if menge == 0:
    st.info("Bei 0 Uhren pro Jahr gibt es nichts zu simulieren. Links eine Jahresmenge einstellen.")
    st.stop()

try:
    pruefung = lade_modell(None, overrides)
except KonfigurationsFehler as fehler:
    st.error(f"Diese Einstellung ist nicht möglich: {fehler}")
    st.stop()

if starten or "lauf" not in st.session_state:
    st.session_state["lauf"] = schluessel
lauf_ov, lauf_ids, lauf_reps = json.loads(st.session_state["lauf"])
with st.spinner(f"Simuliere {len(lauf_ids)} Szenarien mit je {lauf_reps} Wiederholungen …"):
    vgl = rechne(json.dumps(lauf_ov, sort_keys=True), tuple(lauf_ids), int(lauf_reps))
if st.session_state["lauf"] != schluessel:
    st.info("Die Einstellungen wurden geändert. Die Ergebnisse unten gehören noch zum letzten Lauf. "
            "Links auf **Simulation starten** drücken, um neu zu rechnen.")

tab, erg_alle = vgl["tabelle"], vgl["ergebnisse"]
# Anzeigen beziehen sich immer auf die Maßnahmen des gerechneten Laufs, nicht auf geänderte Regler
lauf_mass = lauf_ov.get("massnahmen", {})
sellita = bool(lauf_mass.get("sellita_notfall"))
luft = bool(lauf_mass.get("luftfracht_notfall"))
armband_it = bool(lauf_mass.get("armband_italien"))
puffer = int((lauf_mass.get("puffer_extra") or {}).get("uhrwerk", 0) or 0)
ziel_beta = float(pruefung.param["bewertung"]["mindestwerte"].get("beta", 0.95))
zusage = int(pruefung.param["distribution"]["lieferzusage_tage"])

# --------------------------------------------------------------------------- Szenario und Karte
sid = st.segmented_control("Szenario anzeigen", list(erg_alle), default=list(erg_alle)[0], selection_mode="single",
                           format_func=lambda s: SZ[s]["kurz"], key="szenario") or list(erg_alle)[0]
e = erg_alle[sid]
z = tab.loc[sid]
erg = e["beispiellauf"]
m = erg.modell
st.markdown(f'<div class="untertitel"><b>{sid} · {SZ[sid]["name"]}:</b> {SZ[sid]["beschreibung"]}</div>',
            unsafe_allow_html=True)

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
| **Strich** | Meldebestand s. Sinkt der **helle** Balken auf den Strich, wird automatisch die Bestellmenge Q nachbestellt („⟵ bestellt“), der helle Balken springt nach rechts. |
| Farbe | braun = normal · gelb = Bestellung fällig · rot = weniger als 5 Tage im Lager, Fehlteil droht |

**Warum wird mit dem hellen Balken verglichen?** Sonst würde jeden Tag neu bestellt, solange die erste Lieferung
noch auf dem Schiff ist. Was schon unterwegs ist, zählt mit.

**Beispiel Uhrwerk in diesem Lauf:** Bedarf {de(pol_u.bedarf_tag, 1) if pol_u else "-"} Stück pro Tag,
Wiederbeschaffungszeit {de(pol_u.wbz_tage) if pol_u else "-"} Tage (60 Tage Fertigung bei Miyota plus Schiff über
das Kap). Meldebestand s = Bedarf in der Wiederbeschaffungszeit + Sicherheitsbestand =
{de(pol_u.meldebestand) if pol_u else "-"} Stück, also {de(pol_u.meldebestand / pol_u.bedarf_tag) if pol_u else "-"}
Tage. Bestellmenge Q = {de(pol_u.losgroesse) if pol_u else "-"} Stück. Deshalb liegt der dunkle Balken beim Uhrwerk
meist weit unter dem Strich, ohne dass das ein Problem ist: Der größte Teil steckt in Fertigung und auf dem Schiff.

**Was ändert sich mit den Einstellungen?** Mehr Uhren pro Jahr erhöhen den Bedarf, die Tage bleiben ähnlich. Der
Uhrwerk-Puffer schiebt den Strich beim Uhrwerk nach rechts (mehr Bestand, mehr Kapitalbindung). Der Notfallplan
Sellita ändert den Strich nicht, aber während eines Miyota-Ausfalls kommen Uhrwerke in rund einem Tag per Lkw, der
dunkle Balken erholt sich dann schnell. Luftfracht füllt einen roten Balken innerhalb weniger Tage auf.
""")

# --------------------------------------------------------------------------- Kennzahlen
st.markdown("### Kennzahlen")
ist_basis = sid == "S0"


def delta(spalte: str, faktor: float = 1.0, einheit: str = "", nachkomma: int = 1) -> str | None:
    if ist_basis or f"delta_{spalte}" not in z or pd.isna(z[f"delta_{spalte}"]):
        return None
    d = z[f"delta_{spalte}"] * faktor
    return f"{'+' if d >= 0 else ''}{de(d, nachkomma)} {einheit} ggü. Basis"


karten = [
    ("Lieferfähigkeit", f"{de(z['beta'] * 100, 1)} %", delta("beta", 100, "Pp"), "normal",
     f"Anteil der Bestellungen, die in höchstens {zusage} Werktagen ankommen. Ziel: mindestens "
     f"{de(ziel_beta * 100)} %."),
    ("Kosten je Uhr", f"{de(z['kosten_pro_uhr'], 2)} €", delta("kosten_pro_uhr", 1, "€", 2), "inverse",
     "Einkauf, Montage, Transport, Lager und Verspätungen, geteilt durch die gelieferten Uhren."),
    ("CO2e je Uhr", f"{de(z['co2e_pro_uhr_kg'] * 1000, 1)} g", delta("co2e_pro_uhr_kg", 1000, "g"), "inverse",
     "Emissionen von Transport, Umschlag und Zustellung je Uhr nach GLEC."),
    ("Lieferzeit (95 %)", f"{de(z['lieferzeit_p95'], 1)} Tage", delta("lieferzeit_p95", 1, "Tage"), "inverse",
     "95 % der Kunden haben ihre Uhr spätestens nach so vielen Werktagen."),
    ("Montagestillstand", f"{de(z['fehlteil_tage'], 1)} Tage", None, "inverse",
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
Jedes Szenario wurde {lauf_reps}-mal mit anderen Zufallszahlen simuliert, angezeigt ist der Mittelwert.

**Lieferfähigkeit β** = pünktliche Bestellungen / alle Bestellungen. Pünktlich heißt: zugestellt nach höchstens
{zusage} Werktagen (Zusage aus dem Poster). Wert hier: {de(z['beta'] * 100, 1)} % ± {de(z['beta_ki95'] * 100, 1)} Pp.

**Kosten je Uhr** = (Einkauf + Montage {de(m.param['kosten']['montage_eur_uhr'])} €/Uhr + Paket + Transport je
Tonnenkilometer und Sendung + Umschlag + Zoll + Kapitalbindung {de(m.param['kosten']['kapitalzins_pa'] * 100)} % p. a.
auf den Lagerwert + Lagerplatz + {de(m.param['kosten']['verspaetung_eur_auftrag'])} € je verspäteter Bestellung)
/ gelieferte Uhren.

**CO2e je Uhr** = Σ (Sendungsgewicht × Distanz × GLEC-Faktor des Verkehrsträgers) + Umschlag + Zustellung,
geteilt durch die gelieferten Uhren.

**Lieferzeit (95 %)** = 95-%-Quantil der Werktage zwischen Bestellung und Zustellung.

**Montagestillstand** = Anzahl Arbeitstage mit mindestens einem Fehlteil.

**± Wert (95-%-Konfidenzintervall):** Mit 95 % Wahrscheinlichkeit liegt der wahre Mittelwert in diesem Bereich.
Je mehr Wiederholungen, desto schmaler. Die Differenzen zur Basis sind gepaart berechnet: Basis und Szenario
nutzen dieselben Zufallszahlen, deshalb zeigt die Differenz nur die Wirkung der Störung oder Maßnahme.
""")

# --------------------------------------------------------------------------- Einschätzung
st.markdown("### Was bedeutet das?")


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
    st.markdown(f"**Szenario {sid}: {SZ[sid]['name']}**")
    jahr = float(z["auftraege"])
    spaet = float(z["verspaetet"])
    if z["beta"] >= ziel_beta:
        hinweis("gut", "Lieferversprechen gehalten.",
                f"{de(z['beta'] * 100, 1)} % der Bestellungen kommen pünktlich an (Ziel {de(ziel_beta * 100)} %).")
    else:
        hinweis("schlecht", "Lieferversprechen verfehlt.",
                f"Nur {de(z['beta'] * 100, 1)} % pünktlich. Rund {de(spaet)} von {de(jahr)} Bestellungen im Jahr "
                f"kommen zu spät.")

    # Ursache
    fehl = erg.log[erg.log.ereignis == "fehlteil"]
    komps = sorted({k for x in fehl.komponente.astype(str) for k in x.split(";") if k and k != "nan"})
    if z["fehlteil_tage"] >= 0.5:
        namen = ", ".join(viz.komp_name(m, k) for k in komps) or "Komponenten"
        hinweis("schlecht", "Ursache: Fehlteile.",
                f"An durchschnittlich {de(z['fehlteil_tage'], 1)} Arbeitstagen fehlte {namen}, die Montage stand "
                f"still.")
    elif spaet >= 1:
        kap = m.param["montage"]["kapazitaet_pro_tag"]
        hinweis("warn", "Ursache: Montagekapazität.",
                f"Alle Teile waren da, aber {de(kap)} Montagen/Tag ({m.param['montage']['mitarbeiter']} "
                f"Mitarbeiter) reichen in Nachfragespitzen nicht. Ein weiterer Mitarbeiter brächte "
                f"{de(PRO_MA)} Montagen/Tag Reserve.")

    # Resilienz je Störung
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
                komp = str(d.komponenten.iloc[0]).split(", ")[0]
                pol = erg.politik.get(komp)
                stueck = luecke * pol.bedarf_tag if pol else float("nan")
                hinweis("schlecht", f"{name}: Lager reicht nicht.",
                        f"Der Bestand hält {de(tts_m)} Tage (TTS), bis wieder Ware ankommt, vergehen aber "
                        f"{de(ttr)} Tage (TTR). Es fehlen rund {de(luecke)} Tage, also etwa {de(stueck, 0)} Stück "
                        f"{viz.komp_name(m, komp)}. Ein Puffer in dieser Höhe oder der Notfallplan Sellita würde die "
                        f"Lücke schließen.")

    # Maßnahmen und Kosten
    n_uhr = max(float(z["zugestellt"]), 1.0)
    einkauf = float(z.get("kosten_einkauf", 0) or 0) / n_uhr
    logistik = sum(float(z.get(f"kosten_{a}", 0) or 0) for a in
                   ("transport", "umschlag", "zoll", "kapitalbindung", "lagerplatz")) / n_uhr
    hinweis("", "Kostenstruktur:",
            f"{de(z['kosten_pro_uhr'], 2)} € je Uhr, davon {de(einkauf, 2)} € "
            f"({de(einkauf / z['kosten_pro_uhr'] * 100)} %) für die Komponenten. Transport, Umschlag und Lager "
            f"machen zusammen nur {de(logistik, 2)} € aus.")
    if not ist_basis and not pd.isna(z.get("delta_kosten_pro_uhr")):
        dk, dc = z["delta_kosten_pro_uhr"], z["delta_co2e_pro_uhr_kg"] * 1000
        hinweis("", "Gegenüber der Basis:",
                f"Kosten {'+' if dk >= 0 else ''}{de(dk, 2)} € je Uhr, CO2e {'+' if dc >= 0 else ''}{de(dc, 1)} g "
                f"je Uhr.")
    if luft and z["luftfracht_sendungen"] >= 0.5:
        hinweis("warn", "Luftfracht ausgelöst:",
                f"im Mittel {de(z['luftfracht_sendungen'], 1)} Eilsendungen. Das hält die Montage am Laufen, "
                f"treibt aber Kosten und CO2e.")
    if sellita:
        wechsel = erg.log[erg.log.ereignis == "lieferantenwechsel"]
        if wechsel.empty:
            hinweis("", "Notfallplan Sellita bereit, aber nicht ausgelöst:",
                    "in diesem Szenario fällt Miyota nicht aus, alle Uhrwerke kommen aus Japan, keine Mehrkosten.")
        else:
            zu = wechsel[wechsel.zustand_neu != "miyota_nagano"]
            zurueck = wechsel[wechsel.zustand_neu == "miyota_nagano"]
            n_storno = int((erg.log.ereignis == "storno").sum())
            n_sel = int(((erg.log.ereignis == "bestellung") & (erg.log.ort == "sellita_lcf")).sum())
            txt = (f"ab {viz._datum(zu.zeit.min() - m.einschwingen)} wird bei Sellita bestellt "
                   f"({n_sel} Bestellungen, {n_storno} Miyota-Aufträge storniert)")
            if not zurueck.empty:
                txt += f", ab {viz._datum(zurueck.zeit.min() - m.einschwingen)} wieder bei Miyota"
            hinweis("warn", "Notfallplan Sellita ausgelöst:", txt + ". Kostet den doppelten Preis je Werk, "
                    "verhindert aber den Montagestillstand.")

with rechts:
    st.markdown("**Hinweise zur Konfiguration**")
    probleme = list(dict.fromkeys(pruefung.warnungen + list(e.get("warnungen", []))))
    if auslastung > 0.95:
        probleme.append(f"Die Montage ist zu {de(auslastung * 100)} % ausgelastet. Schwankungen und Saisonspitzen "
                        f"führen dann schnell zu Verspätungen.")
    if puffer and not any(s.typ == "knoten_ausfall" for s in m.stoerungen):
        probleme.append(f"Der Uhrwerk-Puffer von {de(puffer)} Stück kostet in diesem Szenario nur Kapitalbindung, "
                        f"weil kein Lieferant ausfällt.")
    if sellita and puffer:
        probleme.append("Puffer und Notfallplan Sellita sichern denselben Ausfall doppelt ab. Mit Sellita reicht ein "
                        "kleiner Puffer für die Reaktionszeit und die Sellita-Fertigung.")
    if probleme:
        for p in probleme:
            hinweis("warn", "", p)
    else:
        hinweis("gut", "Keine Auffälligkeiten.", "Die Einstellungen sind in sich stimmig.")

# --------------------------------------------------------------------------- Maßnahmen je Szenario
st.markdown("### Welche Maßnahme greift in welchem Szenario?")
st.markdown("**Szenario** = was passiert (eine Störung, von außen vorgegeben). **Maßnahme** = wie ihr vorbereitet "
            "seid (links eingestellt). Jede Simulation rechnet alle gewählten Szenarien mit denselben Maßnahmen. "
            "Die Tabelle zeigt für einen Beispiellauf je Szenario, ob ein Notfallplan tatsächlich ausgelöst wurde.")
zeilen = []
for s_id, e_s in erg_alle.items():
    lg, ms = e_s["beispiellauf"].log, e_s["beispiellauf"].modell
    w = lg[lg.ereignis == "lieferantenwechsel"]
    if not sellita:
        sel = "aus"
    elif w.empty:
        sel = "bereit, nicht ausgelöst"
    else:
        sel = f"ausgelöst {viz._datum(w.zeit.min() - ms.einschwingen)}"
        zur = w[w.zustand_neu == ms.primaerquelle("uhrwerk").lieferant]
        if not zur.empty:
            sel += f" bis {viz._datum(zur.zeit.min() - ms.einschwingen)}"
    luft_n = float(tab.loc[s_id, "luftfracht_sendungen"]) if "luftfracht_sendungen" in tab else 0.0
    zeilen.append({
        "Szenario": f"{s_id} · {SZ[s_id]['name']}",
        "Was passiert": SZ[s_id]["beschreibung"],
        "Puffer": f"wirkt immer (+{de(puffer)} Stück)" if puffer else "aus",
        "Notfallplan Sellita": sel,
        "Notfallplan Luftfracht": ("aus" if not luft else
                                   (f"{de(luft_n, 1)} Eilsendungen (Mittel)" if luft_n >= 0.05 else "bereit, nicht ausgelöst")),
        "Lieferfähigkeit [%]": round(float(tab.loc[s_id, "beta"]) * 100, 1),
    })
st.dataframe(pd.DataFrame(zeilen), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- Szenariovergleich
st.markdown("### Alle Szenarien im Vergleich")
erfuellt = tab[tab.beta >= ziel_beta]
schwach = tab[tab.beta < ziel_beta].sort_values("beta")
satz = f"{len(erfuellt)} von {len(tab)} Szenarien halten das Lieferversprechen."
if not schwach.empty:
    satz += " Kritisch: " + ", ".join(f"{s} {SZ[s]['name']} ({de(r.beta * 100, 1)} %)" for s, r in schwach.iterrows()) + "."
st.markdown(satz)
vergleich = pd.DataFrame({
    "Szenario": [f"{s} · {SZ[s]['name']}" for s in tab.index],
    "Ziel erreicht": ["✅" if b >= ziel_beta else "❌" for b in tab.beta],
    "Lieferfähigkeit [%]": tab.beta * 100,
    "Kosten je Uhr [€]": tab.kosten_pro_uhr,
    "CO2e je Uhr [g]": tab.co2e_pro_uhr_kg * 1000,
    "Lieferzeit 95 % [Tage]": tab.lieferzeit_p95,
    "Stillstand [Tage]": tab.fehlteil_tage,
})
st.dataframe(vergleich, hide_index=True, width="stretch", column_config={
    "Lieferfähigkeit [%]": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
    "Kosten je Uhr [€]": st.column_config.NumberColumn(format="%.2f"),
    "CO2e je Uhr [g]": st.column_config.NumberColumn(format="%.1f"),
    "Lieferzeit 95 % [Tage]": st.column_config.NumberColumn(format="%.1f"),
    "Stillstand [Tage]": st.column_config.NumberColumn(format="%.1f"),
})

# --------------------------------------------------------------------------- Details
st.markdown("### Details")
with st.expander(f"Verlauf über das Jahr ({sid})"):
    st.plotly_chart(viz.serviceverlauf(erg), width="stretch")
    st.plotly_chart(viz.bestandsverlauf(erg), width="stretch")
    komp = st.selectbox("Bestand einer Komponente im Detail", list(erg.politik),
                        format_func=lambda k: viz.komp_name(m, k))
    st.plotly_chart(viz.bestand_einzeln(erg, komp), width="stretch")
    st.caption("Rote Flächen markieren Störungen. Gezeigt ist eine der Wiederholungen.")

with st.expander(f"Kosten im Detail ({sid})"):
    st.plotly_chart(viz.kostenaufteilung(z), width="stretch")

with st.expander("Wie sicher sind die Ergebnisse?"):
    st.markdown("Jeder Punkt der Streuung ist eine Wiederholung mit anderen Zufallszahlen. Liegen die Kästen "
                "verschiedener Szenarien weit auseinander, ist der Unterschied echt und kein Zufall.")
    kz = st.selectbox("Kennzahl", ["beta", "kosten_pro_uhr", "co2e_pro_uhr_kg", "lieferzeit_p95"],
                      format_func=lambda k: viz.BESCHRIFTUNG[k][0])
    st.plotly_chart(viz.boxplot_vergleich(erg_alle, kz), width="stretch")
    st.dataframe(viz.kennzahlentabelle(tab), width="stretch", hide_index=True)
    st.download_button("Kennzahlen als CSV", tab.to_csv().encode("utf-8"), "kennzahlen.csv", "text/csv")

# --------------------------------------------------------------------------- Optimierung
st.markdown("### Optimierung: beste Maßnahmen je Szenario")
st.markdown("Probiert für **jedes Szenario** alle 12 Kombinationen aus Uhrwerk-Puffer (0, 500, 1.000 Stück), "
            "Notfallplan Sellita und Notfallplan Luftfracht durch. Ergebnis: das Optimum je Szenario und eine "
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
