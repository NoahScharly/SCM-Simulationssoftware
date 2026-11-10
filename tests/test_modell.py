"""Verifikation (Folie 11): Konfiguration, Mengenbilanz, deterministischer Modus, Reproduzierbarkeit.

Kennzeichnung: G (Architektur und Fachlogik N. Scharly, Code teilweise selbst, teilweise mit Claude Opus 5.5 umgesetzt, Fehlerbehebung mit KI), siehe KI_NACHWEIS.md.
Start: pytest -q
"""

import math

import pandas as pd
import pytest

from uhrensim.data import KonfigurationsFehler, lade_modell
from uhrensim.evaluation import bewerte, kennzahlen, pareto_front
from uhrensim.experiment import szenario_pfad
from uhrensim.simulation import Simulation, simuliere

KURZ = {"simulation": {"auswertung_tage": 120}}


# --------------------------------------------------------------------------- Konfiguration
def test_basis_laedt_und_ist_plausibel():
    m = lade_modell()
    assert len(m.komponenten) == 7
    assert m.werk == "heilbronn"
    assert not m.warnungen, m.warnungen


def test_alle_szenarien_laden():
    for sid in ("S0", "S1", "S2", "S3", "S4"):
        assert lade_modell(szenario_pfad(sid)).szenario_id == sid


def test_wbz_unter_minimum_wird_abgelehnt():
    with pytest.raises(KonfigurationsFehler, match="physischen Minimum"):
        lade_modell(overrides={"wbz": {"uhrwerk": 10}})


def test_wbz_ueber_minimum_wird_als_zuschlag_uebernommen():
    m = lade_modell(overrides={"wbz": {"uhrwerk": 130}})
    q = m.primaerquelle("uhrwerk")
    assert q.wbz_zuschlag_tage == pytest.approx(130 - q.wbz_min_tage)


@pytest.mark.parametrize("menge", [-1, 25001])
def test_menge_ausserhalb_der_grenzen(menge):
    with pytest.raises(KonfigurationsFehler):
        lade_modell(overrides={"nachfrage": {"uhren_pro_jahr": menge}})


def test_zu_kleine_kapazitaet_erzeugt_warnung():
    m = lade_modell(overrides={"nachfrage": {"uhren_pro_jahr": 25000}, "montage": {"kapazitaet_pro_tag": 50}})
    assert any("Mindestens 102/Tag" in w for w in m.warnungen)


# --------------------------------------------------------------------------- Bilanz und Zeiten
def test_mengenbilanz_je_komponente():
    """Kein Bestand ohne Ereignis: Anfang + Eingang - Verbrauch = Ende; Bestellt = Eingang + unterwegs."""
    sim_m = lade_modell(szenario_pfad("S1"), KURZ)
    from uhrensim.simulation import Simulation
    sim = Simulation(sim_m, replikation=0)
    anfang = dict(sim.bestand)
    erg = sim.lauf()
    log = erg.log
    verbrauch = log[log.ereignis == "montage_start"].menge.sum()
    for k in sim_m.komponenten:
        eingang = log[(log.ereignis == "wareneingang") & (log.komponente == k)].menge.sum()
        assert anfang[k] + eingang - verbrauch == pytest.approx(sim.bestand[k])
        bestellt = log[(log.ereignis == "bestellung") & (log.komponente == k)].menge.sum()
        storniert = log[(log.ereignis == "storno") & (log.komponente == k)].menge.sum()
        assert bestellt - storniert == pytest.approx(eingang + sim.pipeline[k])
        assert sim.bestand[k] >= 0


def test_deterministischer_transit_entspricht_handrechnung():
    """Seeschiff Yokohama-Rotterdam: 26.700 km / 800 km/Tag = 33,375 Tage."""
    erg = simuliere(lade_modell(), deterministisch=True)
    log = erg.log
    ab = log[(log.ereignis == "abfahrt") & (log.kante == "e_yok_rtm_kap")].set_index("objekt_id").zeit
    an = log[(log.ereignis == "ankunft") & (log.kante == "e_yok_rtm_kap")].set_index("objekt_id").zeit
    dauer = (an - ab.reindex(an.index)).dropna()
    assert not dauer.empty
    assert dauer.to_numpy() == pytest.approx(26700 / 800)


def test_lieferzeit_nie_negativ_und_auftraege_vollstaendig():
    erg = simuliere(lade_modell(None, KURZ))
    a = erg.auftraege.dropna()
    assert (a.t_zustellung >= a.t_versand).all() and (a.t_versand >= a.t_eingang).all()
    assert erg.log.event_id.is_unique


# --------------------------------------------------------------------------- Zufall
def test_gleicher_seed_gleiches_ergebnis():
    m = lade_modell(None, KURZ)
    assert kennzahlen(simuliere(m, 3)) == kennzahlen(simuliere(m, 3))


def test_anderer_seed_anderes_ergebnis():
    a = kennzahlen(simuliere(lade_modell(None, KURZ), 0))
    b = kennzahlen(simuliere(lade_modell(None, KURZ | {"simulation": {"auswertung_tage": 120, "seed": 1}}), 0))
    assert a["auftraege"] != b["auftraege"] or a["kosten_gesamt"] != b["kosten_gesamt"]


def test_common_random_numbers_gleiche_nachfrage_in_allen_szenarien():
    e0 = simuliere(lade_modell(szenario_pfad("S0"), KURZ), 2)
    e2 = simuliere(lade_modell(szenario_pfad("S2"), KURZ), 2)
    pd.testing.assert_series_equal(e0.auftraege.t_eingang, e2.auftraege.t_eingang)


# --------------------------------------------------------------------------- Extremwerte und Störungen
def test_keine_nachfrage_kein_absturz():
    k = kennzahlen(simuliere(lade_modell(None, {"nachfrage": {"uhren_pro_jahr": 0}} | KURZ)))
    assert k["auftraege"] == 0 and math.isnan(k["beta"])


def test_kapazitaet_eins_bricht_lieferfaehigkeit_ein():
    k = kennzahlen(simuliere(lade_modell(None, {"montage": {"kapazitaet_pro_tag": 1}} | KURZ)))
    assert k["beta"] < 0.5


def test_niedrigwasser_mit_ausweichroute_leitet_auf_bahn_um():
    m = lade_modell(szenario_pfad("S2"), {"massnahmen": {"ausweichroute": True}})
    erg = simuliere(m)
    s = m.stoerungen[0]
    bahn = erg.log[(erg.log.ereignis == "abfahrt") & (erg.log.kante == "e_rtm_hn_bahn")]
    assert ((bahn.zeit >= s.start) & (bahn.zeit < s.ende)).any()
    binnen = erg.log[(erg.log.ereignis == "abfahrt") & (erg.log.kante == "e_rtm_hn_binnen")]
    assert not ((binnen.zeit >= s.start) & (binnen.zeit < s.ende)).any()


def test_niedrigwasser_ohne_massnahme_wartet_ware():
    m = lade_modell(szenario_pfad("S2"))
    erg = simuliere(m)
    s = m.stoerungen[0]
    assert (erg.log.ereignis == "warten_sperre").any()
    bahn = erg.log[(erg.log.ereignis == "abfahrt") & (erg.log.kante == "e_rtm_hn_bahn")]
    assert bahn.empty


def test_sellita_notfallplan_erst_miyota_dann_sellita_dann_zurueck():
    m = lade_modell(szenario_pfad("S1"), {"massnahmen": {"sellita_notfall": True}})
    erg = simuliere(m, 0)
    s = m.stoerungen[0]
    best = erg.log[(erg.log.ereignis == "bestellung") & (erg.log.komponente == "uhrwerk")]
    vorher = best[best.zeit < s.start]
    waehrend = best[(best.zeit >= s.start) & (best.zeit < s.ende)]
    assert set(vorher.ort) == {"miyota_nagano"}                 # vor dem Ausfall nur Miyota
    assert set(waehrend.ort) == {"sellita_lcf"}                 # während des Ausfalls nur Sellita
    assert waehrend.zeit.min() >= s.start + m.param["massnahmen"]["notfall_reaktion_tage"] - 1e-9
    wechsel = erg.log[erg.log.ereignis == "lieferantenwechsel"]
    assert list(wechsel.zustand_neu) == ["sellita_lcf", "miyota_nagano"]
    ohne = simuliere(lade_modell(szenario_pfad("S0"), {"massnahmen": {"sellita_notfall": True}}), 0)
    assert "sellita_lcf" not in set(ohne.log[ohne.log.ereignis == "bestellung"].ort)  # ohne Ausfall nie Sellita


def test_sellita_macht_unabhaengig_vom_miyota_ausfall():
    ohne = kennzahlen(simuliere(lade_modell(szenario_pfad("S1")), 0))
    mit = kennzahlen(simuliere(lade_modell(szenario_pfad("S1"), {"massnahmen": {"sellita_notfall": True}}), 0))
    assert mit["beta"] > ohne["beta"]
    assert mit["fehlteil_tage"] <= ohne["fehlteil_tage"]
    assert mit["kosten_pro_uhr"] > ohne["kosten_pro_uhr"] - 50  # teureres Werk, aber keine Fehlmengenkosten


def test_montage_wird_aus_jahresmenge_ausgelegt():
    m = lade_modell(overrides={"nachfrage": {"uhren_pro_jahr": 5000}})
    assert m.param["montage"]["mitarbeiter"] == 5          # 5000 / (250 × 0,99) = 20,2 Montagen -> 5 × 5
    assert m.param["montage"]["kapazitaet_pro_tag"] == 25
    assert not m.warnungen or all("Montagekapazität" not in w for w in m.warnungen)


def test_gutquote_fest_mit_uebertrag():
    """99 % Gutquote ohne Zufall: über das Jahr entsteht rund 1 % Ausschuss, nicht 0 % durch Runden."""
    erg = simuliere(lade_modell(None, KURZ), 0)
    gut = erg.log[erg.log.ereignis == "montage_ende"].menge.sum()
    aus = erg.log[erg.log.ereignis == "ausschuss"].menge.sum()
    assert aus / (gut + aus) == pytest.approx(0.01, abs=0.002)


# --------------------------------------------------------------------------- Bewertung
def test_pareto_front():
    df = pd.DataFrame({"kosten_pro_uhr": [1, 2, 3, 2], "co2e_pro_uhr_kg": [3, 2, 1, 3], "beta": [1, 1, 1, 1]})
    assert pareto_front(df, ["kosten_pro_uhr", "co2e_pro_uhr_kg"]).tolist() == [True, True, True, False]


def test_mindestwerte_filtern_unzulaessige_varianten():
    df = pd.DataFrame({"kosten_pro_uhr": [100, 120], "co2e_pro_uhr_kg": [1, 1], "beta": [0.90, 0.97],
                       "lieferzeit_p95": [2, 2]})
    out = bewerte(df, {"methode": "mindestwerte", "mindestwerte": {"beta": 0.95}})
    assert out.iloc[0].kosten_pro_uhr == 120 and not out.iloc[1].zulaessig


# --------------------------------------------------------------------------- Optimierung
def test_optimum_je_szenario_und_robuste_empfehlung():
    from uhrensim.optimization import szenario_optimierung
    raster = {"puffer_uhrwerk": [0, 1000], "sellita_notfall": [False], "luftfracht_notfall": [False]}
    o = szenario_optimierung([szenario_pfad("S0"), szenario_pfad("S1")], raster=raster,
                             overrides=KURZ, replikationen=1)
    assert len(o["detail"]) == 4                               # 2 Kombinationen × 2 Szenarien
    assert list(o["je_szenario"].szenario) == ["S0", "S1"]
    r = o["robust"]
    assert len(r) == 2 and (r.beta_min <= r.beta + 1e-12).all()
    assert r.rang.iloc[0] == 1


def test_bestandsbalken_laufen_synchron_zur_karte():
    from uhrensim.visualization import netzkarte
    erg = simuliere(lade_modell(None, KURZ), replikation=0)
    fig = netzkarte(erg, schritt_tage=10)
    assert len(fig.frames[0].traces) == 13         # je Karte Pakete, Luftroute, Sendungen, Störung, Lieferant; 3 Balken
    assert len(fig.frames[0].data[-2].x) == len(erg.politik)


# --------------------------------------------------------------------------- Bezugsoption und Kapital
def test_armband_italien_ersetzt_brasilien_ohne_zoll():
    m = lade_modell(None, {"massnahmen": {"armband_italien": True}})
    assert m.primaerquelle("armband").lieferant == "santacroce_armband"
    erg = simuliere(m, 0)
    band = erg.log[erg.log.komponente == "armband"]
    assert not (band.ort == "estancia_leder").any()
    assert not (band.ereignis == "zollfreigabe").any()


def test_mehr_bestand_bindet_kapital_und_kostet_zinsen():
    ohne = kennzahlen(simuliere(lade_modell(None, KURZ), 0))
    mit = kennzahlen(simuliere(lade_modell(None, KURZ | {"massnahmen": {"puffer_extra": {"uhrwerk": 1000}}}), 0))
    # 1.000 Uhrwerke × 70 € = 70.000 € mehr gebundenes Kapital
    assert mit["kapital_gebunden_eur"] - ohne["kapital_gebunden_eur"] == pytest.approx(70000, rel=0.15)
    assert mit["kapitalbindung_pro_uhr"] > ohne["kapitalbindung_pro_uhr"]


# --------------------------------------------------------------------------- Kunden
def test_kunden_zufaellig_nach_einwohnern_und_laufzeit_nach_entfernung():
    m = lade_modell()
    assert len(m.kunden) >= 30 and all(k["km"] < 800 for k in m.kunden)   # alle in Deutschland
    erg = simuliere(m, 0)
    a = erg.auftraege.dropna(subset=["t_zustellung"])
    anteil_berlin = (a.kunde == "k_berlin").mean()
    erwartet = next(k["gewicht"] for k in m.kunden if k["id"] == "k_berlin") / sum(k["gewicht"] for k in m.kunden)
    assert abs(anteil_berlin - erwartet) < 0.03                           # Ziehung nach Einwohnergewicht
    lauf = a.t_zustellung - a.t_versand
    nah, fern = lauf[a.km <= 250].mean(), lauf[a.km > 450].mean()
    assert nah < fern                                                     # weiter weg = längere Paketlaufzeit


# --------------------------------------------------------------------------- Zusammenführung der Gruppenkonzepte
def test_personalkosten_fallen_auch_ohne_montage_an():
    """Fixkosten (Idee P. Maywald): Ohne Nachfrage wird nichts montiert, die Belegschaft kostet trotzdem."""
    m = lade_modell(None, {"nachfrage": {"uhren_pro_jahr": 1000}})
    e = simuliere(m, 0)
    personal = e.log[e.log.ereignis == "personal"]
    assert not personal.empty and (personal.kosten_eur > 0).all()
    assert e.log[(e.log.ereignis == "montage_start")].kosten_eur.sum() == 0


def test_aushilfe_nur_bei_rueckstand():
    aus = {"massnahmen": {"aushilfe": {"mitarbeiter": 1, "ausloeser_tage": 5, "mindest_tage": 10, "aufschlag": 0.3}}}
    basis = simuliere(lade_modell(None, aus), 0)
    assert (basis.log.ereignis == "aushilfe").sum() == 0           # Normalbetrieb: kein Rückstand, keine Aushilfe
    s1 = simuliere(lade_modell(szenario_pfad("S1"), aus), 0)
    st = s1.modell.stoerungen[0]
    tage = s1.log[s1.log.ereignis == "aushilfe"].zeit
    assert not ((tage >= st.start) & (tage <= st.ende)).any()      # während des Lieferausfalls fehlen Teile, keine Aushilfe
    s4 = simuliere(lade_modell(szenario_pfad("S4"), {**aus, "nachfrage": {"monatsfaktoren": {11: 1.2, 12: 1.2}}}), 0)
    assert (s4.log.ereignis == "aushilfe").sum() > 0               # starke Saisonspitze: Montage ist der Engpass


def test_reservequelle_hochlauf_und_kapazitaet():
    """Sellita (Idee B. Hufgard): erst nach dem Hochlauf, höchstens die Wochenkapazität."""
    ms = {"sellita_notfall": True, "notfall_hochlauf_tage": 14, "notfall_kapazitaet_stk_woche": 150}
    e = simuliere(lade_modell(szenario_pfad("S1"), {"massnahmen": ms}), 0)
    wechsel = e.log[(e.log.ereignis == "lieferantenwechsel") & (e.log.zustand_neu == "sellita_lcf")].zeit.min()
    ein = e.log[(e.log.ereignis == "wareneingang") & (e.log.objekt_id.isin(
        e.log[(e.log.ereignis == "bestellung") & (e.log.ort == "sellita_lcf")].objekt_id))]
    assert ein.zeit.min() >= wechsel + 14                           # nichts vor Ende des Hochlaufs
    menge, dauer = ein.menge.sum(), ein.zeit.max() - (wechsel + 14)
    assert menge <= 150 / 7 * dauer + ein.menge.max() + 1e-6        # Wochenkapazität eingehalten


@pytest.mark.parametrize("sid", ["S0", "S1", "S2", "S3", "S4", "S5"])
def test_stresstest_mengenbilanz_alle_szenarien(sid):
    """Stresstest (Idee P. Maywald): auch bei Extremwerten kein Absturz, kein negativer Bestand, keine Ware aus dem
    Nichts. Je Komponente gilt: Anfangsbestand + Wareneingang - Verbrauch = Endbestand."""
    from uhrensim.handlungsoptionen import lade_optionen
    for opt in lade_optionen():
        for menge in (500, 25000):
            m = lade_modell(szenario_pfad(sid), {"nachfrage": {"uhren_pro_jahr": menge},
                                                  "massnahmen": opt.get("massnahmen") or {}})
            sim = Simulation(m, 1)
            anfang = dict(sim.bestand)
            e = sim.lauf()
            log = e.log
            for k in m.komponenten:
                zu = log[(log.ereignis == "wareneingang") & (log.komponente == k)].menge.sum()
                ab = log[log.ereignis == "montage_start"].menge.sum()
                assert anfang[k] + zu - ab == pytest.approx(sim.bestand[k])
                assert sim.bestand[k] >= -1e-9


def test_tippfehler_in_konfiguration_bricht_ab():
    """Strenge Feldnamen (Idee P. Maywald)."""
    with pytest.raises(KonfigurationsFehler, match="vermutlich 'uhren_pro_jahr'"):
        lade_modell(None, {"nachfrage": {"uhren_pro_jahre": 5000}})
    with pytest.raises(KonfigurationsFehler, match="unbekannte Felder"):
        lade_modell({"id": "SX", "stoerungen": [{"typ": "nachfrage", "fakor": 0.5}]})
    lade_modell(None, {"nachfrage": {"monatsfaktoren": {11: 1.3}}})   # freie Zuordnung bleibt erlaubt


def test_konfig_hash_im_protokoll():
    """Reproduzierbarkeit (Idee B. Hufgard): gleicher Hash und Seed -> gleiches Protokoll, andere Konfig -> anderer Hash."""
    a, b = simuliere(lade_modell(szenario_pfad("S1")), 0), simuliere(lade_modell(szenario_pfad("S1")), 0)
    c = lade_modell(szenario_pfad("S2"))
    start = a.log[a.log.ereignis == "laufstart"]
    assert len(start) == 1 and start.objekt_id.iloc[0] == a.modell.konfig_hash != c.konfig_hash
    pd.testing.assert_frame_equal(a.log, b.log)


def test_co2_letzte_meile_getrennt():
    """Inbound und Zustellung getrennt (Idee F. Lautenbach)."""
    k = kennzahlen(simuliere(lade_modell(), 0))
    assert k["co2e_inbound_g"] + k["co2e_letzte_meile_g"] == pytest.approx(k["co2e_pro_uhr_kg"] * 1000)

