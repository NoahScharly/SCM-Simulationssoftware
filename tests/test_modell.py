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
from uhrensim.simulation import simuliere

KURZ = {"simulation": {"auswertung_tage": 120}}


# --------------------------------------------------------------------------- Konfiguration
def test_basis_laedt_und_ist_plausibel():
    m = lade_modell()
    assert len(m.komponenten) == 7
    assert m.werk == "heilbronn"
    assert not m.warnungen, m.warnungen


def test_alle_szenarien_laden():
    for sid in ("S0", "S1", "S2", "S3", "S4", "C1"):
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


def test_niedrigwasser_leitet_auf_bahn_um():
    m = lade_modell(szenario_pfad("S2"))
    erg = simuliere(m)
    s = m.stoerungen[0]
    bahn = erg.log[(erg.log.ereignis == "abfahrt") & (erg.log.kante == "e_rtm_hn_bahn")]
    assert ((bahn.zeit >= s.start) & (bahn.zeit < s.ende)).any()
    binnen = erg.log[(erg.log.ereignis == "abfahrt") & (erg.log.kante == "e_rtm_hn_binnen")]
    assert not ((binnen.zeit >= s.start) & (binnen.zeit < s.ende)).any()


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


def test_wbz_folgt_dauerhafter_routenaenderung():
    basis = lade_modell().primaerquelle("uhrwerk").wbz_min_tage
    suez = lade_modell(szenario_pfad("C1")).primaerquelle("uhrwerk").wbz_min_tage
    assert suez < basis


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
    assert len(fig.frames[0].traces) == 8                      # je Karte Sendungen, Störung, Lieferant; 2 Balken
    assert len(fig.frames[0].data[-1].x) == len(erg.politik)


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
