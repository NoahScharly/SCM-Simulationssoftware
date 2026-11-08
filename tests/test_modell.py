"""Verifikation (Folie 11): Konfiguration, Mengenbilanz, deterministischer Modus, Reproduzierbarkeit.

Kennzeichnung: KI (Claude Opus 5.5), Prompts siehe KI_NACHWEIS.md.
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


def test_dual_sourcing_verbessert_uhrwerk_ausfall():
    ohne = kennzahlen(simuliere(lade_modell(szenario_pfad("S1")), 0))
    mit = kennzahlen(simuliere(lade_modell(szenario_pfad("S1"), {"massnahmen": {"dual_sourcing": True}}), 0))
    assert mit["beta"] > ohne["beta"]
    assert mit["fehlteil_tage"] <= ohne["fehlteil_tage"]


# --------------------------------------------------------------------------- Bewertung
def test_pareto_front():
    df = pd.DataFrame({"kosten_pro_uhr": [1, 2, 3, 2], "co2e_pro_uhr_kg": [3, 2, 1, 3], "beta": [1, 1, 1, 1]})
    assert pareto_front(df, ["kosten_pro_uhr", "co2e_pro_uhr_kg"]).tolist() == [True, True, True, False]


def test_mindestwerte_filtern_unzulaessige_varianten():
    df = pd.DataFrame({"kosten_pro_uhr": [100, 120], "co2e_pro_uhr_kg": [1, 1], "beta": [0.90, 0.97],
                       "lieferzeit_p95": [2, 2]})
    out = bewerte(df, {"methode": "mindestwerte", "mindestwerte": {"beta": 0.95}})
    assert out.iloc[0].kosten_pro_uhr == 120 and not out.iloc[1].zulaessig
