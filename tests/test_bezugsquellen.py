"""Zusätzliche Bezugsquellen (Portland, China, Israel, Japan, Spanien, Marokko, Schweiz): Umschaltung, Ausschluss, Ausfall.

Kennzeichnung: G (Fachlogik N. Scharly, Code mit Claude Sonnet 5.5 umgesetzt, Standortdaten sind Annahmen [A]),
siehe KI_NACHWEIS.md.
Start: pytest -q tests/test_bezugsquellen.py
"""

import pytest

from uhrensim.data import KonfigurationsFehler, lade_modell
from uhrensim.evaluation import kennzahlen
from uhrensim.experiment import szenario_pfad
from uhrensim.handlungsoptionen import _optionen_fuer, lade_optionen
from uhrensim.netzanalyse import ausschluss_regeln, genutzte_knoten, pruefe_ausschluss, single_points_of_failure
from uhrensim.simulation import simuliere

KURZ = {"simulation": {"auswertung_tage": 120}}


def _mit_bezug(bezug: dict, szenario=None):
    return lade_modell(szenario, KURZ | {"massnahmen": {"bezug": bezug}})


@pytest.mark.parametrize("komp,lief", [
    ("saphirglas", "portland_glas"), ("saphirglas", "dongguan_glas"), ("zifferblatt", "haifa_zifferblatt"),
    ("zifferblatt", "tokyo_zifferblatt"), ("zeiger", "barcelona_zeiger"), ("armband", "fes_armband"),
    ("uhrwerk", "eta_grenchen"),
])
def test_bezugsoption_wird_primaer_und_alter_lieferant_option(komp, lief):
    m = _mit_bezug({komp: lief})
    assert m.primaerquelle(komp).lieferant == lief
    assert any(q.rolle == "option" for q in m.quellen[komp])
    assert not m.warnungen


def test_unbekannte_bezugsoption_wird_abgelehnt():
    with pytest.raises(KonfigurationsFehler):
        _mit_bezug({"saphirglas": "miyota_nagano"})


def test_bezug_beeinflusst_nur_die_gewaehlte_komponente():
    m = _mit_bezug({"zeiger": "barcelona_zeiger"})
    assert m.primaerquelle("uhrwerk").lieferant == "miyota_nagano"
    assert m.primaerquelle("saphirglas").lieferant == "rochester_glas"


def test_lkw_quelle_hat_keinen_seeweg_und_weniger_co2e():
    basis = kennzahlen(simuliere(lade_modell(None, KURZ), 0))
    ch = kennzahlen(simuliere(_mit_bezug({"uhrwerk": "eta_grenchen"}), 0))
    assert ch["co2e_pro_uhr_kg"] < basis["co2e_pro_uhr_kg"]
    assert ch["kosten_pro_uhr"] > basis["kosten_pro_uhr"]   # 110 statt 70 € je Uhrwerk


def test_uhrwerk_schweiz_macht_miyota_ausfall_wirkungslos():
    ov = {"massnahmen": {"bezug": {"uhrwerk": "eta_grenchen"}}}
    ohne = kennzahlen(simuliere(lade_modell(szenario_pfad("S1"), None), 0))
    mit = kennzahlen(simuliere(lade_modell(szenario_pfad("S1"), ov), 0))
    assert ohne["beta"] < 0.9
    assert mit["beta"] > 0.99


def test_hamburg_sperre_trifft_portland_und_china_nicht():
    for lief in ("portland_glas", "dongguan_glas"):
        k = kennzahlen(simuliere(lade_modell(szenario_pfad("S3"), {"massnahmen": {"bezug": {"saphirglas": lief}}}), 0))
        assert k["beta"] > 0.99, lief


def test_panama_und_suez_sperre_treffen_nur_ihre_quellen():
    s7, s6 = szenario_pfad("S7"), szenario_pfad("S6")
    portland = kennzahlen(simuliere(lade_modell(s7, {"massnahmen": {"bezug": {"saphirglas": "portland_glas"}}}), 0))
    china = kennzahlen(simuliere(lade_modell(s7, {"massnahmen": {"bezug": {"saphirglas": "dongguan_glas"}}}), 0))
    assert portland["beta"] < 0.8 < china["beta"]
    def beta(ov):   # Mittel über drei Wiederholungen, einzelne Läufe streuen
        return sum(kennzahlen(simuliere(lade_modell(s6, ov), i))["beta"] for i in range(3)) / 3
    china6 = beta({"massnahmen": {"bezug": {"saphirglas": "dongguan_glas"}}})
    kap = beta({"massnahmen": {"bezug": {"saphirglas": "dongguan_glas"}, "ausweichroute": True}})
    assert china6 < 0.95 <= kap


def test_ausschluss_china_und_israel_sortiert_optionen_aus():
    m = _mit_bezug({"saphirglas": "dongguan_glas"})
    erg = simuliere(m, 0)
    regeln = ausschluss_regeln(m, laender=["CN"])
    assert pruefe_ausschluss(genutzte_knoten(erg), regeln, m)       # China genutzt: verletzt
    assert not pruefe_ausschluss(genutzte_knoten(erg), ausschluss_regeln(m, laender=["IL"]), m)
    israel = simuliere(_mit_bezug({"zifferblatt": "haifa_zifferblatt"}), 0)
    assert pruefe_ausschluss(genutzte_knoten(israel), ausschluss_regeln(m, laender=["IL"]), m)
    basis = simuliere(lade_modell(None, KURZ), 0)
    assert not pruefe_ausschluss(genutzte_knoten(basis), ausschluss_regeln(m, laender=["CN", "IL"]), m)


def test_spof_nennt_bezugsoption_als_absicherung():
    m = lade_modell()
    df = single_points_of_failure(m, lade_optionen())
    zeile = df[(df.komponente == "uhrwerk") & (df.element == "miyota_nagano")].iloc[0]
    assert "Uhrwerk Schweiz" in zeile.abgesichert_durch


def test_nur_relevante_optionen_je_szenario_werden_gerechnet():
    optionen = lade_optionen()
    s7 = {o["id"] for o in _optionen_fuer(szenario_pfad("S7"), optionen)}
    assert "keine" in s7 and "glas_portland" in s7 and "dial_japan" not in s7
