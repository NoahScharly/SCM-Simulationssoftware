"""Tests für den automatischen Vergleich der Handlungsoptionen.

Kennzeichnung: G (Testfälle N. Scharly, Code mit Claude Opus 5.5 umgesetzt), siehe KI_NACHWEIS.md.
"""

import pandas as pd
import pytest

from uhrensim.experiment import szenario_pfad
from uhrensim.handlungsoptionen import (REFERENZ, lade_optionen, rangiere, robuste_option, vergleiche_optionen,
                                        vor_und_nachteile, wirkt)


def test_optionen_aus_konfiguration_referenz_zuerst():
    opt = lade_optionen()
    assert opt[0]["id"] == REFERENZ
    assert len({o["id"] for o in opt}) == len(opt) >= 4
    assert all("massnahmen" in o for o in opt)


def test_rangiere_erst_mindestwert_dann_prioritaet():
    df = pd.DataFrame({"option": ["billig_schlecht", "teuer_gut", "mittel_gut"],
                       "beta": [0.90, 0.99, 0.97], "kosten_pro_uhr": [180.0, 210.0, 195.0],
                       "co2e_pro_uhr_kg": [0.03, 0.03, 0.04]})
    r = rangiere(df, 0.95, ["kosten_pro_uhr"])
    assert list(r.option) == ["mittel_gut", "teuer_gut", "billig_schlecht"]
    assert not r.set_index("option").loc["billig_schlecht", "zulaessig"]


def test_rangiere_ohne_zulaessige_option_nach_lieferfaehigkeit():
    df = pd.DataFrame({"option": ["a", "b"], "beta": [0.60, 0.80], "kosten_pro_uhr": [150.0, 300.0],
                       "co2e_pro_uhr_kg": [0.03, 0.03]})
    assert rangiere(df, 0.95, ["kosten_pro_uhr"]).iloc[0].option == "b"


def test_vor_und_nachteile_und_wirkung():
    v, n = vor_und_nachteile({"delta_beta": 0.2, "delta_kosten_pro_uhr": 3.0, "delta_co2e_pro_uhr_kg": 0.0,
                              "delta_lieferzeit_p95": 0, "delta_fehlteil_tage": -10, "delta_kapital_gebunden_eur": 0})
    assert any("Lieferfähigkeit" in t for t in v) and any("Stillstand" in t or "Montagestillstand" in t for t in v)
    assert any("€ je Uhr" in t for t in n)
    assert not wirkt({f"delta_{k}": 0.0 for k in ("beta", "kosten_pro_uhr", "co2e_pro_uhr_kg", "lieferzeit_p95",
                                                  "fehlteil_tage", "kapital_gebunden_eur")})


@pytest.fixture(scope="module")
def vergleich():
    opt = [o for o in lade_optionen() if o["id"] in (REFERENZ, "sellita")]
    return vergleiche_optionen([szenario_pfad("S0"), szenario_pfad("S1")], opt, replikationen=2)


def test_notfallplan_wirkt_nur_bei_ausloeser(vergleich):
    d = vergleich["detail"].set_index(["szenario", "option"])
    # S0: Miyota fällt nicht aus, Sellita wird nie ausgelöst, gleiche Zufallszahlen -> identisches Ergebnis
    assert d.loc[("S0", "sellita"), "kosten_pro_uhr"] == pytest.approx(d.loc[("S0", REFERENZ), "kosten_pro_uhr"])
    assert not wirkt(d.loc[("S0", "sellita")])
    # S1: Sellita schließt die Lücke
    assert d.loc[("S1", "sellita"), "beta"] > d.loc[("S1", REFERENZ), "beta"] + 0.1
    assert d.loc[("S1", REFERENZ), "delta_beta"] == 0


def test_robuste_option_nutzt_schlechtestes_szenario(vergleich):
    r = robuste_option(vergleich["detail"], 0.95, ["kosten_pro_uhr"])
    assert r.iloc[0].option == "sellita"
    assert r.set_index("option").loc[REFERENZ, "beta_min"] < 0.95


def test_luftfracht_brueckenlieferung_nach_ausfall():
    """S1: Während des Ausfalls wird nicht geflogen, beim Wiederanlauf kommt eine Brückenlieferung per Flugzeug."""
    from uhrensim.experiment import fuehre_aus
    e = fuehre_aus(szenario_pfad("S1"), {"massnahmen": {"luftfracht_notfall": True}}, 1)
    erg = e["beispiellauf"]
    log, m = erg.log, erg.modell
    st = m.stoerungen[0]
    bruecke = log[log.ereignis == "luft_wiederanlauf"]
    assert len(bruecke) == 1 and bruecke.zeit.iloc[0] == pytest.approx(st.ende)
    flug = log[(log.ereignis == "abfahrt") & log.kante.astype(str).str.contains("luft")]
    assert not ((flug.zeit > st.start) & (flug.zeit < st.ende)).any()
    # Eilzuschlag: Notfallbestellung teurer als regulärer Stückpreis
    b = log[(log.ereignis == "bestellung") & (log.objekt_id.isin(flug.objekt_id))].iloc[0]
    assert b.kosten_eur > b.menge * m.primaerquelle(b.komponente).preis_eur


def test_luftfracht_nicht_im_normalbetrieb():
    from uhrensim.experiment import fuehre_aus
    z = fuehre_aus(szenario_pfad("S0"), {"massnahmen": {"luftfracht_notfall": True}}, 2)["zusammenfassung"]
    assert z.loc["luftfracht_sendungen", "mittel"] == 0
