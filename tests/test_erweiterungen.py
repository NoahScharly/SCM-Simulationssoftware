# Urheberschaft: Ferdinand Lautenbach (F), einzelner Test Philipp Maywald (P), markiert; KI-gestützt (Claude)
"""Tests der Erweiterungen von Version 2: p*, SPOF, Bandbreite, Szenario S5."""

import math

import pandas as pd
import pytest

from uhrensim.data import lade_modell
from uhrensim.experiment import szenario_pfad
from uhrensim.handlungsoptionen import bandbreite_definition, break_even, lade_optionen, p_text, rangiere
from uhrensim.netzanalyse import single_points_of_failure


@pytest.fixture(scope="module")
def modell():
    return lade_modell()


def _detail(kosten: dict[tuple[str, str], float], menge: float = 1000.0) -> pd.DataFrame:
    return pd.DataFrame([{"szenario": s, "option": o, "kosten_pro_uhr": k, "auftraege": menge}
                         for (s, o), k in kosten.items()])


def test_break_even_formel():
    # Normalbetrieb: Option kostet 1 €/Uhr mehr (M = 1.000 €), Störung: spart 3 €/Uhr (E = 3.000 €) -> p* = 25 %
    d = _detail({("S0", "keine"): 100, ("S0", "puffer"): 101, ("S1", "keine"): 110, ("S1", "puffer"): 107})
    r = break_even(d).iloc[0]
    assert r.p_art == "ab"
    assert r.p_stern == pytest.approx(1000 / (1000 + 3000))
    assert "25" in p_text(r.p_art, r.p_stern)


def test_break_even_sonderfaelle():
    d = _detail({("S0", "keine"): 100, ("S0", "a"): 100, ("S0", "b"): 102, ("S0", "c"): 100,
                 ("S1", "keine"): 110, ("S1", "a"): 105, ("S1", "b"): 111, ("S1", "c"): 110})
    arten = break_even(d).set_index("option").p_art
    assert arten["a"] == "immer"   # keine Mehrkosten, spart im Störfall
    assert arten["b"] == "nie"     # kostet und hilft nicht
    assert arten["c"] == "ohne"    # wirkt nicht, kostet nichts


def test_rauschen_entscheidet_nicht():
    # 0,03 Pp Unterschied in der Lieferfähigkeit darf eine 1,50 € teurere Option nicht nach vorn bringen
    d = pd.DataFrame([{"option": "keine", "beta": 0.9997, "kosten_pro_uhr": 192.0},
                      {"option": "puffer", "beta": 1.0, "kosten_pro_uhr": 193.5}])
    r = rangiere(d, {"beta": 0.95}, ["beta", "kosten_pro_uhr"])
    assert r.iloc[0].option == "keine"


def test_gepaarter_test_entscheidet_statt_schwelle():
    """Nur CO2e priorisiert, Ersatz-Uhrwerk 4,9 g besser (unter der alten Schwelle von 5 g), Puffer 14 € billiger.
    Gesicherter Unterschied (kleine Streuung der Differenzen): Ersatz-Uhrwerk gewinnt. Nicht gesichert (Differenz
    schwankt um die 0): gleichwertig, die Kosten entscheiden."""
    puffer = {r: 0.0594 + 0.002 * ((-1) ** r) for r in range(10)}
    gesichert = {r: v - 0.0049 + 0.0002 * ((-1) ** r) for r, v in puffer.items()}
    rauschen = {r: v - 0.0049 + 0.03 * ((-1) ** r) for r, v in puffer.items()}
    for ersatz, sieger in ((gesichert, "ersatz"), (rauschen, "puffer")):
        d = pd.DataFrame([{"option": o, "beta": 1.0, "kosten_pro_uhr": k, "lieferzeit_p95": 2.1,
                           "co2e_pro_uhr_kg": sum(w.values()) / len(w), "_laeufe_co2e_pro_uhr_kg": w}
                          for o, k, w in (("puffer", 247.18, puffer), ("ersatz", 261.21, ersatz))])
        assert rangiere(d, {"beta": 0.95, "lieferzeit_p95": 3}, ["co2e_pro_uhr_kg"]).iloc[0].option == sieger


def test_spof(modell):
    spof = single_points_of_failure(modell, lade_optionen())
    spof = spof[spof.komponente == "uhrwerk"].set_index("element").combine_first(
        spof[spof.komponente == "saphirglas"].set_index("element"))
    assert bool(spof.loc["rochester_glas", "spof"])          # einziger Saphirglas-Lieferant
    assert not bool(spof.loc["miyota_nagano", "spof"])       # Ersatz-Uhrwerk Sellita
    assert "Ersatz-Uhrwerk" in spof.loc["miyota_nagano", "abgesichert_durch"]
    assert "Ausweichroute" in spof.loc["e_rtm_hn_binnen", "abgesichert_durch"]


# Urheberschaft dieses Tests: Philipp Maywald (P)
def test_szenario_s5_und_bandbreiten():
    m = lade_modell(szenario_pfad("S5"))
    assert m.stoerungen[0].typ == "nachfrage" and m.stoerungen[0].faktor == pytest.approx(0.7)
    for sid in ("S1", "S2", "S3", "S4", "S5"):
        defi = bandbreite_definition(szenario_pfad(sid))
        assert defi is not None and len(defi[1]) >= 3
    assert bandbreite_definition(szenario_pfad("S0")) is None
    assert not math.isnan(m.horizont)
