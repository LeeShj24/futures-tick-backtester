import pytest

from ftb import ContractSpec, FeeSchedule, default_contracts, load_contracts


@pytest.mark.parametrize(
    ("symbol", "tick_size", "point_value", "tick_value"),
    [
        ("NQ", 0.25, 20.0, 5.0),
        ("ES", 0.25, 50.0, 12.5),
        ("MNQ", 0.25, 2.0, 0.5),
        ("MES", 0.25, 5.0, 1.25),
    ],
)
def test_default_specs(symbol, tick_size, point_value, tick_value):
    spec = default_contracts()[symbol]
    assert spec.tick_size == tick_size
    assert spec.point_value == point_value
    assert spec.tick_value == pytest.approx(tick_value)


def test_fees_are_per_side_and_sum_components():
    nq = default_contracts()["NQ"]
    assert nq.fees.per_side == pytest.approx(1.38 + 0.50 + 0.02)
    assert nq.fees.round_turn == pytest.approx(2 * 1.90)
    assert nq.commission(3) == pytest.approx(3 * 1.90)
    assert nq.commission(-3) == pytest.approx(3 * 1.90)


@pytest.mark.parametrize(
    ("symbol", "root"),
    [
        ("NQ", "NQ"),
        ("NQZ4", "NQ"),
        ("NQZ24", "NQ"),
        ("MNQH25", "MNQ"),
        ("ESU5", "ES"),
        ("MESM24", "MES"),
        ("NQ.c.0", "NQ"),
        ("ES.FUT", "ES"),
    ],
)
def test_symbol_resolution(symbol, root):
    assert default_contracts().resolve(symbol).symbol == root


def test_unknown_symbol_raises():
    reg = default_contracts()
    with pytest.raises(KeyError):
        reg.resolve("CLZ4")
    assert "CLZ4" not in reg
    assert "NQZ4" in reg


def test_tick_conversion():
    nq = default_contracts()["NQ"]
    assert nq.to_ticks(18000.25) == 72001
    assert nq.from_ticks(72001) == 18000.25
    assert nq.round_price(18000.3) == 18000.25
    assert nq.is_on_grid(18000.75)
    assert not nq.is_on_grid(18000.1)
    with pytest.raises(ValueError):
        nq.to_ticks(18000.1)


def test_fine_tick_sizes_round_trip():
    spec = ContractSpec("6E", tick_size=0.00005, point_value=125_000)
    assert spec.from_ticks(spec.to_ticks(1.08765)) == 1.08765
    assert spec.tick_value == pytest.approx(6.25)


def test_pnl_helper():
    es = default_contracts()["ES"]
    assert es.pnl(5000.0, 5001.25, 2) == pytest.approx(125.0)
    assert es.pnl(5000.0, 5001.25, -2) == pytest.approx(-125.0)


def test_yaml_overrides_and_extends(tmp_path):
    path = tmp_path / "contracts.yaml"
    path.write_text(
        """
contracts:
  NQ:
    tick_size: 0.25
    point_value: 20
    fees: {exchange: 1.38, broker: 1.00, regulatory: 0.02}
  RTY:
    description: E-mini Russell 2000
    tick_size: 0.1
    point_value: 50
    fees: {exchange: 1.38, broker: 0.5}
"""
    )
    reg = load_contracts(path)
    assert reg["NQ"].fees.broker == 1.00
    assert reg["RTY"].tick_value == pytest.approx(5.0)
    assert reg["ES"].fees.broker == 0.50  # defaults kept
    only = load_contracts(path, include_defaults=False)
    assert set(only) == {"NQ", "RTY"}


@pytest.mark.parametrize(
    "body",
    [
        "contracts: {XX: {tick_size: -0.25, point_value: 20}}",
        "contracts: {XX: {tick_size: 0.25, point_value: 0}}",
        "contracts: {XX: {tick_size: 0.25}}",
        "contracts: {XX: {tick_size: 0.25, point_value: 20, fees: {clearing: 1}}}",
        "contracts: {XX: {tick_size: 0.25, point_value: 20, fees: {broker: -1}}}",
        "contracts: {XX: {tick_size: 0.25, point_value: 20, multiplier: 2}}",
        "something_else: 1",
    ],
)
def test_invalid_yaml_is_rejected(tmp_path, body):
    path = tmp_path / "bad.yaml"
    path.write_text(body)
    with pytest.raises(ValueError):
        load_contracts(path)


def test_fee_schedule_validation():
    with pytest.raises(ValueError):
        FeeSchedule(exchange=-0.1)
