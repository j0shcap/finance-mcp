"""YFinanceClient.get_analyst_data."""

import pandas as pd
import pytest
from yfinance.exceptions import (
    YFException,
)

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.models import (
    AnalystData,
)
from finance_mcp.data.yahoo import _recommendation_trend
from tests.fakes import (
    fake_ticker_factory,
    make_client,
    make_recommendations_df,
)

ANALYST_INFO = {
    "longName": "Apple Inc.",
    "shortName": "Apple",
    "currency": "USD",
    "currentPrice": 190.0,
    "recommendationKey": "buy",
    "recommendationMean": 1.9,
    "numberOfAnalystOpinions": 40,
    "targetMeanPrice": 220.0,
    "targetMedianPrice": 218.0,
    "targetHighPrice": 300.0,
    "targetLowPrice": 150.0,
}


ANALYST_TREND = [
    ("0m", 12, 20, 6, 1, 0),
    ("-1m", 11, 19, 7, 1, 0),
    ("-2m", 10, 18, 8, 2, 1),
    ("-3m", 9, 17, 9, 2, 1),
]


def test_get_analyst_data_happy_path() -> None:
    df = make_recommendations_df(ANALYST_TREND)
    client = make_client(factory=fake_ticker_factory(info=ANALYST_INFO, recommendations=df))
    a = client.get_analyst_data("AAPL")
    assert isinstance(a, AnalystData)
    assert a.symbol == "AAPL" and a.currency == "USD"
    assert a.current_price == 190.0
    assert a.recommendation_key == "buy" and a.recommendation_mean == 1.9
    assert a.number_of_analysts == 40
    assert a.target_mean_price == 220.0 and a.target_median_price == 218.0
    assert a.target_high_price == 300.0 and a.target_low_price == 150.0
    assert len(a.recommendation_trend) == 4
    first = a.recommendation_trend[0]
    assert first.period == "0m"
    assert (first.strong_buy, first.buy, first.hold, first.sell, first.strong_sell) == (
        12,
        20,
        6,
        1,
        0,
    )
    assert [p.period for p in a.recommendation_trend] == ["0m", "-1m", "-2m", "-3m"]
    last = a.recommendation_trend[-1]
    assert (last.strong_buy, last.buy, last.hold, last.sell, last.strong_sell) == (9, 17, 9, 2, 1)


def test_get_analyst_data_no_coverage_is_data_unavailable_not_symbol_not_found() -> None:
    info = {"longName": "SPDR S&P 500 ETF", "currency": "USD"}
    client = make_client(factory=fake_ticker_factory(info=info))
    with pytest.raises(DataUnavailable) as exc:
        client.get_analyst_data("SPY")
    assert type(exc.value) is DataUnavailable
    assert "No analyst coverage for 'SPY'" in str(exc.value)


def test_get_analyst_data_raw_error_is_symbol_not_found() -> None:
    client = make_client(factory=fake_ticker_factory(info_error=KeyError("boom")))
    with pytest.raises(SymbolNotFound) as exc:
        client.get_analyst_data("AAPL")
    assert "No analyst data for 'AAPL'" in str(exc.value)


def test_get_analyst_data_empty_info_is_symbol_not_found() -> None:
    client = make_client(factory=fake_ticker_factory(info={}))
    with pytest.raises(SymbolNotFound):
        client.get_analyst_data("BAD")


def test_get_analyst_data_typed_error_is_data_unavailable() -> None:
    client = make_client(factory=fake_ticker_factory(info_error=YFException("rate limited")))
    with pytest.raises(DataUnavailable) as exc:
        client.get_analyst_data("AAPL")
    assert "rate limited" in str(exc.value)


def test_get_analyst_data_nan_and_missing_numerics_are_none() -> None:
    info = {
        "longName": "Apple Inc.",
        "currency": "USD",
        "numberOfAnalystOpinions": 40,  # ensures coverage
        "recommendationMean": float("nan"),
        # target prices all missing
    }
    client = make_client(factory=fake_ticker_factory(info=info))
    a = client.get_analyst_data("AAPL")
    assert a.number_of_analysts == 40
    assert a.recommendation_mean is None
    assert a.target_mean_price is None and a.target_median_price is None
    assert a.target_high_price is None and a.target_low_price is None


def test_get_analyst_data_coverage_via_targets_only_empty_trend() -> None:
    info = {"longName": "Apple Inc.", "currency": "USD", "targetMeanPrice": 220.0}
    client = make_client(factory=fake_ticker_factory(info=info))
    a = client.get_analyst_data("AAPL")
    assert a.target_mean_price == 220.0
    assert a.recommendation_trend == []  # empty recommendations frame


def test_recommendation_trend_none_df_returns_empty() -> None:
    # yfinance can return None for recommendations; the helper must tolerate it.
    assert _recommendation_trend(None) == []


def test_get_analyst_data_parse_error_is_data_unavailable() -> None:
    # Coverage exists (numberOfAnalystOpinions), but a non-coercible recommendation
    # count makes RecommendationPeriod construction fail in the parse stage.
    info = {"longName": "Apple Inc.", "currency": "USD", "numberOfAnalystOpinions": 40}
    bad_df = pd.DataFrame(
        [{"period": "0m", "strongBuy": object(), "buy": 1, "hold": 1, "sell": 0, "strongSell": 0}]
    )
    client = make_client(factory=fake_ticker_factory(info=info, recommendations=bad_df))
    with pytest.raises(DataUnavailable) as exc:
        client.get_analyst_data("AAPL")
    assert "Failed to parse analyst data for 'AAPL'" in str(exc.value)


def test_get_analyst_data_recommendations_read_error_is_data_unavailable() -> None:
    # .info has already identified the instrument, so this is not SymbolNotFound.
    info = {"longName": "Apple Inc.", "currency": "USD", "numberOfAnalystOpinions": 40}
    client = make_client(
        factory=fake_ticker_factory(info=info, recommendations_error=YFException("recs down"))
    )
    with pytest.raises(DataUnavailable) as exc:
        client.get_analyst_data("AAPL")
    assert type(exc.value) is DataUnavailable
    assert "recs down" in str(exc.value)


@pytest.mark.parametrize("field", ["recommendationMean", "targetMeanPrice"])
def test_get_analyst_data_non_numeric_value_is_data_unavailable(field: str) -> None:
    info = {"longName": "Apple Inc.", "numberOfAnalystOpinions": 40, field: "n/a"}
    client = make_client(factory=fake_ticker_factory(info=info))
    with pytest.raises(DataUnavailable) as exc:
        client.get_analyst_data("AAPL")
    assert "Failed to parse analyst data for 'AAPL'" in str(exc.value)
