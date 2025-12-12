import polars as pl


def compute_rsi(df: pl.DataFrame, price_col: str = "close", period: int = 14) -> pl.Series:
    delta = df.sort("date")[price_col].diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)

    avg_gain = gain.rolling_mean(window_size=period)
    avg_loss = loss.rolling_mean(window_size=period)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
