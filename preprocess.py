import pandas as pd

def SMA(df, ndays):
    SMA = pd.Series(df.close.rolling(ndays).mean(), name="SMA_" + str(ndays))
    return SMA.astype(float).round(2)


def BBANDS(df, n):
    MA = df.close.rolling(window=n).mean()
    SD = df.close.rolling(window=n).std()
    upperBand = MA + (2 * SD)
    lowerBand = MA - (2 * SD)
    return upperBand.astype(float).round(2), lowerBand.astype(float).round(2)


def RSI(df, periods=14):
    close_delta = df.close.diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()

    _rsi = ma_up / ma_down
    return (100 - (100 / (1 + _rsi))).astype(float).round(2)


def MACD(df):
    k = df["close"].ewm(span=12, adjust=False, min_periods=12).mean()
    d = df["close"].ewm(span=26, adjust=False, min_periods=26).mean()
    macd = k - d
    macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_h = macd - macd_s
    return df.index.map(macd), df.index.map(macd_s), df.index.map(macd_h)


def preprocess(df):
    df["volume"] = df.volume.astype(float).round(2)
    df["feature_close"] = df.close
    df["feature_open"] = df.open
    df["feature_high"] = df.high
    df["feature_low"] = df.low
    df["feature_volume"] = df.volume
    df["feature_SMA"] = SMA(df, 50)
    df["feature_MiddleBand"], df["feature_LowerBand"] = BBANDS(df, 50)
    df["feature_RSI"] = RSI(df, periods=14)
    df["feature_MACD"], df["feature_MACD_S"], df["feature_MACD_H"] = MACD(df)
    df = df.dropna()

    return df