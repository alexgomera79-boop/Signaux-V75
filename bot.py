"""Bot de signaux Volatility 75 Index (Deriv, symbole R_75).

Stratégie (bougies H1) :
  - ACHAT : l'EMA20 croise l'EMA50 vers le haut ET le RSI(14) est entre 50 et 70
  - VENTE : l'EMA20 croise l'EMA50 vers le bas  ET le RSI(14) est entre 30 et 50
  - Stop-loss = 1,5 x ATR(14) ; objectif = 3 x ATR(14)

Les prix viennent de l'API publique de Deriv (aucun compte ni clé nécessaire).
Le V75 est ouvert 24h/24, 7j/7.

Usage :
  python bot.py          # analyse et envoie le signal sur Telegram
  python bot.py --test   # envoie seulement un message de test
  python bot.py --dry    # analyse et affiche, sans envoyer
"""

import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

# ----------------------------- RÉGLAGES ------------------------------------
NAME = "Volatility 75 Index"
SYMBOL = "R_75"
GRANULARITY = 3600        # secondes : 3600 = bougies H1
COUNT = 500               # nombre de bougies téléchargées
WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"
EMA_FAST, EMA_SLOW = 20, 50
RSI_PERIOD = 14
ATR_PERIOD = 14
SL_ATR, TP_ATR = 1.5, 3.0
MAX_SIGNAL_AGE_MIN = 59   # n'envoie que si la bougie vient de se fermer (anti-doublon)
# ---------------------------------------------------------------------------


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["Close"].shift()
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def fetch() -> pd.DataFrame:
    """Télécharge les bougies H1 du V75 (la bougie en cours est retirée)."""
    import websocket  # paquet "websocket-client"

    ws = websocket.create_connection(WS_URL, timeout=20)
    try:
        ws.send(json.dumps({
            "ticks_history": SYMBOL,
            "end": "latest",
            "count": COUNT,
            "style": "candles",
            "granularity": GRANULARITY,
        }))
        data = json.loads(ws.recv())
    finally:
        ws.close()

    if "error" in data:
        raise RuntimeError(data["error"].get("message", "erreur Deriv"))
    candles = data.get("candles") or []
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df.index = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df = df[["Open", "High", "Low", "Close"]].astype(float).dropna()

    now = pd.Timestamp.now(tz="UTC")
    if len(df) and df.index[-1] + pd.Timedelta(seconds=GRANULARITY) > now:
        df = df.iloc[:-1]
    return df


def detect_signal(df: pd.DataFrame):
    """Analyse la dernière bougie clôturée. Retourne un dict ou None."""
    if len(df) < EMA_SLOW + 5:
        return None
    df = df.copy()
    df["ema_fast"] = ema(df["Close"], EMA_FAST)
    df["ema_slow"] = ema(df["Close"], EMA_SLOW)
    df["rsi"] = rsi(df["Close"], RSI_PERIOD)
    df["atr"] = atr(df, ATR_PERIOD)
    last, prev = df.iloc[-1], df.iloc[-2]
    if last[["ema_fast", "ema_slow", "rsi", "atr"]].isna().any():
        return None

    crossed_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
    crossed_down = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

    price, a = float(last["Close"]), float(last["atr"])
    if crossed_up and 50 <= last["rsi"] <= 70:
        side, sl, tp = "ACHAT", price - SL_ATR * a, price + TP_ATR * a
    elif crossed_down and 30 <= last["rsi"] <= 50:
        side, sl, tp = "VENTE", price + SL_ATR * a, price - TP_ATR * a
    else:
        return None
    return {"side": side, "price": price, "sl": sl, "tp": tp,
            "rsi": float(last["rsi"]), "time": df.index[-1]}


def is_fresh(sig: dict, now=None) -> bool:
    """Vrai si la bougie du signal vient de se fermer (évite les doublons)."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    t = pd.Timestamp(sig["time"])
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return now - (t + pd.Timedelta(seconds=GRANULARITY)) <= pd.Timedelta(minutes=MAX_SIGNAL_AGE_MIN)


def send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise SystemExit("TELEGRAM_TOKEN ou TELEGRAM_CHAT_ID manquant.")
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": text},
        timeout=20,
    )
    r.raise_for_status()


def build_message(sig: dict) -> str:
    emoji = "🟢" if sig["side"] == "ACHAT" else "🔴"
    return (
        f"{emoji} SIGNAL {sig['side']} - {NAME} (H1)\n"
        f"Entrée : {sig['price']:.2f}\n"
        f"Stop-loss : {sig['sl']:.2f}\n"
        f"Objectif : {sig['tp']:.2f}\n"
        f"RSI : {sig['rsi']:.0f}\n"
        f"Bougie : {sig['time']:%Y-%m-%d %H:%M} UTC\n\n"
        "Rappel : le V75 est un indice synthétique, aucun signal n'est garanti. "
        "Testez en démo et risquez très peu par trade."
    )


def main() -> None:
    args = sys.argv[1:]
    if "--test" in args:
        send_telegram("✅ Test réussi : votre bot V75 est bien connecté.")
        print("Message de test envoyé.")
        return

    dry = "--dry" in args
    found = 0
    try:
        sig = detect_signal(fetch())
    except Exception as e:
        raise SystemExit(f"[{NAME}] erreur : {e}")

    if sig is None:
        print(f"[{NAME}] pas de signal.")
    elif not is_fresh(sig):
        print(f"[{NAME}] signal déjà traité lors d'un passage précédent, ignoré.")
    else:
        found = 1
        msg = build_message(sig)
        print(msg, "\n")
        if not dry:
            send_telegram(msg)
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC - {found} signal(s).")


if __name__ == "__main__":
    main()
        
