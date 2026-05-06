"""Refresh CSI300 constituents → ~/quant-trading/data/cn_universe.json.

Run periodically (monthly is fine — CSI300 reshuffles twice a year).
"""
import json
from datetime import datetime
from pathlib import Path

import akshare as ak


def to_yf_suffix(code: str) -> str:
    """Add Shanghai/Shenzhen exchange suffix to a 6-digit A-share code."""
    code = str(code).zfill(6)
    # Shanghai main board (60xxxx, 68xxxx STAR), Shanghai ETF (51xxxx, 58xxxx)
    if code.startswith(("60", "68", "51", "58")):
        return f"{code}.SH"
    # Shenzhen main board (00xxxx), ChiNext (30xxxx), Shenzhen ETF (15xxxx, 16xxxx)
    if code.startswith(("00", "30", "15", "16")):
        return f"{code}.SZ"
    # Fallback: assume Shanghai
    return f"{code}.SH"


def main() -> None:
    df = ak.index_stock_cons_csindex(symbol="000300")
    if "成分券代码" in df.columns:
        codes = df["成分券代码"]
    else:
        # Defensive: column position fallback
        codes = df.iloc[:, 4]
    tickers = sorted({to_yf_suffix(c) for c in codes if c})

    out = Path.home() / "quant-trading" / "data" / "cn_universe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated": datetime.utcnow().isoformat() + "Z",
        "source": "akshare.index_stock_cons_csindex(symbol='000300')",
        "count": len(tickers),
        "tickers": tickers,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {len(tickers)} tickers → {out}")
    print(f"sample: {tickers[:5]} ... {tickers[-5:]}")


if __name__ == "__main__":
    main()
