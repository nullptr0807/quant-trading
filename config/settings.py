"""
Quantitative Trading System - Configuration Settings
US stocks via moomoo Australia broker
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# API Keys
# =============================================================================
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

# =============================================================================
# Stock Universe - ~200 liquid S&P 500 tickers (expanded for diversification)
# =============================================================================
STOCK_UNIVERSE = [
    # Russell 1000 (iShares IWB holdings, Apr 28 2026)
    # Grouped by GICS sector. Class-share tickers normalized for yfinance
    # (BRKB → BRK-B, HEIA → HEI-A, etc.).

    # Information Technology (137)
    "AAPL", "ACN", "ADBE", "ADI", "ADSK", "AKAM", "ALAB", "ALGM", "AMAT", "AMD",
    "AMKR", "ANET", "APH", "APP", "APPF", "ARW", "AUR", "AVGO", "AVT", "BILL",
    "BSY", "CCC", "CDNS", "CDW", "CGNX", "CIEN", "COHR", "CRCL", "CRM", "CRUS",
    "CRWD", "CSCO", "CTSH", "CXT", "DBX", "DDOG", "DELL", "DLB", "DOCU", "DOX",
    "DT", "DXC", "ENPH", "ENTG", "EPAM", "ESTC", "FFIV", "FICO", "FLEX", "FSLR",
    "FTNT", "GDDY", "GEN", "GFS", "GLOB", "GLW", "GTLB", "GWRE", "HPE", "HPQ",
    "HUBS", "IBM", "INGM", "INTC", "INTU", "IOT", "IPGP", "IT", "JBL", "KD",
    "KEYS", "KLAC", "LFUS", "LITE", "LRCX", "LSCC", "MANH", "MCHP", "MDB", "MKSI",
    "MPWR", "MRVL", "MSFT", "MSI", "MSTR", "MTSI", "MU", "NCNO", "NET", "NOW",
    "NTAP", "NTNX", "NVDA", "OKTA", "OLED", "ON", "ONTO", "ORCL", "P", "PANW",
    "PATH", "PCOR", "PEGA", "PLTR", "PTC", "Q", "QCOM", "QRVO", "RAL", "RBRK",
    "RNG", "ROP", "S", "SAIL", "SMCI", "SNDK", "SNOW", "SNPS", "SNX", "SWKS",
    "TDC", "TDY", "TEAM", "TER", "TRMB", "TWLO", "TXN", "TYL", "U", "UI",
    "VNT", "VRSN", "WDAY", "WDC", "ZBRA", "ZM", "ZS",

    # Communication (48)
    "ASTS", "CHTR", "CMCSA", "DIS", "DJT", "DV", "EA", "FOX", "FOXA", "FWONA",
    "FWONK", "GLIBA", "GLIBK", "GOOG", "GOOGL", "GTM", "IAC", "IRDM", "LBRDA", "LBRDK",
    "LBTYA", "LBTYK", "LYV", "META", "MSGS", "MTCH", "NFLX", "NIQ", "NWS", "NWSA",
    "NXST", "NYT", "OMC", "PINS", "RBLX", "RDDT", "ROKU", "SIRI", "SPOT", "T",
    "TIGO", "TKO", "TMUS", "TTD", "TTWO", "VSNT", "VZ", "WBD",

    # Consumer Discretionary (127)
    "ABNB", "ADT", "AMZN", "AN", "APTV", "ARMK", "AS", "AZO", "BBWI", "BBY",
    "BC", "BFAM", "BIRK", "BKNG", "BLD", "BROS", "BURL", "BWA", "BYD", "CAVA",
    "CCL", "CHDN", "CHH", "CHWY", "CMG", "COLM", "CPNG", "CROX", "CVNA", "CZR",
    "DASH", "DDS", "DECK", "DHI", "DKNG", "DKS", "DPZ", "DRI", "DUOL", "EBAY",
    "ETSY", "EXPE", "F", "FIVE", "FLUT", "FND", "GAP", "GM", "GME", "GNTX",
    "GPC", "GRMN", "H", "HAS", "HD", "HLT", "HOG", "HRB", "KMX", "LAD",
    "LCID", "LEA", "LEN", "LEN-B", "LKQ", "LLYVA", "LLYVK", "LOPE", "LOW", "LULU",
    "LVS", "M", "MAR", "MAT", "MCD", "MGM", "MHK", "MTN", "MUSA", "NCLH",
    "NKE", "NVR", "NWL", "OLLI", "ONON", "ORLY", "PAG", "PENN", "PHM", "PLNT",
    "POOL", "PVH", "QS", "QSR", "RCL", "RH", "RIVN", "RL", "ROST", "SBUX",
    "SCI", "SGI", "SN", "THO", "TJX", "TNL", "TOL", "TPR", "TSCO", "TSLA",
    "TXRH", "UA", "UAA", "ULTA", "VFC", "VGNT", "VIK", "VVV", "W", "WEN",
    "WH", "WHR", "WING", "WSM", "WYNN", "YETI", "YUM",

    # Consumer Staples (60)
    "ACI", "ADM", "BF-A", "BF-B", "BG", "BJ", "BRBR", "CAG", "CART", "CASY",
    "CELH", "CHD", "CL", "CLX", "COKE", "COST", "COTY", "CPB", "DAR", "DG",
    "DLTR", "EL", "ELF", "FLO", "FRPT", "GIS", "HRL", "HSY", "INGR", "KDP",
    "KHC", "KMB", "KO", "KR", "KVUE", "LW", "MDLZ", "MKC", "MNST", "MO",
    "PEP", "PFGC", "PG", "PM", "POST", "PPC", "PRMB", "REYN", "SAM", "SEB",
    "SFD", "SFM", "SJM", "STZ", "SYY", "TAP", "TGT", "TSN", "USFD", "WMT",

    # Financials (159)
    "ACGL", "AFG", "AFL", "AFRM", "AGNC", "AGO", "AIG", "AIZ", "AJG", "ALL",
    "ALLY", "AMG", "AMP", "AON", "APO", "ARES", "AXP", "AXS", "BAC", "BAM",
    "BEN", "BHF", "BK", "BLK", "BLSH", "BOKF", "BPOP", "BRK-B", "BRO", "BX",
    "C", "CACC", "CB", "CBC", "CBOE", "CBSH", "CFG", "CFR", "CG", "CINF",
    "CME", "CNA", "COF", "COIN", "COLB", "CPAY", "CRBG", "EEFT", "EG", "EQH",
    "EVR", "EWBC", "FAF", "FCNCA", "FDS", "FHB", "FHN", "FIGR", "FIS", "FISV",
    "FITB", "FNB", "FNF", "FOUR", "FRHC", "GL", "GPN", "GS", "HBAN", "HIG",
    "HLI", "HLNE", "HOOD", "IBKR", "ICE", "IVZ", "JEF", "JHG", "JKHY", "JPM",
    "KEY", "KKR", "KMPR", "KNSL", "L", "LAZ", "LNC", "LPLA", "MA", "MCO",
    "MET", "MKL", "MKTX", "MORN", "MRSH", "MS", "MSCI", "MTB", "MTG", "NDAQ",
    "NLY", "NTRS", "NU", "OMF", "ORI", "OWL", "OZK", "PB", "PFG", "PGR",
    "PNC", "PNFP", "PRI", "PRU", "PYPL", "RF", "RGA", "RITM", "RJF", "RKT",
    "RLI", "RNR", "RYAN", "SCHW", "SEIC", "SF", "SLM", "SOFI", "SPGI", "SSB",
    "STT", "STWD", "SYF", "TFC", "TFSL", "THG", "TOST", "TPG", "TROW", "TRV",
    "TW", "UNM", "USB", "UWMC", "V", "VIRT", "VOYA", "WAL", "WBS", "WEX",
    "WFC", "WRB", "WTFC", "WTM", "WTW", "WU", "XP", "XYZ", "ZION",

    # Health Care (106)
    "A", "ABBV", "ABT", "ACHC", "ALGN", "ALNY", "AMGN", "APLS", "AVTR", "BAX",
    "BDX", "BIIB", "BIO", "BMRN", "BMY", "BRKR", "BSX", "CAH", "CAI", "CERT",
    "CHE", "CI", "CNC", "COO", "COR", "CORT", "CRL", "CVS", "DGX", "DHR",
    "DOCS", "DVA", "DXCM", "EHC", "ELAN", "ELV", "EW", "EXEL", "GEHC", "GILD",
    "GMED", "HALO", "HCA", "HSIC", "HUM", "IDXX", "ILMN", "INCY", "INSM", "INSP",
    "IONS", "IQV", "ISRG", "JAZZ", "JNJ", "LH", "LLY", "MASI", "MCK", "MDLN",
    "MDT", "MEDP", "MOH", "MRK", "MRNA", "MTD", "NBIX", "NTRA", "NVST", "OGN",
    "PEN", "PFE", "PODD", "PRGO", "QGEN", "RARE", "REGN", "RGEN", "RMD", "ROIV",
    "RPRX", "RVMD", "RVTY", "SHC", "SMMT", "SOLV", "SRPT", "STE", "SYK", "TECH",
    "TEM", "TFX", "THC", "TMO", "UHS", "UNH", "UTHR", "VEEV", "VKTX", "VRTX",
    "VTRS", "WAT", "WST", "XRAY", "ZBH", "ZTS",

    # Industrials (172)
    "AAL", "AAON", "ACM", "ADP", "AGCO", "AIT", "ALK", "ALLE", "ALSN", "AME",
    "AMTM", "AOS", "APG", "ATI", "AWI", "AXON", "AYI", "BA", "BAH", "BLDR",
    "BR", "BWXT", "CACI", "CAR", "CARR", "CAT", "CHRW", "CLH", "CLVT", "CMI",
    "CNH", "CNM", "CNXC", "CPRT", "CR", "CRS", "CSL", "CSX", "CTAS", "CW",
    "DAL", "DCI", "DE", "DOV", "DRS", "ECG", "EFX", "EME", "EMR", "ESAB",
    "ETN", "EXLS", "EXPD", "FAST", "FBIN", "FCN", "FDX", "FERG", "FIX", "FLS",
    "FTAI", "FTV", "G", "GD", "GE", "GEV", "GGG", "GNRC", "GTES", "GWW",
    "GXO", "HAYW", "HEI", "HEI-A", "HII", "HON", "HUBB", "HWM", "HXL", "IEX",
    "IR", "ITT", "ITW", "J", "JBHT", "JCI", "KBR", "KEX", "KNX", "KRMN",
    "LDOS", "LECO", "LHX", "LII", "LMT", "LOAR", "LSTR", "LUV", "LYFT", "MAN",
    "MAS", "MIDD", "MLI", "MMM", "MSA", "MSM", "MTZ", "NDSN", "NOC", "NSC",
    "NVT", "OC", "ODFL", "OSK", "OTIS", "PAYC", "PAYX", "PCAR", "PCTY", "PH",
    "PNR", "PSN", "PWR", "QXO", "R", "RBA", "RBC", "RHI", "RKLB", "ROK",
    "ROL", "RRX", "RSG", "RTX", "SAIA", "SAIC", "SARO", "SITE", "SNA", "SNDR",
    "SSD", "SSNC", "ST", "SWK", "TDG", "TKR", "TREX", "TRU", "TT", "TTC",
    "TTEK", "TXT", "UAL", "UBER", "UHAL", "UHAL-B", "UNP", "UPS", "URI", "VLTO",
    "VMI", "VRSK", "VRT", "WAB", "WCC", "WM", "WMS", "WSC", "WSO", "WWD",
    "XPO", "XYL",

    # Energy (36)
    "AM", "APA", "AR", "BKR", "CHRD", "COP", "CTRA", "CVX", "DINO", "DTM",
    "DVN", "EOG", "EQT", "EXE", "FANG", "FTI", "HAL", "KMI", "LNG", "MPC",
    "MTDR", "NOV", "OKE", "OVV", "OXY", "PR", "PSX", "RRC", "SLB", "TPL",
    "TRGP", "VLO", "VNOM", "WFRD", "WMB", "XOM",

    # Materials (54)
    "AA", "ALB", "AMCR", "APD", "ASH", "ATR", "AU", "AVY", "AXTA", "BALL",
    "CCK", "CE", "CF", "CLF", "CRH", "CTVA", "DD", "DOW", "ECL", "EMN",
    "ESI", "EXP", "FCX", "FMC", "GPK", "HUN", "IFF", "IP", "JHX", "LIN",
    "LPX", "LYB", "MLM", "MOS", "MP", "NEM", "NEU", "NUE", "OLN", "PKG",
    "PPG", "RGLD", "RPM", "RS", "SCCO", "SHW", "SLGN", "SMG", "SOLS", "SON",
    "STLD", "SW", "VMC", "WLK",

    # Real Estate (64)
    "ADC", "AMH", "AMT", "ARE", "AVB", "BRX", "BXP", "CBRE", "CCI", "COLD",
    "CPT", "CSGP", "CUBE", "CUZ", "DLR", "DOC", "EGP", "ELS", "EPR", "EQIX",
    "EQR", "ESS", "EXR", "FR", "FRMI", "FRT", "GLPI", "HHH", "HIW", "HR",
    "HST", "INVH", "IRM", "JLL", "KIM", "KRC", "LAMR", "LINE", "MAA", "MPT",
    "MRP", "NNN", "NSA", "O", "OHI", "PK", "PLD", "PSA", "REG", "REXR",
    "RYN", "SBAC", "SPG", "STAG", "SUI", "UDR", "VICI", "VNO", "VTR", "WELL",
    "WPC", "WY", "Z", "ZG",

    # Utilities (41)
    "AEE", "AEP", "AES", "ATO", "AWK", "BEPC", "CEG", "CMS", "CNP", "CWEN",
    "CWEN-A", "D", "DTE", "DUK", "ED", "EIX", "ES", "ETR", "EVRG", "EXC",
    "FE", "IDA", "LNT", "MDU", "NEE", "NFG", "NI", "NRG", "OGE", "PCG",
    "PEG", "PNW", "PPL", "SO", "SRE", "TLN", "UGI", "VST", "WEC", "WTRG",
    "XEL",

]

# =============================================================================
# Trading Hours (UTC) - US market 9:30-16:00 ET = 13:30-20:00 UTC (EST)
# During EDT (Mar-Nov): 13:30-20:00 UTC
# During EST (Nov-Mar): 14:30-21:00 UTC
# =============================================================================
TRADING_HOURS = {
    "market_open_edt": "13:30",
    "market_close_edt": "20:00",
    "market_open_est": "14:30",
    "market_close_est": "21:00",
    "pre_market_open": "08:00",   # UTC, approx
    "post_market_close": "01:00", # UTC next day, approx
    "timezone": "US/Eastern",
}

# =============================================================================
# Moomoo Australia Fee Structure (US stocks)
# =============================================================================
@dataclass(frozen=True)
class FeeStructure:
    commission_per_order: float = 0.0
    platform_fee_per_order: float = 0.99
    settlement_per_share: float = 0.003
    settlement_max_pct: float = 0.01       # max 1% of trade amount
    sec_fee_rate: float = 0.0000206        # sells only, applied to notional
    sec_fee_min: float = 0.01              # minimum SEC fee
    finra_taf_per_share: float = 0.000195  # sells only
    finra_taf_min: float = 0.01
    finra_taf_max: float = 9.79
    slippage_pct: float = 0.0005           # 0.05%

    def calc_fees(self, shares: float, price: float, is_sell: bool = False) -> dict:
        """Calculate total fees for a trade."""
        notional = abs(shares) * price
        platform = self.platform_fee_per_order
        settlement = min(abs(shares) * self.settlement_per_share,
                         notional * self.settlement_max_pct)
        sec = 0.0
        taf = 0.0
        if is_sell:
            sec = max(notional * self.sec_fee_rate, self.sec_fee_min)
            taf_raw = abs(shares) * self.finra_taf_per_share
            taf = max(min(taf_raw, self.finra_taf_max), self.finra_taf_min)
        slippage = notional * self.slippage_pct
        total = platform + settlement + sec + taf + slippage
        return {
            "platform": round(platform, 4),
            "settlement": round(settlement, 4),
            "sec": round(sec, 4),
            "taf": round(taf, 4),
            "slippage": round(slippage, 4),
            "total": round(total, 4),
        }


FEES = FeeStructure()


# =============================================================================
# CN A-share Fee Structure (T+1, lot=100; we accept fractional in paper sim)
# Sources: typical Chinese broker (commission 0.025% double-sided, ¥5 min);
# stamp tax 0.05% sells only (since 2023-08); transfer fee 0.001% double-sided.
# =============================================================================
@dataclass(frozen=True)
class CNFeeStructure:
    commission_rate: float = 0.00025      # 0.025% double-sided
    commission_min: float = 5.0           # ¥5 minimum per order
    stamp_tax_rate: float = 0.0005        # 0.05% sells only
    transfer_fee_rate: float = 0.00001    # 0.001% double-sided (unified 2022+)
    slippage_pct: float = 0.0005          # 0.05% slippage assumption

    def calc_fees(self, shares: float, price: float, is_sell: bool = False) -> dict:
        notional = abs(shares) * price
        commission = max(notional * self.commission_rate, self.commission_min)
        transfer = notional * self.transfer_fee_rate
        stamp = notional * self.stamp_tax_rate if is_sell else 0.0
        slippage = notional * self.slippage_pct
        total = commission + transfer + stamp + slippage
        return {
            "commission": round(commission, 4),
            "transfer": round(transfer, 4),
            "stamp": round(stamp, 4),
            "slippage": round(slippage, 4),
            "total": round(total, 4),
        }


CN_FEES = CNFeeStructure()

# =============================================================================
# Account Configurations - 10 accounts, each $1000 starting capital
# =============================================================================
@dataclass
class AccountConfig:
    name: str
    strategy: str
    starting_capital: float = 10000.0
    max_positions: int = 5
    max_position_pct: float = 0.25
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    rebalance_frequency: str = "daily"
    params: dict = field(default_factory=dict)


ACCOUNTS = [
    AccountConfig(
        name="momentum_fast",
        strategy="momentum",
        max_positions=5,
        stop_loss_pct=0.05,
        params={"lookback_days": 10, "top_n": 5},
    ),
    AccountConfig(
        name="momentum_slow",
        strategy="momentum",
        max_positions=5,
        stop_loss_pct=0.08,
        params={"lookback_days": 30, "top_n": 5},
    ),
    AccountConfig(
        name="mean_reversion",
        strategy="mean_reversion",
        max_positions=5,
        stop_loss_pct=0.05,
        params={"lookback_days": 20, "z_threshold": -1.5},
    ),
    AccountConfig(
        name="value_quality",
        strategy="value",
        max_positions=10,
        max_position_pct=0.15,
        rebalance_frequency="weekly",
        params={"pe_max": 20, "roe_min": 0.15},
    ),
    AccountConfig(
        name="volatility_breakout",
        strategy="volatility_breakout",
        max_positions=3,
        stop_loss_pct=0.04,
        take_profit_pct=0.10,
        params={"atr_multiplier": 2.0, "lookback_days": 14},
    ),
    AccountConfig(
        name="trend_following",
        strategy="trend_following",
        max_positions=5,
        stop_loss_pct=0.07,
        params={"fast_ma": 10, "slow_ma": 50},
    ),
    AccountConfig(
        name="pairs_trading",
        strategy="pairs",
        max_positions=4,
        stop_loss_pct=0.06,
        params={"zscore_entry": 2.0, "zscore_exit": 0.5},
    ),
    AccountConfig(
        name="dividend_growth",
        strategy="dividend",
        max_positions=10,
        max_position_pct=0.15,
        rebalance_frequency="monthly",
        params={"min_yield": 0.02, "min_growth_years": 5},
    ),
    AccountConfig(
        name="sector_rotation",
        strategy="sector_rotation",
        max_positions=5,
        rebalance_frequency="weekly",
        params={"lookback_days": 30, "top_sectors": 3},
    ),
    AccountConfig(
        name="ml_ensemble",
        strategy="ml_ensemble",
        max_positions=5,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
        params={"retrain_days": 30, "features": ["momentum", "volatility", "volume"]},
    ),
]

# =============================================================================
# Report Settings
# =============================================================================
REPORT = {
    "output_dir": "reports",
    "format": "html",
    "daily_summary": True,
    "weekly_summary": True,
    "include_charts": True,
    "benchmark": "SPY",
    "metrics": [
        "total_return", "sharpe_ratio", "max_drawdown", "win_rate",
        "profit_factor", "avg_trade_return", "volatility", "calmar_ratio",
    ],
    "email_enabled": False,
    "email_to": os.environ.get("REPORT_EMAIL", ""),
}

# =============================================================================
# 自适应换仓配置 (Adaptive Rebalancing)
# =============================================================================
# =============================================================================
# Churn controls (applies to Alpha158 / GP / Qlib trading loops)
#   - stop_cooldown_hours: after stop_loss sell, suppress same-ticker re-buys
#   - hold_band_mult:      hysteresis. buy band = top_n; hold band = top_n*mult
#   - min_hold_days:       non-stop-loss sells require position age >= N days
# Each can be set to 0 to disable individually. enabled=False kills all three.
# =============================================================================
CHURN_CONTROLS = {
    "enabled": True,
    "stop_cooldown_hours": 24,
    "hold_band_mult": 3,
    "min_hold_days": 2,
}

ADAPTIVE_REBALANCE = {
    "enabled": True,              # 总开关：True=启用自适应, False=固定周期
    "vol_window": 5,               # 波动率计算窗口（天）
    "ref_window": 60,              # 参考波动率窗口（天，取中位数）
    "min_hours": 12,               # 最短换仓周期（小时）
    "max_hours": 96,               # 最长换仓周期（小时）
    "silence_threshold": 0.3,      # 静默区阈值（σ_t < σ_ref * 此值时暂停交易）
}

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")


# =============================================================================
# Multi-Market Registry (added 2026-04-23)
# Markets coexist in the same SQLite DB; rows in account-scoped tables carry
# a `market` column (default 'US'). Code paths must filter by market.
# =============================================================================
MARKETS = ["US", "CN"]
DEFAULT_MARKET = "US"

# CN universe: loaded from data/cn_universe.json (refreshed by
# scripts/refresh_cn_universe.py). Empty list if the file is missing —
# CN orchestrator will skip until populated.
CN_UNIVERSE: list[str] = []
_cn_universe_path = os.path.join(DATA_DIR, "cn_universe.json")
if os.path.exists(_cn_universe_path):
    try:
        import json as _json
        with open(_cn_universe_path) as _f:
            CN_UNIVERSE = _json.load(_f).get("tickers", [])
    except Exception:
        CN_UNIVERSE = []

UNIVERSES = {
    "US": STOCK_UNIVERSE,
    "CN": CN_UNIVERSE,
}

# Initial cash per market (currency-native; do NOT convert across markets)
INITIAL_CASH = {
    "US": 10000.0,   # USD
    "CN": 100000.0,   # CNY
}

CURRENCY = {
    "US": "USD",
    "CN": "CNY",
}

CURRENCY_SYMBOL = {
    "US": "$",
    "CN": "¥",
}

FEES_BY_MARKET = {
    "US": FEES,
    "CN": CN_FEES,
}

BENCHMARKS_BY_MARKET = {
    "US": [
        {"id": "IDX1", "name": "纳斯达克100指数", "ticker": "QQQ", "group": "IDX"},
        {"id": "IDX2", "name": "标普500指数", "ticker": "SPY", "group": "IDX"},
    ],
    "CN": [
        {"id": "IDX3", "name": "沪深300指数", "ticker": "000300.SH", "group": "IDX"},
    ],
}

# Account ID prefix per market. US keeps existing A01-A10/B01-B10; CN gets
# CA01-CA10/CB01-CB10 to keep the namespaces disjoint.
ACCOUNT_PREFIX = {
    "US": "",
    "CN": "C",
}


# ─── Risk regime / trailing stop ────────────────────────────────────────────
# Trailing stop is OFF by default (verified on 865-trade backtest 2026-05:
# fixed stop has higher mean PnL, trail has higher Sharpe but lower returns).
#
# Three ways to enable:
#   1. Manual : set TRAILING_STOP_PCT = 0.03 (or any value) here
#   2. Auto   : if portfolio drawdown from rolling-30d-peak >= AUTO_ARM_DD_PCT,
#               the system auto-arms TRAILING_STOP_PCT = AUTO_ARM_TRAIL_PCT and
#               sends a Telegram alert. Disarms when drawdown recovers below
#               AUTO_DISARM_DD_PCT for 3 consecutive days.
#   3. Visible: current state is shown in every daily Telegram digest, so even
#               if you forget you turned it on, you'll see it next morning.
#
# Backtest evidence (mean / std / Sharpe per trade, 865 trades):
#   fixed (current):      +1.72% / 5.19% / 5.27
#   trail 3%:             +1.63% / 4.59% / 5.65   ← best Sharpe, lower mean
#   trail 5%:             +1.21% / 4.44% / 4.32
TRAILING_STOP_PCT: float | None = None     # None = disabled
AUTO_ARM_DD_PCT = 0.06       # arm when portfolio drawdown >= 6%
AUTO_ARM_TRAIL_PCT = 0.03    # use 3% trailing stop when armed
AUTO_DISARM_DD_PCT = 0.03    # disarm when drawdown recovers below 3%
AUTO_DISARM_DAYS = 3         # for 3 consecutive days
