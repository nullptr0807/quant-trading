# quant-trading

> 多策略、多市场（US/CN）的纸面交易（paper-trading）量化研究平台。
> 30+ 个虚拟账户在真实行情数据上并行 7×24 跑策略，cron 驱动，结果通过 [trading-dashboard](https://github.com/nullptr0807/trading-dashboard) 可视化。

**这不是真钱交易系统，也不是投资建议。** 这是一个把因子研究、ML 模型、遗传编程因子挖掘三种范式放在一起做对照实验的赛马平台。

## 1. 系统概述

把 **3 种 alpha 范式** × **2 个市场** × **N 个超参数** 同时跑成独立账户，用同一套交易引擎、同一份手续费模型、同一份风控规则，让真实的市场行情来决定哪个策略能活下来。

### 三大类策略

| 组别 | 别名 | 策略来源 | 账户 (US / CN) |
|---|---|---|---|
| **A 组** | Alpha158 手写因子 | 微软 Qlib Alpha158 套件，每个账户用一个经典因子组合 | A01–A10 / CA01–CA10 |
| **B 组** | gplearn GP 进化 | 遗传编程自动挖掘的 alpha 表达式（每账户独立超参） | B01–B16 / CB01–CB16 |
| **Q 组** | Qlib ML 模型 | LightGBM / XGB / CatBoost / Ridge / MLP / LSTM / GRU / Transformer / TCN / ALSTM | Q01–Q10 / CQ01–CQ10 |
| **IDX** | 基准指数 | QQQ、SPY (US)；沪深300 (CN) — 不是策略，是对标 | IDX1, IDX2 / IDX3 |

### 两大市场

| | US | CN |
|---|---|---|
| 标的池 | Russell 1000 (~1004 支) | 沪深 300 |
| 初始资金/账户 | $10,000 | ¥100,000 |
| 行情 | yfinance + Finnhub (实时) | akshare |
| 费率 | moomoo AU 实际费率 | 万 2.5 + 印花税 |
| 交易时段 | 美东 9:30–16:00 | 京 9:30–11:30 / 13:00–15:00 |
| 账户前缀 | A/B/Q | C+A/B/Q（镜像） |

CN 账户是 US 账户的**镜像**：跨市场对照同一策略思路在 T+1 vs T+0、涨跌停 vs 自由波动、散户 vs 机构主导市场上的表现差异。

## 2. 架构

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────┐
│  数据获取    │───▶│  因子计算         │───▶│  信号生成     │
│  yfinance   │    │  Alpha158 (32+)  │    │  排序 + 过滤  │
│  finnhub    │    │  GP (gplearn)    │    │  + 风控       │
│  akshare    │    │  Qlib ML 模型     │    │              │
└─────────────┘    └──────────────────┘    └──────┬───────┘
                                                  │
┌─────────────┐    ┌──────────────┐    ┌──────────▼───────┐
│  Telegram   │◀───│  报告系统     │◀───│  交易引擎         │
│  Bot 报告    │    │  每小时汇总   │    │  30+ 虚拟账户     │
└─────────────┘    └──────────────┘    │  纯多头模拟撮合   │
                          │             └────────┬─────────┘
                          ▼                      │
                   ┌──────────────┐              │
                   │ trading.db   │◀─────────────┘
                   │ (SQLite)     │
                   └──────┬───────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ trading-dashboard    │
                │ (FastAPI + vanilla JS)│
                └──────────────────────┘
```

## 3. 模块布局

```
quant-trading/
├── main.py                    # 主调度器：fetch → factor → signal → trade → report
├── config/settings.py         # 账户定义、universe、费率、调度参数
├── data/
│   ├── fetcher.py             # US (yfinance + finnhub)
│   ├── cn_fetcher.py          # CN (akshare)，统一 get_fetcher_for(market)
│   └── store.py               # SQLite schema + DataStore CRUD
├── factors/
│   ├── alpha_factors.py       # Alpha158 风格手写因子
│   ├── gp_miner.py            # gplearn GP 因子挖掘
│   ├── gp_signal.py           # GP 表达式打分
│   ├── qlib_signal.py         # Qlib 模型推理 (Q 组)
│   └── intraday_factors.py    # 日内因子（小时级）
├── accounts/
│   ├── strategies.py          # A 组 Alpha158 策略
│   ├── gp_strategies.py       # B 组 GP 策略
│   └── qlib_strategies.py     # Q 组 ML 策略
├── trading/
│   ├── engine.py              # 撮合 + 仓位管理
│   ├── account.py             # 账户状态机
│   ├── costs.py               # 手续费 / 印花税 / 滑点
│   ├── vol_stop.py            # 波动率止损
│   └── risk_regime.py         # 市场状态识别
├── reports/
│   ├── generator.py           # 每小时报告
│   └── telegram.py            # Telegram bot 推送
└── scripts/                   # backfill / retrain / replay 等运维脚本
```

## 4. 调度流程（典型一日）

```
盘前
├── 价格数据更新 (1d / 1h K 线)
├── 因子重算 (Alpha158 全量 + GP 表达式 + Qlib 推理)
└── 信号生成 (各账户根据自家因子打分)

盘中（每 15-30 分钟一次 cron tick）
├── 实时报价拉取 (yfinance fast_info / akshare 1m)
├── 信号变化、止盈止损触发检测
├── 模拟撮合下单 (按经纪商费率算成本)
└── mark-to-market

盘后
├── 当日 equity 快照 → accounts 表
├── 持仓快照 → snapshots 表
└── 23:00 UTC：Qlib 模型 rolling retrain (10 个 Q 账户)

实时
└── lifecycle / 风控 / 错误事件 → events 表
```

## 5. 安装与运行

### 前置

- Python **3.11+**（Qlib 训练用 3.10 也行）
- Linux/macOS（Windows 没测过）
- ~5GB 磁盘（历史价格 + Qlib 模型 checkpoint）

### 安装

```bash
git clone https://github.com/nullptr0807/quant-trading.git
cd quant-trading
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt    # 自行 pip freeze 生成；核心依赖见下
```

**核心依赖**（手动 pip install 也可）：
```
yfinance pandas numpy scipy scikit-learn
gplearn akshare finnhub-python
pyqlib lightgbm xgboost catboost
torch  # Qlib 神经网模型用
python-dotenv
```

### 配置

复制 `.env.example` → `.env` 并填写：

```bash
TELEGRAM_BOT_TOKEN=<你的 bot token，可选>
TELEGRAM_CHAT_ID=<聊天 ID，可选>
FINNHUB_API_KEY=<finnhub key，可选 — 没有就只用 yfinance>
QUANT_DISABLE_TELEGRAM=0           # 1 = 完全禁用 Telegram 推送
```

### 单次运行

```bash
source venv/bin/activate
python main.py --once              # 跑一个完整 cycle 然后退出
python main.py --once --market CN  # 只跑 CN 市场
```

> ⚠️ 不要用 `python -c "from main import run_*_trading_cycle"` —— 会绕过 `is_market_hours_for` 时段判断，可能写出不该有的盘外交易。务必通过 `main.py --once` 入口。

### 持续运行（cron）

参考 `scripts/run_cycle.sh`，在 crontab 里：

```cron
*/15 * * * *  cd ~/quant-trading && ./scripts/run_cycle.sh >> logs/cron.log 2>&1
0 23 * * *    cd ~/quant-trading && ./scripts/qlib_retrain_daily.sh >> logs/retrain.log 2>&1
```

### 数据回填

首次部署，需要先回填历史价格：

```bash
python scripts/backfill_us_history.py   # Russell 1000 × 400 天日线 + 90 天小时
python scripts/backfill_cn_history.py   # 沪深 300 × 同上
```

### 退役 / 复活账户

```bash
python scripts/retire_account.py B07 --reason "short-momentum cluster (kept B05)"
python scripts/retire_account.py B07 --unretire
```

## 6. 数据库 schema (SQLite)

`data/trading.db`，被 dashboard 只读访问：

| 表 | 内容 |
|---|---|
| `prices` | OHLCV 1d/1h，PK = (ticker, timestamp, freq, market) |
| `factor_values` | 每天每标的的因子值（含 `qlib_QXX_score` ML 预测列） |
| `accounts` | 账户每日 equity 快照（一行 = 一个账户在一个时刻的现金/持仓总值） |
| `snapshots` | 持仓快照（dashboard hover tooltip 用） |
| `trades` | 所有模拟成交记录 |
| `account_meta` | 账户元数据：策略名、初始资金、status (active/retired)、retire_reason |
| `events` | lifecycle / 风控 / 错误事件 流水 |

## 7. 与 dashboard 的关系

[**trading-dashboard**](https://github.com/nullptr0807/trading-dashboard) 是这个系统的**只读可视化前端**：

- 数据流：`quant-trading` 写 `trading.db` → dashboard 通过 `core/price_cache.py` 适配器读取
- 单一数据源：dashboard **不重复存储**任何价格/账户数据，只查询
- 实时 LiveStream：dashboard 通过轮询 `events` / `accounts` 表展示账户生命周期、交易、风控

## 8. 设计原则

1. **数据获取与策略解耦** — `data/` 模块不依赖 `accounts/` / `trading/` / `factors/`，单一 fetcher 抽象 (`get_fetcher_for(market)`)
2. **写入器隔离** — main loop 与 dashboard 永不并发写 `trading.db`（dashboard 只读）
3. **价格数据原始保留** — DB 里的 prices 行为下游审计提供原貌；过滤 0 成交量等清洗在 read-time 做（`skip_zero_volume=True`）
4. **无 look-ahead bias** — 所有信号严格基于 t-1 数据，回放可复现
5. **失败要可见** — 任何 lifecycle 变化（账户退役、风控触发、训练失败）必须落 `events` 表，dashboard 立刻可见

## 9. 文档

- [`DESIGN.md`](DESIGN.md) — 详细设计文档（中文）
- [`docs/plans/`](docs/plans) — 历次大改造的方案文档（CN 市场接入、Qlib 集成等）

## 10. License

MIT — 仅供研究和学习。**任何亏损与作者无关，使用者自行承担。**

---

🤖 _Built and maintained with [Hermes Agent](https://hermes-agent.nousresearch.com)._
