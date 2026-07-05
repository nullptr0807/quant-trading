# quant-trading

> 多市场、多策略的纸面交易（paper-trading）研究引擎。它负责拉取真实行情、计算因子、生成信号、模拟成交、保存账本；可视化由 [trading-dashboard](https://github.com/nullptr0807/trading-dashboard) 读取同一个 SQLite 数据库完成。

**这不是真钱交易系统，也不是投资建议。** 它是一个长期运行的量化策略赛马场：让不同 alpha 思路在相同撮合、成本、风控、审计规则下对照生存。

## 当前系统状态

| 维度 | US | CN A-share |
|---|---|---|
| 股票池 | Russell 1000（约 1004 支） | 沪深 300 |
| 初始资金 | `$10,000` / account | `¥100,000` / account |
| 基准 | IDX1 = QQQ, IDX2 = SPY | IDX3 = 沪深300 `000300.SH` |
| 行情 | yfinance + Finnhub realtime fallback | akshare realtime / history |
| 交易时段 | US regular session | A 股 09:30–11:30 / 13:00–15:00 CST |
| 成本 | moomoo AU US-stock fee model | 佣金 + 过户费 + 卖出印花税 + 滑点 |
| 交易单位 | whole shares | **买入必须 100 股整数倍** |

CN 账户是 US 策略族的镜像：`A01 → CA01`, `B11 → CB11`, `Q08 → CQ08`。这样能比较同一类信号在两个市场结构里的迁移能力。

## 账户族

| 族 | 账户 | 来源 | 用途 |
|---|---|---|---|
| **A / CA** | A01–A10, CA01–CA10 | Alpha158 风格手写因子 | 人类可解释的经典因子基线 |
| **B / CB** | B01–B16, CB01–CB16 | gplearn GP 因子挖掘 | 机器自动搜索表达式，做 alpha 探索 |
| **F / CF** | F11–F16, CF11–CF16 | FactorMiner-style GP | 带记忆/相关性筛选的第二套 GP 实验 |
| **Q / CQ** | Q01–Q10, CQ01–CQ10 | Qlib ML 模型 | LightGBM/XGBoost/CatBoost/Ridge/MLP/LSTM/GRU/Transformer/TCN/ALSTM |
| **IDX** | IDX1–IDX3 | Buy-and-hold benchmark | 判断策略是否跑赢指数 |

账户生命周期写入 `account_meta.status`：active 账户继续交易；retired 账户冻结交易但保留完整历史，供 dashboard 墓碑墙和研究复盘使用。

## 核心交易规则

- **Long-only**：所有账户只做多，不做空。
- **现金约束**：买入必须有足够现金支付成交价 + 费用。
- **单票仓位上限**：每个策略有自己的 `max_position_pct`。
- **调仓节奏**：每个账户有 `rebalance_hours`，可叠加自适应静默区。
- **止损 / trailing stop**：盘中 `scripts/update_prices.py` 可执行保护性卖出。
- **A 股 100 股一手**：CN 买入会向下取整到 100 股整数倍；如果 `budget` 连 1 手都买不起，就跳过该候选并记录事件。
- **legacy odd-lot 可卖出**：早期系统已有非 100 股历史持仓，卖出不强制 round，避免旧仓永远清不掉。

### A 股高价股处理

高价股（如茅台、寒武纪）信号再好，如果 `100股 × price` 超过账户当前预算/仓位上限，系统不会破坏风控去硬买一手，而是：

1. 记录 `events`：`⏭️ Skip <ticker>: 1手买不起/超仓位`
2. 在同一 ranked signal list 里继续向后找可执行候选
3. 尽量填满该策略目标持仓数 `top_n`

事件 detail 会保存：预算、价格、滑点后执行价、1 手名义金额、rank、score、strategy_kind。

## 数据与写入路径

单一数据库：`data/trading.db`。

| 表 | 内容 |
|---|---|
| `prices` | OHLCV cache，按 `(ticker, datetime, interval)` 存储；dashboard 也复用这一份价格源 |
| `factor_values` | Alpha158 / GP / Qlib scores |
| `account_meta` | 账户元数据、初始资金、市场、active/retired 状态 |
| `account_state` | 当前 cash / initial_cash |
| `positions` | 当前持仓、avg_cost、current_price |
| `positions_history` | 持仓时间序列快照 |
| `accounts` | equity curve 快照 |
| `trades` | 模拟成交流水 |
| `events` | trade / rebalance / system / lifecycle 事件流 |
| `adaptive_state` | 自适应调仓状态 |

价格写入保留原始行情；脏 0-volume intraday bar 等清洗在读取层处理，以保留审计能力。

## 关键脚本

| 脚本 | 作用 |
|---|---|
| `main.py --once` | 唯一推荐的手动交易 cycle 入口 |
| `scripts/update_prices.py` | 盘中实时 quote、equity snapshot、stop-loss watchdog |
| `scripts/update_prices.sh` | cron wrapper，带全局 lock 和 hard timeout，防 provider 卡死持锁 |
| `scripts/backfill_prices.py` | US/CN price cache 增量/全量回填，包含当前持仓 legacy tickers |
| `scripts/ledger_watchdog.py` | cash/positions/trades/equity/events 对账 |
| `scripts/qlib_retrain.py` | Q 组模型 rolling retrain |
| `scripts/retire_account.py` | 账户退役 / 复活 |
| `scripts/replay_us.py`, `scripts/replay_cn.py` | 历史 replay / 修复辅助 |

## 常用命令

```bash
cd ~/quant-trading
source venv/bin/activate

# 单次运行（会按市场时段 gate；不要绕过 main.py）
python main.py --once
python main.py --once --market CN

# 价格回填 / 增量刷新
python -m scripts.backfill_prices --market US --interval 1h --days 3
python -m scripts.backfill_prices --market US --interval 1d --days 5
python -m scripts.backfill_prices --market CN --interval 1d --days 5

# 盘中价格刷新 smoke test（周末/闭市手动修复 stale snapshot 时）
QUANT_FORCE_PRICE_UPDATE=1 python -m scripts.update_prices

# 当前账本对账：quiet-ok 无输出且 exit 0 表示通过
PYTHONPATH=$PWD python scripts/ledger_watchdog.py --market US --history-days 0 --quiet-ok
PYTHONPATH=$PWD python scripts/ledger_watchdog.py --market CN --history-days 0 --quiet-ok

# CN board-lot sizing regression tests
PYTHONPATH=$PWD pytest tests/test_cn_lot_size.py -q
```

> 不要用 `python -c "from main import ..."` 直接调用 trading cycle。那会绕过市场时段、状态恢复、报告/快照等保护逻辑。

## Cron 运行方式

当前设计是统一脚本 + market-aware gate，而不是每个市场复制一套脚本：

- `run_cycle.sh` / `run_cycle_quiet.sh`：交易 cycle，和 update_prices 共用 `/tmp/quant_run_cycle.lock`
- `update_prices.sh`：实时价格和止损，外层 `timeout` 防止 akshare/yfinance 挂住后长期持锁
- `qlib_retrain_daily.sh`：每日 Qlib retrain
- Hermes cron job：收盘后 price incremental refresh + ledger watchdog

## 对账与审计哲学

这个项目优先保留可追溯性：

- 不轻易删除 `trades` / `accounts` / `events`
- 需要修复时先备份 DB 或建 archive table
- 先修 current ledger（cash + positions + active negative cash），再评价策略表现
- `history_curve` mismatch 要区分真实账本污染 vs 缺少 intraday price coverage 的 false positive
- dashboard 展示依赖 `accounts` 历史曲线，所以修复 state 后也要检查曲线 scar

`ledger_watchdog.py` 的当前权威级别：

1. `cash_replay` / `positions_replay` / oversell / negative cash：真实账本问题
2. latest `equity_snapshot`：当前价格/快照一致性
3. `history_curve`：需要足够细粒度价格覆盖，否则降噪/跳过
4. `trade_events`：审计展示完整性，不等于账本真相

## 与 trading-dashboard 的关系

`trading-dashboard` 是只读可视化层：

```
quant-trading writes data/trading.db
        ↓
trading-dashboard reads the same DB and renders UI
```

dashboard 不应该复制价格 cache，也不应该写账户状态。两边通过同一个 `prices` / `accounts` / `events` 数据源保持一致。

## 安装说明（简版）

```bash
git clone https://github.com/nullptr0807/quant-trading.git
cd quant-trading
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # 如需 Telegram / Finnhub
python main.py --once
```

主要依赖：`pandas`, `numpy`, `yfinance`, `finnhub-python`, `akshare`, `gplearn`, `scikit-learn`, `qlib`, `lightgbm`, `xgboost`, `catboost`, `python-dotenv`。

## 免责声明

MIT。仅供研究和学习。所有交易都是模拟成交；任何基于本项目做出的真实投资决策，风险自担。

---

🤖 Built and maintained with [Hermes Agent](https://hermes-agent.nousresearch.com).
