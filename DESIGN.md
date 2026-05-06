# 美股量化交易系统 设计文档

> 最后更新: 2026-04-16

## 系统概述

美股量化模拟交易系统，采用 Alpha158 风格因子 + GP(遗传编程)因子挖掘双引擎架构。
20个虚拟账户并行交易 (A01-A10 手工因子策略 + B01-B10 GP自动挖掘策略)，
通过 Telegram 每小时报告交易状态。

## 架构

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────┐
│  数据获取    │───▶│  因子计算         │───▶│  信号生成     │
│  yfinance   │    │  Alpha158 (32因子)│    │  排序+过滤    │
│  finnhub    │    │  GP挖掘 (gplearn) │    │              │
└─────────────┘    └──────────────────┘    └──────┬───────┘
                                                  │
┌─────────────┐    ┌──────────────┐    ┌──────────▼───────┐
│  Telegram   │◀───│  报告系统     │◀───│  交易引擎         │
│  Bot报告     │    │  每小时汇总   │    │  20个虚拟账户     │
└─────────────┘    └──────────────┘    │  (A01-A10+B01-B10)│
                                       └──────────────────┘
```

## 模块设计

### 1. 数据获取 (data/)

- **历史数据**: yfinance — OHLCV日线，用于因子计算 (30天) 和 GP 挖掘 (120天)
- **实时报价**: finnhub-python — 实时/盘前盘后行情
- **标的池**: S&P 500 流动性前50只 (config/settings.py STOCK_UNIVERSE)
- **存储**: SQLite (data/trading.db) — prices/trades/accounts/positions 四表

#### DataFetcher API
- `get_historical(tickers, days)` → 合并 DataFrame，含 `ticker` 列 (小写 OHLCV)
- `get_realtime_quotes(tickers)` → list[dict]
- `get_extended_hours_quote(ticker)` → dict

### 2. 因子引擎 (factors/)

#### 2a. Alpha158 手工因子 (factors/alpha_factors.py)

FactorEngine 使用纯 pandas/numpy 计算 **32个** Alpha158 风格因子:

| 类别 | 数量 | 因子 | 公式 |
|------|------|------|------|
| KBAR | 9 | KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2 | K线形态比率 |
| 动量 | 6 | ROC_5/10/20, MA_RATIO_5/10/20 | Close(t)/Close(t-w)-1, Close/SMA(Close,w) |
| 量价 | 6 | VMOM_5/10/20, VSTD_5/10/20 | Volume/SMA(Vol,w), Std(Vol,w)/SMA(Vol,w) |
| 波动率 | 6 | STD_5/10/20, BBPOS_5/10/20 | Std(Close,w)/Close, 布林带位置 |
| 均值回归 | 2 | RSV, RSI_14 | 随机指标, 相对强弱指标 |
| 趋势 | 3 | BETA_5/10/20 | OLS_Slope(Close,w)/SMA(Close,w) |

#### 2b. GP遗传编程因子挖掘 (factors/gp_miner.py)

使用 gplearn 库通过遗传编程自动挖掘 alpha 因子。

**GP 输入特征 (13个):**

| 变量 | 含义 |
|------|------|
| o_c | (Open-Close)/Close |
| h_c | (High-Close)/Close |
| l_c | (Low-Close)/Close |
| v_vma20 | Volume/SMA(Volume,20) |
| ma_5/10/20 | SMA(Close,w)/Close |
| std_5/10/20 | Std(Close,w)/Close |
| ret_1/5/10 | Close(t)/Close(t-w)-1 |

**GP 函数集:** add, sub, mul, div(protected), sqrt_abs, log_abs1, neg, inv(protected), max2, min2

**挖掘流程:**
1. 获取120天历史数据
2. 每个B账户独立挖掘 (不同seed/种群/代数/简约系数)
3. 适应度函数: Spearman rank IC (因子值 vs 未来5日收益)
4. 多轮运行取最优，去重后保存
5. 结果缓存到 `factors/mined_alphas_per_account.json`

**GP 表达式转数学公式:** main.py 内置 `gp_expr_to_math()` 递归解析器，
将 gplearn 表达式 (如 `max2(X11, log_abs1(X10))`) 转为
严格数学符号 (如 `max(Close(t)/Close(t-5)-1, ln(|Close(t)/Close(t-1)-1|+1))`)

#### 2c. 信号生成

- **手工因子信号** (factors/signal.py): SignalGenerator
  - 4种策略类型: momentum, mean_reversion, volatility, composite
  - 跨截面百分位排名 → 等权平均 → Top N 买入 / Bottom N 卖出
  - mean_reversion 反转排名 (低 RSI = 买入)

- **GP因子信号** (factors/gp_signal.py): GPSignalGenerator
  - 跨截面百分位排名 → 等权平均
  - 支持 factor_selection 过滤: all / top5 / top10 / bottom5
  - 支持 scoring_method: equal_weight / ic_weighted / top3_only

### 3. 交易引擎 (trading/)

- **虚拟账户** (account.py): VirtualAccount — 现金+仓位+成本基准+交易日志
- **费用计算** (costs.py): MoomooAUCosts — moomoo Australia 费率
- **执行引擎** (engine.py): TradingEngine — 风控+多账户管理

**moomoo Australia 费率 (美股):**

| 费用项 | 金额 | 适用 |
|--------|------|------|
| 佣金 | $0 | 买卖 |
| 平台费 | $0.99/order | 买卖 |
| 结算费 | $0.003/share (上限 1% 交易额) | 买卖 |
| SEC费 | $0.0000206 × 交易额 (最低 $0.01) | 仅卖出 |
| FINRA TAF | $0.000195/share ($0.01-$9.79) | 仅卖出 |
| 滑点 | 0.05% | 买卖 |

**风控:**
- 单只股票最大仓位: 由策略配置 (15%-30%)
- 止损: 由策略配置 (2%-5%)

### 4. 账户策略

#### A组: 手工因子策略 (accounts/strategies.py)

每个账户初始资金 $10,000。

| 账户 | 策略 | 类型 | 因子 | 持仓 | 换仓 | 止损 | 最大仓位 |
|------|------|------|------|------|------|------|----------|
| A01 | 动量Alpha | momentum | ROC_5/10/20, MA_RATIO_5/10 | 5 | 24h | 3% | 20% |
| A02 | 均值回归 | mean_reversion | RSV, RSI_14, BBPOS_5/10/20 | 5 | 12h | 3% | 20% |
| A03 | 量价策略 | momentum | VMOM_5/10, VSTD_5, ROC_5, MA_RATIO_5 | 5 | 12h | 3% | 20% |
| A04 | 趋势跟踪 | momentum | BETA_5/10/20, MA_RATIO_10/20 | 3 | 48h | 5% | 30% |
| A05 | 波动率突破 | volatility | STD_5/10/20, BBPOS_5/10 | 5 | 12h | 4% | 20% |
| A06 | 综合多因子 | composite | 全部32因子 | 5 | 24h | 3% | 15% |
| A07 | 短期动量 | momentum | ROC_5, MA_RATIO_5, VMOM_5, KMID | 8 | 6h | 2% | 15% |
| A08 | 价值+动量 | momentum | KMID, KLEN, KSFT, ROC_10/20 | 4 | 48h | 4% | 25% |
| A09 | 反转策略 | mean_reversion | RSI_14, RSV, KSFT, KSFT2, BBPOS_20 | 5 | 8h | 3% | 20% |
| A10 | 自适应策略 | composite | ROC_10, STD_20, RSI_14, BETA_10, VMOM_10 | 5 | 24h | 3% | 20% |

#### B组: GP遗传编程策略 (accounts/gp_strategies.py)

每个账户独立挖掘 GP 因子 (不同种子/参数)，初始资金 $10,000。

| 账户 | 名称 | 因子选择 | 打分方式 | 持仓 | 换仓 | 止损 | GP种子 | 种群 | 代数 | 运行次数 | 简约系数 | 挖掘因子数 |
|------|------|----------|----------|------|------|------|--------|------|------|----------|----------|-----------|
| B01 | 基因突变 | all | equal_weight | 5 | 4h | 2% | 42 | 300 | 20 | 5 | 0.01 | 20 |
| B02 | 物竞天择 | top5 | ic_weighted | 8 | 24h | 4% | 137 | 500 | 25 | 3 | 0.005 | 15 |
| B03 | 适者生存 | top10 | top3_only | 3 | 8h | 3% | 256 | 200 | 30 | 7 | 0.02 | 25 |
| B04 | 遗传漂变 | all | ic_weighted | 6 | 48h | 5% | 389 | 400 | 15 | 4 | 0.008 | 18 |
| B05 | 双螺旋 | bottom5 | equal_weight | 4 | 6h | 2% | 512 | 350 | 22 | 6 | 0.015 | 20 |
| B06 | 寒武纪爆发 | top10 | equal_weight | 10 | 12h | 3% | 666 | 600 | 18 | 3 | 0.003 | 30 |
| B07 | 自然选择 | top5 | top3_only | 5 | 16h | 3% | 777 | 250 | 25 | 5 | 0.012 | 15 |
| B08 | 染色体交叉 | all | top3_only | 7 | 36h | 5% | 888 | 450 | 20 | 4 | 0.007 | 22 |
| B09 | 种群进化 | bottom5 | ic_weighted | 3 | 8h | 2% | 999 | 300 | 28 | 6 | 0.018 | 18 |
| B10 | 达尔文密码 | top10 | ic_weighted | 9 | 24h | 4% | 1024 | 550 | 22 | 4 | 0.005 | 25 |

### 5. 报告系统 (reports/)

- **频率**: 工作日 08:00-00:00 UTC (cron 每小时)
- **静默窗口**: 01:00-08:00 UTC 不发送 (可用 force=True 覆盖)
- **排序**: 按账户总盈亏排序
- **内容**:
  - 各账户盈利排名 (含 GP 因子的数学公式)
  - 每个账户: 策略名/因子/当前持仓/仓位占比/盈亏
  - 权益曲线图表 (matplotlib)
- **发送**: Telegram Bot API (urllib, 自动分片 >4096字符)
- **目标**: 由环境变量 `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` 配置

### 6. 调度 (scripts/ + cron)

**run_cycle.sh**: 激活venv → 检查工作日+市场时间 → `python main.py --once`

**Cron 配置:**
```
0 8-23 * * 1-5   run_cycle.sh   # 工作日 08:00-23:00 UTC 每小时
0 0 * * 1-5      run_cycle.sh   # 工作日 00:00 UTC (盘后收尾)
```

**main.py 执行流程:**
1. 获取30天历史数据 (yfinance)
2. 计算32个 Alpha158 因子
3. GP因子挖掘 (首次120天数据, 之后从缓存加载)
4. 计算 GP 因子值 (当前30天数据)
5. A01-A10: 手工因子 → 信号 → 交易
6. B01-B10: GP因子 → 信号 → 交易
7. 保存快照到 SQLite
8. 生成报告 (文字+图表) → 发送 Telegram

## 交易时间 (ET → UTC)

- 盘前: 04:00-09:30 ET = 08:00-13:30 UTC
- 正常: 09:30-16:00 ET = 13:30-20:00 UTC
- 盘后: 16:00-20:00 ET = 20:00-00:00 UTC

## 技术栈

- Python 3.12
- **因子计算**: pandas, numpy (Alpha158 手工因子)
- **GP挖掘**: gplearn (遗传编程符号回归)
- **历史数据**: yfinance
- **实时数据**: finnhub-python
- **图表**: matplotlib
- **存储**: sqlite3
- **调度**: cron + bash
- **报告**: Telegram Bot API (urllib, 无第三方依赖)

## 文件结构

```
quant-trading/
├── DESIGN.md                          # 本文档
├── main.py                            # 主调度器 (QuantSystem)
├── config/
│   └── settings.py                    # 全局配置 (标的池/费率/账户)
├── data/
│   ├── fetcher.py                     # 数据获取 (yfinance + finnhub)
│   ├── store.py                       # SQLite 存储
│   └── trading.db                     # 数据库文件
├── factors/
│   ├── alpha_factors.py               # Alpha158 因子引擎 (32因子)
│   ├── signal.py                      # 手工因子信号生成
│   ├── gp_miner.py                    # GP 因子挖掘 (gplearn)
│   ├── gp_signal.py                   # GP 因子信号生成
│   └── mined_alphas_per_account.json  # GP 挖掘结果缓存
├── trading/
│   ├── engine.py                      # 交易引擎 (风控+执行)
│   ├── costs.py                       # moomoo AU 费用计算
│   └── account.py                     # 虚拟账户 (仓位+成本基准)
├── accounts/
│   ├── strategies.py                  # A01-A10 策略配置
│   └── gp_strategies.py              # B01-B10 GP策略配置
├── reports/
│   ├── generator.py                   # 报告生成 (文字+图表)
│   └── telegram.py                    # Telegram 发送
├── scripts/
│   └── run_cycle.sh                   # Cron 入口脚本
├── logs/
│   ├── trading.log                    # 交易日志
│   └── cron.log                       # Cron 执行日志
└── venv/                              # Python 虚拟环境
```
