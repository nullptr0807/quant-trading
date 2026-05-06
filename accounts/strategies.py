"""Strategy configurations for 10 virtual trading accounts.

Factor names match FactorEngine output from factors/alpha_factors.py:
  KBAR: KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
  Momentum: ROC_5, ROC_10, ROC_20, MA_RATIO_5, MA_RATIO_10, MA_RATIO_20
  Volume: VMOM_5, VMOM_10, VMOM_20, VSTD_5, VSTD_10, VSTD_20
  Volatility: STD_5, STD_10, STD_20, BBPOS_5, BBPOS_10, BBPOS_20
  Mean reversion: RSV, RSI_14
  Trend: BETA_5, BETA_10, BETA_20
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class StrategyConfig:
    id: str
    name: str
    description: str
    strategy_type: str          # maps to SignalGenerator strategy
    factor_names: List[str]     # actual column names from FactorEngine
    top_n: int                  # how many stocks to hold
    rebalance_hours: int
    stop_loss: float            # e.g. 0.03 = -3%
    max_position_pct: float     # e.g. 0.20 = 20%


STRATEGIES: List[StrategyConfig] = [
    StrategyConfig(
        id="A01", name="动量Alpha",
        description="经典动量策略，使用价格变化率和均线比率",
        strategy_type="momentum",
        factor_names=["ROC_5", "ROC_10", "ROC_20", "MA_RATIO_5", "MA_RATIO_10"],
        top_n=5, rebalance_hours=24,
        stop_loss=0.03, max_position_pct=0.20,
    ),
    StrategyConfig(
        id="A02", name="均值回归",
        description="基于RSI和RSV的均值回归",
        strategy_type="mean_reversion",
        factor_names=["RSV", "RSI_14", "BBPOS_5", "BBPOS_10", "BBPOS_20"],
        top_n=5, rebalance_hours=12,
        stop_loss=0.03, max_position_pct=0.20,
    ),
    StrategyConfig(
        id="A03", name="量价策略",
        description="量价背离和放量突破",
        strategy_type="momentum",
        factor_names=["VMOM_5", "VMOM_10", "VSTD_5", "ROC_5", "MA_RATIO_5"],
        top_n=5, rebalance_hours=12,
        stop_loss=0.03, max_position_pct=0.20,
    ),
    StrategyConfig(
        id="A04", name="趋势跟踪",
        description="多时间框架趋势跟踪",
        strategy_type="momentum",
        factor_names=["BETA_5", "BETA_10", "BETA_20", "MA_RATIO_10", "MA_RATIO_20"],
        top_n=3, rebalance_hours=48,
        stop_loss=0.05, max_position_pct=0.30,
    ),
    StrategyConfig(
        id="A05", name="波动率突破",
        description="低波动压缩后的突破",
        strategy_type="volatility",
        factor_names=["STD_5", "STD_10", "STD_20", "BBPOS_5", "BBPOS_10"],
        top_n=5, rebalance_hours=12,
        stop_loss=0.04, max_position_pct=0.20,
    ),
    StrategyConfig(
        id="A06", name="综合多因子",
        description="混合动量、波动率、均值回归因子",
        strategy_type="composite",
        factor_names=[],  # use all factors
        top_n=5, rebalance_hours=24,
        stop_loss=0.03, max_position_pct=0.15,
    ),
    StrategyConfig(
        id="A07", name="短期动量",
        description="5日超短期动量",
        strategy_type="momentum",
        factor_names=["ROC_5", "MA_RATIO_5", "VMOM_5", "KMID"],
        top_n=8, rebalance_hours=6,
        stop_loss=0.02, max_position_pct=0.15,
    ),
    StrategyConfig(
        id="A08", name="价值+动量",
        description="KBAR价值因子结合动量确认",
        strategy_type="momentum",
        factor_names=["KMID", "KLEN", "KSFT", "ROC_10", "ROC_20"],
        top_n=4, rebalance_hours=48,
        stop_loss=0.04, max_position_pct=0.25,
    ),
    StrategyConfig(
        id="A09", name="反转策略",
        description="超卖反转信号",
        strategy_type="mean_reversion",
        factor_names=["RSI_14", "RSV", "KSFT", "KSFT2", "BBPOS_20"],
        top_n=5, rebalance_hours=8,
        stop_loss=0.03, max_position_pct=0.20,
    ),
    StrategyConfig(
        id="A10", name="自适应策略",
        description="根据波动率动态调整因子权重",
        strategy_type="composite",
        factor_names=["ROC_10", "STD_20", "RSI_14", "BETA_10", "VMOM_10"],
        top_n=5, rebalance_hours=24,
        stop_loss=0.03, max_position_pct=0.20,
    ),
]
