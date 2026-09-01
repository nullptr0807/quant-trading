from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class GPStrategyConfig:
    id: str
    name: str
    description: str
    top_n: int
    rebalance_hours: int
    stop_loss: float
    max_position_pct: float
    factor_selection: str
    scoring_method: str
    gp_seed: int = 42
    gp_population: int = 300
    gp_generations: int = 20
    gp_n_runs: int = 5
    gp_parsimony: float = 0.01
    gp_n_factors: int = 20
    # --- C+D diversification (B11+) ---
    # y target: next_1d_ret | next_3d_ret | next_5d_ret | next_5d_sharpe | next_5d_minret_neg | reversal_2d
    gp_y_target: str = "next_1d_ret"
    # Feature subset: tuple of feature names from FEATURE_COLS, or None = all 13 features
    gp_feature_subset: Optional[Tuple[str, ...]] = None
    # Correlation threshold for dedup (lower = keep more diverse near-siblings)
    gp_dedup_threshold: float = 0.85
    # Account-family metadata. B = legacy gplearn GP; F = FactorMiner-style GP.
    family: str = "B"
    mining_backend: str = "gplearn"
    enabled_markets: Tuple[str, ...] = ("US", "CN")
    # FactorMiner-style global family correlation / replacement controls.
    gp_global_corr_threshold: float = 0.6
    gp_replacement_ic_mult: float = 1.3
    # A paper clone may reuse another account's mined factors/ranking. Portfolio
    # cash, positions, fills, cooldowns and rebalance state remain account-local.
    signal_source_id: Optional[str] = None


GP_STRATEGIES: List[GPStrategyConfig] = [
    GPStrategyConfig(
        id="B01", name="基因突变", description="高频再平衡，全因子等权打分，激进止损",
        top_n=5, rebalance_hours=4, stop_loss=0.02, max_position_pct=0.20,
        factor_selection="all", scoring_method="equal_weight",
        gp_seed=42, gp_population=300, gp_generations=20, gp_n_runs=5, gp_parsimony=0.01, gp_n_factors=20,
    ),
    GPStrategyConfig(
        id="B02", name="物竞天择", description="IC加权Top5因子，中频宽止损",
        top_n=8, rebalance_hours=24, stop_loss=0.04, max_position_pct=0.15,
        factor_selection="top5", scoring_method="ic_weighted",
        gp_seed=137, gp_population=500, gp_generations=25, gp_n_runs=3, gp_parsimony=0.005, gp_n_factors=15,
    ),
    GPStrategyConfig(
        id="B03", name="适者生存", description="集中持仓，仅用Top3因子打分",
        top_n=3, rebalance_hours=8, stop_loss=0.03, max_position_pct=0.30,
        factor_selection="top10", scoring_method="top3_only",
        gp_seed=256, gp_population=200, gp_generations=30, gp_n_runs=7, gp_parsimony=0.02, gp_n_factors=25,
    ),
    GPStrategyConfig(
        id="B04", name="遗传漂变", description="低频大仓位，全因子IC加权",
        top_n=6, rebalance_hours=48, stop_loss=0.05, max_position_pct=0.25,
        factor_selection="all", scoring_method="ic_weighted",
        gp_seed=389, gp_population=400, gp_generations=15, gp_n_runs=4, gp_parsimony=0.008, gp_n_factors=18,
    ),
    GPStrategyConfig(
        id="B05", name="双螺旋", description="反向因子等权，高频紧止损",
        top_n=4, rebalance_hours=6, stop_loss=0.02, max_position_pct=0.25,
        factor_selection="bottom5", scoring_method="equal_weight",
        gp_seed=512, gp_population=350, gp_generations=22, gp_n_runs=6, gp_parsimony=0.015, gp_n_factors=20,
    ),
    GPStrategyConfig(
        id="B06", name="寒武纪爆发", description="广撒网分散持仓，Top10因子等权",
        top_n=10, rebalance_hours=12, stop_loss=0.03, max_position_pct=0.15,
        factor_selection="top10", scoring_method="equal_weight",
        gp_seed=666, gp_population=600, gp_generations=18, gp_n_runs=3, gp_parsimony=0.003, gp_n_factors=30,
    ),
    GPStrategyConfig(
        id="B07", name="自然选择", description="Top5因子仅取前三，中频中仓",
        top_n=5, rebalance_hours=16, stop_loss=0.03, max_position_pct=0.20,
        factor_selection="top5", scoring_method="top3_only",
        gp_seed=777, gp_population=250, gp_generations=25, gp_n_runs=5, gp_parsimony=0.012, gp_n_factors=15,
    ),
    GPStrategyConfig(
        id="B08", name="染色体交叉", description="全因子Top3打分，低频宽止损",
        top_n=7, rebalance_hours=36, stop_loss=0.05, max_position_pct=0.20,
        factor_selection="all", scoring_method="top3_only",
        gp_seed=888, gp_population=450, gp_generations=20, gp_n_runs=4, gp_parsimony=0.007, gp_n_factors=22,
    ),
    GPStrategyConfig(
        id="B09", name="种群进化", description="反向因子IC加权，集中持仓紧止损",
        top_n=3, rebalance_hours=8, stop_loss=0.02, max_position_pct=0.30,
        factor_selection="bottom5", scoring_method="ic_weighted",
        gp_seed=999, gp_population=300, gp_generations=28, gp_n_runs=6, gp_parsimony=0.018, gp_n_factors=18,
    ),
    GPStrategyConfig(
        id="B10", name="达尔文密码", description="Top10因子IC加权，分散低频",
        top_n=9, rebalance_hours=24, stop_loss=0.04, max_position_pct=0.15,
        factor_selection="top10", scoring_method="ic_weighted",
        gp_seed=1024, gp_population=550, gp_generations=22, gp_n_runs=4, gp_parsimony=0.005, gp_n_factors=25,
    ),
    # ---- B11-B16: C+D 组合多样化 (每账户独特 y + 特征子集) ----
    GPStrategyConfig(
        id="B11", name="短打猎手", description="纯短动量子集挖掘1日收益",
        top_n=5, rebalance_hours=4, stop_loss=0.025, max_position_pct=0.20,
        factor_selection="all", scoring_method="ic_weighted",
        gp_seed=1111, gp_population=400, gp_generations=22, gp_n_runs=5, gp_parsimony=0.005, gp_n_factors=20,
        gp_y_target="next_1d_ret",
        gp_feature_subset=("ret_1", "ret_5", "ret_10"),
        gp_dedup_threshold=0.6,
    ),
    GPStrategyConfig(
        id="B12", name="周度趋势", description="均线+日内位置子集挖掘5日收益",
        top_n=6, rebalance_hours=24, stop_loss=0.04, max_position_pct=0.20,
        factor_selection="all", scoring_method="ic_weighted",
        gp_seed=1212, gp_population=450, gp_generations=25, gp_n_runs=5, gp_parsimony=0.005, gp_n_factors=20,
        gp_y_target="next_5d_ret",
        gp_feature_subset=("ma_5", "ma_10", "ma_20", "o_c", "h_c", "l_c"),
        gp_dedup_threshold=0.6,
    ),
    GPStrategyConfig(
        id="B13", name="夏普猎人", description="波动+量能子集挖掘5日Sharpe",
        top_n=4, rebalance_hours=12, stop_loss=0.03, max_position_pct=0.25,
        factor_selection="all", scoring_method="ic_weighted",
        gp_seed=1313, gp_population=400, gp_generations=25, gp_n_runs=5, gp_parsimony=0.005, gp_n_factors=18,
        gp_y_target="next_5d_sharpe",
        gp_feature_subset=("std_5", "std_10", "std_20", "v_vma20"),
        gp_dedup_threshold=0.6,
    ),
    GPStrategyConfig(
        id="B14", name="抗跌守卫", description="防御特征挖掘最差日抗跌信号",
        top_n=5, rebalance_hours=24, stop_loss=0.025, max_position_pct=0.20,
        factor_selection="all", scoring_method="ic_weighted",
        gp_seed=1414, gp_population=400, gp_generations=25, gp_n_runs=5, gp_parsimony=0.005, gp_n_factors=18,
        gp_y_target="next_5d_minret_neg",
        gp_feature_subset=("std_5", "std_20", "ma_20", "l_c", "v_vma20"),
        gp_dedup_threshold=0.6,
    ),
    GPStrategyConfig(
        id="B15", name="量价共振", description="量价交互子集挖掘3日收益",
        top_n=5, rebalance_hours=8, stop_loss=0.03, max_position_pct=0.22,
        factor_selection="all", scoring_method="ic_weighted",
        gp_seed=1515, gp_population=400, gp_generations=22, gp_n_runs=5, gp_parsimony=0.005, gp_n_factors=20,
        gp_y_target="next_3d_ret",
        gp_feature_subset=("v_vma20", "ret_1", "ret_5", "h_c", "l_c"),
        gp_dedup_threshold=0.6,
    ),
    GPStrategyConfig(
        id="B16", name="反转捕手", description="短期均线反转挖掘负相关信号",
        top_n=4, rebalance_hours=12, stop_loss=0.025, max_position_pct=0.22,
        factor_selection="all", scoring_method="ic_weighted",
        gp_seed=1616, gp_population=400, gp_generations=25, gp_n_runs=5, gp_parsimony=0.005, gp_n_factors=18,
        gp_y_target="reversal_2d",
        gp_feature_subset=("ma_5", "ma_10", "ret_1", "std_5"),
        gp_dedup_threshold=0.6,
    ),
    # ---- F11-F16: FactorMiner-style GP 隔离账户族 ----
    # Mirrors B11-B16 personalities but mines with expanded terminals,
    # experience memory, global F-family correlation checks, and replacement logs.
    GPStrategyConfig(
        id="F11", name="记忆短打", description="FactorMiner记忆版：短动量1日收益",
        top_n=5, rebalance_hours=4, stop_loss=0.025, max_position_pct=0.20,
        factor_selection="all", scoring_method="ic_weighted",
        gp_seed=2111, gp_population=350, gp_generations=18, gp_n_runs=4, gp_parsimony=0.006, gp_n_factors=12,
        gp_y_target="next_1d_ret",
        gp_feature_subset=("ret_1", "ret_5", "ret_10", "gap_1", "range_pos", "pv_corr_20"),
        gp_dedup_threshold=0.6,
        family="F", mining_backend="factor_miner_gp", enabled_markets=("US", "CN"),
        gp_global_corr_threshold=0.6,
    ),
    GPStrategyConfig(
        id="F12", name="记忆趋势", description="FactorMiner记忆版：周度趋势/形态5日收益",
        top_n=6, rebalance_hours=24, stop_loss=0.04, max_position_pct=0.20,
        factor_selection="all", scoring_method="ic_weighted",
        gp_seed=2212, gp_population=400, gp_generations=20, gp_n_runs=4, gp_parsimony=0.006, gp_n_factors=12,
        gp_y_target="next_5d_ret",
        gp_feature_subset=("ma_5", "ma_10", "ma_20", "o_c", "h_c", "l_c", "range_pos", "upper_pos", "lower_shadow", "slope_20", "trend_r2_20", "trend_resi_20"),
        gp_dedup_threshold=0.6,
        family="F", mining_backend="factor_miner_gp", enabled_markets=("US", "CN"),
        gp_global_corr_threshold=0.6,
    ),
    GPStrategyConfig(
        id="F13", name="记忆夏普", description="FactorMiner记忆版：波动/流动性5日Sharpe",
        top_n=4, rebalance_hours=12, stop_loss=0.03, max_position_pct=0.25,
        factor_selection="all", scoring_method="ic_weighted",
        gp_seed=2313, gp_population=350, gp_generations=20, gp_n_runs=4, gp_parsimony=0.006, gp_n_factors=10,
        gp_y_target="next_5d_sharpe",
        gp_feature_subset=("std_5", "std_10", "std_20", "vol_of_vol_20", "v_vma20", "dvol_vma20", "ret_1_dvol", "absret_1_dvol", "skew_20", "kurt_20"),
        gp_dedup_threshold=0.6,
        family="F", mining_backend="factor_miner_gp", enabled_markets=("US", "CN"),
        gp_global_corr_threshold=0.6,
    ),
    GPStrategyConfig(
        id="F14", name="记忆防御", description="FactorMiner记忆版：抗跌/回撤防御信号",
        top_n=5, rebalance_hours=24, stop_loss=0.025, max_position_pct=0.20,
        factor_selection="all", scoring_method="ic_weighted",
        gp_seed=2414, gp_population=350, gp_generations=20, gp_n_runs=4, gp_parsimony=0.006, gp_n_factors=10,
        gp_y_target="next_5d_minret_neg",
        gp_feature_subset=("std_5", "std_20", "ma_20", "l_c", "range_pos", "lower_shadow", "v_vma20", "dvol_vma20", "trend_r2_20", "trend_resi_20"),
        gp_dedup_threshold=0.6,
        family="F", mining_backend="factor_miner_gp", enabled_markets=("US", "CN"),
        gp_global_corr_threshold=0.6,
    ),
    GPStrategyConfig(
        id="F15", name="记忆量价", description="FactorMiner记忆版：量价交互3日收益",
        top_n=5, rebalance_hours=8, stop_loss=0.03, max_position_pct=0.22,
        factor_selection="all", scoring_method="ic_weighted",
        gp_seed=2515, gp_population=350, gp_generations=18, gp_n_runs=4, gp_parsimony=0.006, gp_n_factors=12,
        gp_y_target="next_3d_ret",
        gp_feature_subset=("v_vma20", "dvol_vma20", "ret_1", "ret_5", "h_c", "l_c", "range_pos", "pv_corr_20", "ret_1_dvol", "absret_1_dvol"),
        gp_dedup_threshold=0.6,
        family="F", mining_backend="factor_miner_gp", enabled_markets=("US", "CN"),
        gp_global_corr_threshold=0.6,
    ),
    GPStrategyConfig(
        id="F16", name="记忆反转", description="FactorMiner记忆版：短反转/残差信号",
        top_n=4, rebalance_hours=12, stop_loss=0.025, max_position_pct=0.22,
        factor_selection="all", scoring_method="ic_weighted",
        gp_seed=2616, gp_population=350, gp_generations=20, gp_n_runs=4, gp_parsimony=0.006, gp_n_factors=10,
        gp_y_target="reversal_2d",
        gp_feature_subset=("ma_5", "ma_10", "ret_1", "ret_5", "std_5", "gap_1", "range_pos", "slope_20", "trend_resi_20"),
        gp_dedup_threshold=0.6,
        family="F", mining_backend="factor_miner_gp", enabled_markets=("US", "CN"),
        gp_global_corr_threshold=0.6,
    ),
]

# Fresh paper control for the live B16 rollout. ``replace`` keeps every B16
# portfolio parameter identical; only identity/label and signal provenance vary.
_B16 = next(g for g in GP_STRATEGIES if g.id == "B16")
GP_STRATEGIES.append(replace(
    _B16,
    id="PB16",
    name="反转捕手·实盘对照",
    description="B16同因子同排名的独立Paper对照账户（2026-08-27起）",
    enabled_markets=("US",),
    signal_source_id="B16",
))


def active_gp_strategies_for_market(market: str) -> List[GPStrategyConfig]:
    """Return GP strategies enabled for a market before market-prefixing ids."""
    m = market.upper()
    return [g for g in GP_STRATEGIES if m in tuple(x.upper() for x in g.enabled_markets)]
