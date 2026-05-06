"""Qlib model accounts (Q01-Q10).

Each Q account reads scores from `factor_values` table where
factor_name = f'qlib_{id}_score' (written by scripts/qlib_retrain.py via
factors/qlib_signal.py).

Descriptions are bilingual (zh/en). The dashboard picks one based on the
`lang` query param; main.py persists the zh version into account_meta.description
for backward compat with hourly reports.
"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass(frozen=True)
class QlibStrategyConfig:
    id: str
    name: str            # human-readable (zh)
    name_en: str         # human-readable (en)
    model_class: str     # short tag (e.g. 'LightGBM')
    desc_zh: str
    desc_en: str
    top_n: int
    rebalance_hours: int
    stop_loss: float
    max_position_pct: float

    @property
    def description(self) -> str:
        # Back-compat: main.py reads `.description` to populate DB. Default = zh.
        return self.desc_zh

    def desc(self, lang: str = "zh") -> str:
        return self.desc_en if lang == "en" else self.desc_zh

    def display_name(self, lang: str = "zh") -> str:
        return self.name_en if lang == "en" else self.name


QLIB_STRATEGIES: List[QlibStrategyConfig] = [
    QlibStrategyConfig(
        id="Q01", name="LightGBM 排序", name_en="LightGBM Ranker", model_class="LightGBM",
        desc_zh=(
            "【截面派 · 决策树】把每只股票当来体检的人——Alpha158 的 158 个指标（最近5日涨跌、成交量比、波动率、换手率…）就是体检报告。"
            "模型学的是『什么样的体检结果，明天涨得最多』。\n\n"
            "类比：一棵问诊树。『成交量是不是放大了？是→走左边；放大了又看动量是不是正？是→预测涨。』模型自动学出几百棵这样的树投票。\n\n"
            "✅ 优点：训练飞快（几秒）、对脏数据/缺失值最不挑、可解释（能告诉你哪个指标最重要）。\n"
            "❌ 缺点：完全不记得昨天发生过什么，每天都是新的一天（零时序记忆）。\n\n"
            "对比：和 Q02 XGBoost / Q03 CatBoost 是 GBDT 三剑客，区别在 LightGBM 用 leaf-wise 生长——最快但最易过拟合。"
        ),
        desc_en=(
            "【Cross-sectional · Decision Trees】Each stock is like a patient at a checkup; Alpha158's 158 indicators "
            "(5-day return, volume ratio, volatility, turnover…) are the lab results. The model learns "
            "\"what kind of report predicts tomorrow's biggest gainer.\"\n\n"
            "Analogy: a triage tree. \"Volume up? Yes → left branch; also positive momentum? Yes → predict up.\" "
            "The model auto-builds hundreds of such trees and votes.\n\n"
            "✅ Pros: trains in seconds, robust to dirty/missing data, interpretable (feature importance available).\n"
            "❌ Cons: zero memory of yesterday — every day is a fresh day.\n\n"
            "Vs siblings: with Q02 XGBoost / Q03 CatBoost it forms the GBDT trio. LightGBM uses leaf-wise growth — "
            "fastest but most prone to overfitting."
        ),
        top_n=8, rebalance_hours=24, stop_loss=0.04, max_position_pct=0.15,
    ),
    QlibStrategyConfig(
        id="Q02", name="XGBoost 排序", name_en="XGBoost Ranker", model_class="XGBoost",
        desc_zh=(
            "【截面派 · 决策树 · 严谨版】和 Q01 LightGBM 同门，但生长方式不同。"
            "LightGBM 像着急的医生，先抓最像病的那条线问到底；XGBoost 像谨慎的医生，每一层把所有问题都问一遍（level-wise 生长）。\n\n"
            "✅ 优点：正则化最完善（L1/L2 + 树剪枝），过拟合控制比 LightGBM 强。\n"
            "❌ 缺点：训练慢于 LightGBM。\n\n"
            "和 Q01 互为对照——同特征、同目标、不同生长策略，看哪种正则化在金融噪声里更稳。"
        ),
        desc_en=(
            "【Cross-sectional · Decision Trees · Rigorous】Same family as Q01, different growth strategy. "
            "LightGBM is the impatient doctor who drills down on the most suspicious branch; XGBoost is the cautious "
            "one asking every question on each level (level-wise growth).\n\n"
            "✅ Pros: most thorough regularization (L1/L2 + tree pruning), better overfitting control than Q01.\n"
            "❌ Cons: slower training than LightGBM.\n\n"
            "Paired with Q01 — same features, same target, different growth — to see which regularization wins in noisy markets."
        ),
        top_n=8, rebalance_hours=24, stop_loss=0.04, max_position_pct=0.15,
    ),
    QlibStrategyConfig(
        id="Q03", name="CatBoost 排序", name_en="CatBoost Ranker", model_class="CatBoost",
        desc_zh=(
            "【截面派 · 决策树 · 对称版】GBDT 第三家。每棵树长得『对称』——同一层只问同一个问题（oblivious tree），强迫模型简洁。"
            "外加『有序提升』（ordered boosting）防止训练数据泄漏。\n\n"
            "✅ 优点：对市场风格切换最稳（训练分布 ≠ 实盘分布时不容易翻车），对类别特征原生支持。\n"
            "❌ 缺点：训练最慢，单树容量受对称结构限制。\n\n"
            "Q01+Q02+Q03 形成『三专家会诊』——三家都看好同一票时，可作信号置信度的额外参考。"
        ),
        desc_en=(
            "【Cross-sectional · Decision Trees · Symmetric】The third GBDT. Every tree is symmetric — same question "
            "across each level (oblivious tree) — forcing the model to stay simple. Plus 'ordered boosting' to "
            "prevent training-data leakage.\n\n"
            "✅ Pros: most robust to regime shifts (training ≠ live distribution), native categorical support.\n"
            "❌ Cons: slowest training, per-tree capacity limited by symmetry.\n\n"
            "Q01+Q02+Q03 form a \"three-expert consult\" — when all three agree on a name, that's high-confidence signal."
        ),
        top_n=8, rebalance_hours=24, stop_loss=0.04, max_position_pct=0.15,
    ),
    QlibStrategyConfig(
        id="Q04", name="Ridge 线性", name_en="Ridge Linear", model_class="Ridge",
        desc_zh=(
            "【截面派 · 线性 · 最朴素】L2 正则化线性回归——初中数学水平的模型。"
            "类比：班主任给每个指标打权重，『动量算 30 分、波动率扣 20 分、成交量加 10 分…』加起来就是预测分。\n\n"
            "✅ 优点：训练 1 秒、每个指标的权重清清楚楚（完全可解释）、噪声多时反而最稳（永不过拟合）。\n"
            "❌ 缺点：只会『加加减减』，抓不到『高动量 + 低波动一起出现才好』这种组合规律。\n\n"
            "★ 重要参考：如果 Q04 跑赢 Q01–Q03，说明那些复杂模型只是在过拟合噪声——这是金融里的常见情况，Ridge 是基准照妖镜。"
        ),
        desc_en=(
            "【Cross-sectional · Linear · The simplest】L2-regularized linear regression — middle-school math.\n"
            "Analogy: a homeroom teacher assigns weights to each indicator: \"momentum +30, volatility -20, "
            "volume +10…\" sum it up, that's the prediction.\n\n"
            "✅ Pros: trains in 1s, every weight is fully transparent, most stable when data is noisy (cannot overfit).\n"
            "❌ Cons: only does add/subtract — can't catch interactions like \"high momentum AND low volatility\".\n\n"
            "★ Key benchmark: if Q04 beats Q01–Q03, those complex models are merely overfitting noise — a common "
            "outcome in finance. Ridge is the reality check."
        ),
        top_n=12, rebalance_hours=24, stop_loss=0.05, max_position_pct=0.10,
    ),
    QlibStrategyConfig(
        id="Q05", name="MLP 神经网", name_en="MLP Neural Net", model_class="MLP",
        desc_zh=(
            "【截面派 · 神经网络】多层感知机（128→64→1 的浅层 DNN），把 158 个指标搅在一起非线性地组合。\n\n"
            "✅ 优点：能拟合任意非线性因子交互（树模型靠分裂逼近，MLP 直接连续逼近——更平滑）。\n"
            "❌ 缺点：每次重训随机性大、对量纲敏感（已加 Fillna+Standardize 预处理）。\n\n"
            "在截面派 Q01-Q05 里，MLP 是唯一会『连续平滑思考』的，弥补树模型只会『硬切分』的毛病。"
        ),
        desc_en=(
            "【Cross-sectional · Neural Net】Multi-layer perceptron (128→64→1 shallow DNN), blending 158 indicators "
            "into a smooth non-linear combination.\n\n"
            "✅ Pros: can fit any non-linear interaction (trees approximate via splits, MLP approximates continuously — smoother).\n"
            "❌ Cons: high variance across re-trains, sensitive to feature scaling (Fillna+Standardize preprocessing applied).\n\n"
            "Among Q01–Q05, MLP is the only \"continuous thinker\" — compensating for trees' hard-split limitations."
        ),
        top_n=10, rebalance_hours=24, stop_loss=0.04, max_position_pct=0.12,
    ),
    QlibStrategyConfig(
        id="Q06", name="LSTM 时序", name_en="LSTM Sequential", model_class="LSTM",
        desc_zh=(
            "【时序派 · 老牌 RNN】不再是体检报告，而是过去 60 天每天的盘面（Alpha360 = 6 维 OHLCV × 60 日）。"
            "Q06–Q10 是时序家族，看的是『动态』而非『快照』。\n\n"
            "类比：一个有便签本的人，每看一天数据就在便签上更新『我现在该记住什么、忘掉什么』——这就是 LSTM 的『遗忘门 / 输入门 / 输出门』。\n\n"
            "✅ 优点：有显式的『记忆门』，能学习如『连续 5 日放量上涨』这类时间模式。\n"
            "❌ 缺点：序列太长会梯度衰减（前面的事记不住），训练比 GBDT 慢一个数量级。\n\n"
            "和截面模型 Q01-Q05 的根本区别：Q06 看 60 天的『录像』，截面只看『当前快照』。"
        ),
        desc_en=(
            "【Sequential · Classic RNN】Not a checkup snapshot — but the last 60 days of bar data "
            "(Alpha360 = 6-dim OHLCV × 60 days). Q06–Q10 are the time-series family: they watch the *evolution*, not the snapshot.\n\n"
            "Analogy: a person with a notepad who, each day, updates \"what should I remember, what should I forget\" — "
            "that's LSTM's forget/input/output gates.\n\n"
            "✅ Pros: explicit memory gates can learn temporal patterns like \"5 consecutive days of rising volume.\"\n"
            "❌ Cons: gradients decay over long sequences (early days fade), trains 10× slower than GBDT.\n\n"
            "Fundamental difference from cross-sectional Q01–Q05: Q06 watches a 60-day movie; Q01–Q05 see a single frame."
        ),
        top_n=10, rebalance_hours=24, stop_loss=0.04, max_position_pct=0.12,
    ),
    QlibStrategyConfig(
        id="Q07", name="GRU 时序", name_en="GRU Sequential", model_class="GRU",
        desc_zh=(
            "【时序派 · LSTM 的轻量版】门控循环单元——LSTM 的精简版，3 个门压缩成 2 个。同样吃 Alpha360 的 60 天序列。\n\n"
            "类比：同样的便签本，但格子更少、写字更快。\n\n"
            "✅ 优点：参数量比 LSTM 少 1/3，训练快、小数据上更稳。\n"
            "❌ 缺点：表达力略弱，对超长依赖（>30 日）不如 LSTM。\n\n"
            "和 Q06 对照：60 日窗口下两者性能往往接近——GRU 胜出说明数据量没大到 LSTM 优势体现的地步。"
        ),
        desc_en=(
            "【Sequential · LSTM-lite】Gated Recurrent Unit — a streamlined LSTM with 2 gates instead of 3. "
            "Also feeds on Alpha360's 60-day sequence.\n\n"
            "Analogy: same notepad, fewer columns, faster writing.\n\n"
            "✅ Pros: ~33% fewer parameters than LSTM, faster training, more stable on smaller data.\n"
            "❌ Cons: slightly weaker expressiveness; weaker on very long dependencies (>30 days).\n\n"
            "Vs Q06: in a 60-day window the two are usually neck-and-neck — if GRU wins, it means our data isn't "
            "large enough for LSTM's advantage to materialize."
        ),
        top_n=10, rebalance_hours=24, stop_loss=0.04, max_position_pct=0.12,
    ),
    QlibStrategyConfig(
        id="Q08", name="Transformer 注意力", name_en="Transformer Attention", model_class="Transformer",
        desc_zh=(
            "【时序派 · 自注意力】ChatGPT 同款架构（精简版：d_model=32、1 头 1 层，CPU 友好）。"
            "不再『按顺序读』，而是把 60 天平铺在桌上，自己学『第 3 天和第 47 天有关系，应该一起看』——这就是『自注意力』。\n\n"
            "✅ 优点：可学习任意两天之间的依赖（不像 RNN 必须按顺序传递），并行训练。\n"
            "❌ 缺点：参数最多、最易过拟合、CPU 上训练最慢（约 45min/天）。\n\n"
            "在 Q06/Q07 RNN 家族里，Q08 代表『不靠顺序记忆，靠相关度加权』的另一条技术路线。"
        ),
        desc_en=(
            "【Sequential · Self-Attention】Same architecture as ChatGPT (slimmed: d_model=32, 1 head, 1 layer — CPU-friendly). "
            "Instead of reading day-by-day, it lays all 60 days on the table and learns "
            "\"day 3 is related to day 47 — read them together.\" That's self-attention.\n\n"
            "✅ Pros: can learn dependencies between any two days (no sequential bottleneck like RNN), parallel training.\n"
            "❌ Cons: most parameters, most prone to overfitting, slowest on CPU (~45min/day).\n\n"
            "Within Q06–Q10, Q08 represents the alternative path: not sequential memory, but global similarity weighting."
        ),
        top_n=10, rebalance_hours=24, stop_loss=0.04, max_position_pct=0.12,
    ),
    QlibStrategyConfig(
        id="Q09", name="TCN 卷积时序", name_en="TCN (Temporal CNN)", model_class="TCN",
        desc_zh=(
            "【时序派 · 卷积】时间卷积网络（3 层 dilated 1D 卷积 + 残差），用卷积代替 RNN 处理时序。\n\n"
            "类比：拿一个固定大小的滑动窗口（比如 7 天），每次扫过一段；多层叠加后窗口指数扩大到 60 天。\n\n"
            "✅ 优点：感受野指数扩张、并行训练（比 LSTM 快）、训练稳定（无梯度消失问题）。\n"
            "❌ 缺点：感受野固定，超出 kernel × layer 的远端依赖完全看不到。\n\n"
            "和 RNN（Q06/Q07）的本质区别：RNN 是『串行 + 全历史』，TCN 是『并行 + 固定窗口』。"
        ),
        desc_en=(
            "【Sequential · Convolution】Temporal Convolutional Net (3 dilated 1D conv layers + residual), "
            "replacing RNN with convolution.\n\n"
            "Analogy: a fixed-size sliding window (say 7 days) scans the sequence; stacking layers expands the window "
            "exponentially up to 60 days.\n\n"
            "✅ Pros: exponentially-growing receptive field, parallel training (faster than LSTM), stable gradients.\n"
            "❌ Cons: receptive field is bounded — anything beyond kernel × layers is invisible.\n\n"
            "Vs RNN (Q06/Q07): RNN is \"serial + full history\", TCN is \"parallel + fixed window\"."
        ),
        top_n=10, rebalance_hours=24, stop_loss=0.04, max_position_pct=0.12,
    ),
    QlibStrategyConfig(
        id="Q10", name="ALSTM 注意力 LSTM", name_en="ALSTM (Attention-LSTM)", model_class="ALSTM",
        desc_zh=(
            "【时序派 · 融合体】Attention + LSTM 融合——LSTM 编码 60 日序列，注意力层回头加权选关键日。\n\n"
            "类比：先用 LSTM 把 60 天读一遍记成笔记，再用注意力机制回头标重点：『第 12 天那个跳空缺口最关键』。\n\n"
            "✅ 优点：兼顾 LSTM 的顺序记忆 和 Transformer 的全局加权，金融时序的常胜军（学术与业界口碑最好）。\n"
            "❌ 缺点：层级较深、CPU 训练偏慢、对学习率敏感（调不好就崩）。\n\n"
            "Q10 是 Q06-Q09 时序四模型的『综合体』，理论上下限最高但训练也最娇贵。"
        ),
        desc_en=(
            "【Sequential · Hybrid】Attention + LSTM fusion — LSTM encodes the 60-day sequence; attention layer "
            "looks back and weights the most important days.\n\n"
            "Analogy: first read all 60 days with LSTM, take notes; then use attention to highlight the key day — "
            "\"the gap on day 12 is the critical one.\"\n\n"
            "✅ Pros: combines LSTM's sequential memory with Transformer's global weighting — the academic & industry "
            "favorite for financial time series.\n"
            "❌ Cons: deeper architecture, slow on CPU, sensitive to learning rate (will diverge if mistuned).\n\n"
            "Q10 is the synthesis of Q06–Q09 — highest theoretical ceiling, but the most temperamental to train."
        ),
        top_n=10, rebalance_hours=24, stop_loss=0.04, max_position_pct=0.12,
    ),
]
