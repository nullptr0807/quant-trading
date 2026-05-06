#!/usr/bin/env python3
"""
运行回测并通过Telegram发送报告。
用法: python -m backtest.run [--days 30]
"""

import sys
import os
import logging
import argparse

PROJECT_ROOT = os.path.expanduser("~/quant-trading")
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("backtest")


def main():
    parser = argparse.ArgumentParser(description="回测系统")
    parser.add_argument("--days", type=int, default=30, help="回测天数")
    parser.add_argument("--adaptive", action="store_true", help="强制启用自适应换仓")
    parser.add_argument("--no-adaptive", action="store_true", help="强制关闭自适应换仓")
    parser.add_argument("--no-telegram", action="store_true", help="不发送Telegram")
    args = parser.parse_args()

    from backtest.engine import BacktestEngine
    from backtest.report import generate_report

    # 确定自适应开关
    adaptive = None  # None = 读取配置文件
    if args.adaptive:
        adaptive = True
    elif args.no_adaptive:
        adaptive = False

    log.info("启动回测引擎 (回测 %d 天, 自适应=%s)...", args.days,
             "开" if adaptive is True else "关" if adaptive is False else "按配置")
    engine = BacktestEngine(days=args.days, adaptive_rebalance=adaptive)
    results, benchmarks = engine.run()

    if not results:
        log.error("回测失败，无结果")
        return

    report_text, chart_path = generate_report(results, days=args.days, benchmarks=benchmarks)
    print(report_text)

    if not args.no_telegram:
        try:
            from reports.telegram import TelegramReporter
            reporter = TelegramReporter()
            reporter.send_report(report_text, chart_path, force=True)
            log.info("回测报告已发送至Telegram")
        except Exception as e:
            log.error("发送Telegram失败: %s", e)


if __name__ == "__main__":
    main()
