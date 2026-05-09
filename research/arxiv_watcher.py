"""
arXiv 量化研究前沿日报抓取器

设计原则（严格遵守 arxiv skill 中的反限流规则）:
  1. 单次请求合并多个分类（cat:A OR cat:B），不要循环。
  2. 请求间隔 ≥ 5 秒（arXiv 官方建议 3 秒，留余量）。
  3. 失败指数退避：5s / 15s / 60s，最多 3 次。
  4. 检测 "Rate exceeded"（HTTP 200 + 14 字节正文），视为限流并退避。
  5. 一律走 https://，避免 Fastly 301 浪费 round-trip。
  6. 本地缓存已见过的 arxiv_id（SQLite），仅推送新增。
  7. 每天最多调用 1~2 次（cron 触发），开发期不要循环测试。

使用：
  python -m research.arxiv_watcher           # 抓取 + 打印新增
  python -m research.arxiv_watcher --json    # JSON 输出（供 cron prompt 使用）
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

# ───────────────────────────── 配置 ─────────────────────────────
CATEGORIES = [
    "q-fin.TR",   # Trading & Microstructure
    "q-fin.PM",   # Portfolio Management
    "q-fin.ST",   # Statistical Finance
    "q-fin.CP",   # Computational Finance
    "q-fin.RM",   # Risk Management
]

# 关键词正向过滤（命中任一即收录），降低 cs.LG / stat.ML 噪音
KEYWORDS = [
    "alpha", "factor", "stock", "equity", "market microstructure",
    "limit order book", "high-frequency", "intraday", "portfolio",
    "trading strategy", "backtest", "qlib", "genetic programming",
    "symbolic regression", "reinforcement learning trading",
    "deep learning finance", "transformer finance", "lstm market",
    "asset pricing", "return prediction", "volatility forecasting",
]

MAX_RESULTS = 50
REQUEST_INTERVAL_SEC = 5.0      # 请求间隔
RETRY_BACKOFF = (5, 15, 60)     # 三次重试退避
USER_AGENT = "quant-trading/arxiv-watcher (gexin; daily; <=1 req/day)"

DB_PATH = Path(__file__).parent.parent / "trading.db"
NS = {"a": "http://www.w3.org/2005/Atom"}

# ───────────────────────────── 数据库 ─────────────────────────────
def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS arxiv_seen (
            arxiv_id     TEXT PRIMARY KEY,
            title        TEXT NOT NULL,
            published    TEXT NOT NULL,
            categories   TEXT NOT NULL,
            authors      TEXT NOT NULL,
            summary      TEXT NOT NULL,
            pdf_url      TEXT NOT NULL,
            seen_at      TEXT NOT NULL
        )
    """)
    conn.commit()


# ───────────────────────────── 抓取 ─────────────────────────────
def _build_url() -> str:
    expr = "+OR+".join(f"cat:{c}" for c in CATEGORIES)
    return (
        f"https://export.arxiv.org/api/query?"
        f"search_query={expr}"
        f"&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={MAX_RESULTS}"
    )


def _fetch_with_retry(url: str) -> str:
    """带退避的 HTTP GET，专门处理 arXiv 'Rate exceeded' 软限流。"""
    last_err: Exception | None = None
    for attempt, backoff in enumerate(RETRY_BACKOFF, 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            # arXiv 限流签名: HTTP 200 + 14 字节 "Rate exceeded."
            if body.strip() == "Rate exceeded.":
                raise RuntimeError("arXiv rate-limited (HTTP 200 'Rate exceeded.')")
            if len(body) < 200 or "<feed" not in body:
                raise RuntimeError(f"Unexpected short response ({len(body)} bytes)")
            return body
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, TimeoutError) as e:
            last_err = e
            if attempt < len(RETRY_BACKOFF):
                print(f"[arxiv] attempt {attempt} failed: {e!r}; sleeping {backoff}s",
                      file=sys.stderr)
                time.sleep(backoff)
    raise RuntimeError(f"arXiv fetch failed after {len(RETRY_BACKOFF)} attempts: {last_err!r}")


def _parse(xml_body: str) -> list[dict]:
    root = ET.fromstring(xml_body)
    out: list[dict] = []
    for e in root.findall("a:entry", NS):
        raw_id = e.find("a:id", NS).text.strip()
        arxiv_id = raw_id.split("/abs/")[-1]
        title = e.find("a:title", NS).text.strip().replace("\n", " ")
        summary = e.find("a:summary", NS).text.strip().replace("\n", " ")
        published = e.find("a:published", NS).text[:10]
        authors = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
        cats = [c.get("term") for c in e.findall("a:category", NS)]
        # 撤回检测
        if "withdrawn" in summary.lower()[:200] or "retracted" in summary.lower()[:200]:
            continue
        out.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "summary": summary,
            "published": published,
            "authors": authors,
            "categories": cats,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return out


def _keyword_filter(papers: list[dict]) -> list[dict]:
    out = []
    for p in papers:
        text = (p["title"] + " " + p["summary"]).lower()
        # q-fin.* 全收；其他类必须命中关键词
        is_qfin = any(c.startswith("q-fin") for c in p["categories"])
        if is_qfin or any(kw in text for kw in KEYWORDS):
            out.append(p)
    return out


# ───────────────────────────── 主流程 ─────────────────────────────
def run(emit_json: bool = False) -> int:
    conn = sqlite3.connect(DB_PATH)
    _init_db(conn)

    url = _build_url()
    print(f"[arxiv] GET {url}", file=sys.stderr)
    body = _fetch_with_retry(url)
    time.sleep(REQUEST_INTERVAL_SEC)  # 礼貌停顿，避免影响后续任何请求

    papers = _parse(body)
    papers = _keyword_filter(papers)

    # 过滤已见
    seen = {row[0] for row in conn.execute("SELECT arxiv_id FROM arxiv_seen")}
    new = [p for p in papers if p["arxiv_id"] not in seen]

    now_iso = datetime.now(timezone.utc).isoformat()
    for p in new:
        conn.execute(
            "INSERT OR IGNORE INTO arxiv_seen "
            "(arxiv_id,title,published,categories,authors,summary,pdf_url,seen_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (p["arxiv_id"], p["title"], p["published"],
             ",".join(p["categories"]), ", ".join(p["authors"]),
             p["summary"], p["pdf_url"], now_iso),
        )
    conn.commit()
    conn.close()

    if emit_json:
        print(json.dumps({"new": new, "total_seen_today": len(papers)},
                         ensure_ascii=False, indent=2))
    else:
        print(f"\n=== arXiv Quant Frontier — {now_iso[:10]} ===")
        print(f"扫描分类: {', '.join(CATEGORIES)}")
        print(f"返回 {len(papers)} 篇 / 新增 {len(new)} 篇\n")
        for i, p in enumerate(new, 1):
            print(f"{i}. [{p['arxiv_id']}] {p['title']}")
            print(f"   {p['published']} | {', '.join(p['categories'])}")
            print(f"   {', '.join(p['authors'][:4])}{' et al.' if len(p['authors'])>4 else ''}")
            print(f"   {p['summary'][:240]}...")
            print(f"   {p['abs_url']}\n")

    return len(new)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="输出 JSON（供 cron 使用）")
    args = ap.parse_args()
    sys.exit(0 if run(emit_json=args.json) >= 0 else 1)
