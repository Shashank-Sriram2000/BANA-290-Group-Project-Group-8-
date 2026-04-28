"""
Stack Overflow Data Collector — BANA 290 Group 8
================================================
Research Question:
    What is the causal effect of ChatGPT's November 2022 release on Stack Overflow usage?

Design:
    - Interrupted Time Series (ITS)
    - Difference-in-Differences (DiD) with tag-level treatment intensity as variation

Intervention date: 2022-11-30 (ChatGPT public launch)
Pre-period:        2020-01-01 → 2022-11-30  (nearly 3 years of baseline)
Post-period:       2022-12-01 → 2024-12-31  (2 years post-launch)

Metrics collected per day:
    1. question_count          — total new questions
    2. answered_rate           — share of questions with at least one answer
    3. accepted_answer_rate    — share with an accepted answer
    4. avg_score               — mean question score
    5. avg_view_count          — mean view count (proxy for reader demand)

Tag strategy:
    - "AI-adjacent" tags (python, machine-learning, nlp, pytorch, …)
        → expected HIGH treatment intensity
    - "AI-insulated" tags (sql, excel, vba, bash, r, …)
        → expected LOW treatment intensity  ← DiD control group

Rate limits:
    - Anonymous: 300 requests/day
    - Registered app key: 10,000 requests/day  ← STRONGLY recommended
    - This script respects backoff headers and caches results to disk

Usage:
    # Install dependencies
    pip install requests pandas tqdm python-dotenv

    # Optional: put your key in .env
    echo "SE_API_KEY=your_key_here" > .env

    python so_data_collector.py
"""

import os
import time
import json
import gzip
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import pandas as pd
from tqdm import tqdm

# ── optional: load variables.env so you don't hardcode keys ─────────────────
try:
    from dotenv import load_dotenv
    load_dotenv("variables.env")
except ImportError:
    pass

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE      = "https://api.stackexchange.com/2.3"
SITE          = "stackoverflow"
API_KEY       = os.getenv("SE_API_KEY", "")        # leave blank → 300 req/day
                                                    # register at stackapps.com
OUTPUT_DIR    = Path("data")
CACHE_DIR     = Path("cache")

# Date range ─────────────────────────────────────────────────────────────────
START_DATE    = datetime(2020, 1, 1,  tzinfo=timezone.utc)
END_DATE      = datetime(2024, 12, 31, tzinfo=timezone.utc)
INTERVENTION  = datetime(2022, 11, 30, tzinfo=timezone.utc)

# Sampling granularity ────────────────────────────────────────────────────────
# "daily" is ideal for ITS; use "weekly" to reduce API calls during dev/testing
GRANULARITY   = "weekly"   # "daily" | "weekly"

# Tags ────────────────────────────────────────────────────────────────────────
# Treatment group — AI-adjacent (expected high substitution by ChatGPT)
# Kept: distinct, high-volume tags with strong pre-2022 baseline
# Dropped: deep-learning/keras/scikit-learn (redundant), openai-api/langchain (near-zero pre-2022)
TREATMENT_TAGS = [
    "python",
    "machine-learning",
    "nlp",
    "pytorch",
    "tensorflow",
]

# Control group — AI-insulated (expected low substitution)
# Kept: broad, high-volume tags clearly unaffected by ChatGPT
# Dropped: powershell/sap/oracle/ms-access/tableau (redundant coverage)
CONTROL_TAGS = [
    "sql",
    "excel",
    "vba",
    "bash",
    "r",
]

ALL_TAGS = TREATMENT_TAGS + CONTROL_TAGS

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────

def to_unix(dt: datetime) -> int:
    return int(dt.timestamp())


def date_windows(start: datetime, end: datetime, step_days: int = 1):
    """Yield (window_start, window_end) pairs of UTC datetimes."""
    cur = start
    delta = timedelta(days=step_days)
    while cur < end:
        nxt = min(cur + delta, end)
        yield cur, nxt
        cur = nxt


def cache_path(tag: str, from_dt: datetime, to_dt: datetime) -> Path:
    key = f"{tag}__{from_dt.strftime('%Y%m%d')}__{to_dt.strftime('%Y%m%d')}.json"
    return CACHE_DIR / key


def load_cache(path: Path) -> Optional[dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_cache(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def se_get(endpoint: str, params: dict, retries: int = 5) -> dict:
    """
    Make a GET request to the Stack Exchange API with:
        • automatic gzip decompression
        • backoff on 429 / throttle
        • quota monitoring
    """
    url = f"{API_BASE}/{endpoint}"
    if API_KEY:
        params["key"] = API_KEY
    params["site"] = SITE

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)

            # Stack Exchange always gzip-encodes; requests handles it automatically
            # but they set Content-Encoding, so let's be safe:
            if resp.status_code == 400:
                body = resp.json()
                log.warning("API 400: %s", body.get("error_message", ""))
                return {}

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 30))
                log.warning("Rate limited. Sleeping %ds …", wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Warn when quota is running low; stop cleanly at 10
            quota = data.get("quota_remaining", None)
            if quota is not None:
                if quota < 100:
                    log.warning("⚠  Quota remaining: %d", quota)
                if quota <= 10:
                    log.error("Quota exhausted (%d remaining). Stopping cleanly - re-run tomorrow.", quota)
                    raise SystemExit(1)

            # Respect backoff field — SE sometimes asks you to slow down
            backoff = data.get("backoff", 0)
            if backoff:
                log.debug("Backoff requested: %ds", backoff)
                time.sleep(backoff)

            return data

        except requests.RequestException as exc:
            wait = 2 ** attempt
            log.error("Request error (%s). Retry in %ds …", exc, wait)
            time.sleep(wait)

    log.error("Giving up on %s after %d attempts", endpoint, retries)
    return {}


# ── Core collection ──────────────────────────────────────────────────────────

def fetch_window_stats(tag: str, from_dt: datetime, to_dt: datetime) -> dict:
    """
    Fetch aggregate stats for `tag` in the [from_dt, to_dt) window.
    Returns a dict with our chosen metrics.
    """
    cp = cache_path(tag, from_dt, to_dt)
    cached = load_cache(cp)
    if cached:
        return cached

    from_ts = to_unix(from_dt)
    to_ts   = to_unix(to_dt)

    # ── 1. Question count + basic stats ─────────────────────────────────────
    params = {
        "tagged":   tag,
        "fromdate": from_ts,
        "todate":   to_ts,
        "pagesize": 100,
        "page":     1,
        "order":    "desc",
        "sort":     "creation",
        # Use a filter that returns only the fields we need → smaller payload
        # Default filter already includes: answer_count, score, view_count, is_answered
        "filter":   "default",
    }

    all_items = []
    while True:
        data = se_get("questions", params)
        items = data.get("items", [])
        all_items.extend(items)
        if not data.get("has_more") or len(items) == 0:
            break
        params["page"] += 1
        # Safety: don't paginate past 10 pages per window to conserve quota
        if params["page"] > 10:
            log.debug("Pagination cap hit for tag=%s window=%s", tag, from_dt.date())
            break
        time.sleep(0.11)  # ~9 req/s to stay well under limits

    n = len(all_items)
    if n == 0:
        result = {
            "tag": tag,
            "date": from_dt.date().isoformat(),
            "question_count": 0,
            "answered_rate": None,
            "accepted_answer_rate": None,
            "avg_score": None,
            "avg_view_count": None,
        }
    else:
        result = {
            "tag": tag,
            "date": from_dt.date().isoformat(),
            "question_count": n,
            "answered_rate": sum(1 for q in all_items if q.get("is_answered")) / n,
            "accepted_answer_rate": sum(
                1 for q in all_items if q.get("accepted_answer_id")
            ) / n,
            "avg_score": sum(q.get("score", 0) for q in all_items) / n,
            "avg_view_count": sum(q.get("view_count", 0) for q in all_items) / n,
        }

    save_cache(cp, result)
    return result


# ── Orchestration ────────────────────────────────────────────────────────────

def collect_all():
    OUTPUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    step = 1 if GRANULARITY == "daily" else 7
    windows = list(date_windows(START_DATE, END_DATE, step_days=step))

    total_calls = len(ALL_TAGS) * len(windows)
    log.info(
        "Collection plan: %d tags × %d windows = %d API calls",
        len(ALL_TAGS), len(windows), total_calls,
    )
    if not API_KEY:
        log.warning(
            "No API key set. Anonymous quota = 300 req/day. "
            "You will need %d days to complete this run at current granularity. "
            "Register a free app at https://stackapps.com to get 10,000/day.",
            total_calls // 300 + 1,
        )

    records = []

    for tag in ALL_TAGS:
        group = "treatment" if tag in TREATMENT_TAGS else "control"
        for from_dt, to_dt in tqdm(windows, desc=f"[{group}] {tag}", leave=False):
            rec = fetch_window_stats(tag, from_dt, to_dt)
            rec["group"] = group
            # ITS variables ──────────────────────────────────────────────────
            date_obj = from_dt
            rec["time_index"] = (date_obj - START_DATE).days // step  # t
            rec["post_intervention"] = int(date_obj >= INTERVENTION)   # D
            rec["time_since_intervention"] = max(
                0, (date_obj - INTERVENTION).days // step
            )                                                            # t - T₀
            records.append(rec)
            time.sleep(0.12)  # ~8 req/s

    df = pd.DataFrame(records)

    # ── Derived columns ──────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    # For DiD: mark treatment group
    df["is_treatment"] = (df["group"] == "treatment").astype(int)

    # Interaction term for DiD: post × treatment
    df["did_interaction"] = df["post_intervention"] * df["is_treatment"]

    out = OUTPUT_DIR / "so_panel.csv"
    df.to_csv(out, index=False)
    log.info("Saved %d rows → %s", len(df), out)

    # ── Quick sanity summary ─────────────────────────────────────────────────
    summary = (
        df.groupby(["group", "post_intervention"])["question_count"]
        .agg(["mean", "sum"])
        .round(1)
    )
    print("\n── Sanity check: avg questions per window ──")
    print(summary.to_string())

    return df


# ── Survey supplement ────────────────────────────────────────────────────────

def build_survey_series():
    """
    Manually encode SO Developer Survey AI adoption data (2022-2025).
    These are published figures from survey.stackoverflow.co — use as
    an exogenous time series to validate parallel with your API panel.
    
    Fields captured from the 'Technology' / 'AI' sections:
        - pct_using_ai_tools   : % of respondents currently using AI dev tools
        - pct_want_ai_tools    : % who want to use them (2022/23 only)
    
    Update these numbers once you've reviewed each survey URL.
    """
    survey_data = [
        # 2022 — pre-ChatGPT; AI section was minimal
        {"year": 2022, "pct_using_ai_tools": None, "pct_want_ai_tools": None,
         "notes": "No dedicated AI tools section in 2022 survey"},
        # 2023 — first year with substantial AI tooling questions
        {"year": 2023, "pct_using_ai_tools": 44.0, "pct_want_ai_tools": 26.0,
         "notes": "ChatGPT/Copilot adoption questions first appeared"},
        # 2024 — update from survey.stackoverflow.co/2024/technology
        {"year": 2024, "pct_using_ai_tools": 62.0, "pct_want_ai_tools": None,
         "notes": "Placeholder — verify from survey"},
        # 2025 — update from survey.stackoverflow.co/2025/technology
        {"year": 2025, "pct_using_ai_tools": None, "pct_want_ai_tools": None,
         "notes": "Placeholder — fill after reviewing 2025 survey"},
    ]
    df = pd.DataFrame(survey_data)
    out = OUTPUT_DIR / "survey_ai_adoption.csv"
    df.to_csv(out, index=False)
    log.info("Survey scaffold saved → %s", out)
    return df


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(__doc__)

    # 1. Collect API panel data
    df_panel = collect_all()

    # 2. Build survey supplement scaffold
    df_survey = build_survey_series()

    print("\nDone. Output files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f}  ({f.stat().st_size / 1024:.1f} KB)")
