#!/usr/bin/env python3
"""Build pre-campaign RFM features from the official X5 RetailHero purchases.csv.gz.

ODS describes purchases.csv as pre-communication history; uplift_train.csv contains
treatment and post-communication binary target. The actual SMS timestamp is NOT
provided. Recency is measured relative to the last timestamp in the supplied
history, not an invented campaign date.

The source has one row per transaction PRODUCT LINE. Hence a receipt is aggregated
from all its product lines before calculating client-level spend or check count.

DuckDB processes the large gzip file using bounded memory / temporary disk and
does not assume any order of client_id in the CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import timedelta

import duckdb


def build(input_path: str, output_path: str, memory_limit: str = "4GB") -> None:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Purchase history not found: {source}")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    db = duckdb.connect()
    db.execute(f"SET memory_limit='{memory_limit}'")
    tmp_dir = target.parent / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db.execute("SET temp_directory=?", [str(tmp_dir)])

    print("Aggregating receipt lines (purchases file is pre-campaign by dataset definition)...")
    db.execute("""
        CREATE TEMP TABLE receipts AS
        SELECT
            client_id,
            transaction_id,
            MIN(TRY_CAST(transaction_datetime AS TIMESTAMP)) AS purchase_time,
            SUM(TRY_CAST(purchase_sum AS DOUBLE)) AS receipt_sum
        FROM read_csv_auto(?, header=true, all_varchar=true)
        WHERE client_id IS NOT NULL AND transaction_id IS NOT NULL
        GROUP BY client_id, transaction_id
    """, [str(source)])
    invalid = db.execute("""
        SELECT COUNT(*) FROM receipts
        WHERE purchase_time IS NULL OR receipt_sum IS NULL
    """).fetchone()[0]
    if invalid:
        raise ValueError(f"{invalid} receipts have invalid date/amount; stop and review source.")

    first_date, last_date, receipts_n = db.execute("""
        SELECT MIN(purchase_time), MAX(purchase_time), COUNT(*) FROM receipts
    """).fetchone()
    if last_date is None:
        raise ValueError("No receipts found")
    # History ends BEFORE the campaign according to ODS; the actual SMS date is
    # not available. Anchor is the day after the last transaction in the file.
    reference = last_date + timedelta(days=1)

    features = db.execute("""
        SELECT
            client_id,
            COUNT(*)::BIGINT AS count,
            SUM(receipt_sum) AS sum,
            AVG(receipt_sum) AS mean_check,
            DATE_DIFF('day', MAX(purchase_time), ?) AS recency_days,
            DATE_DIFF('day', MIN(purchase_time), MAX(purchase_time)) AS active_span_days,
            COUNT(*) FILTER (WHERE purchase_time >= ? - INTERVAL '30 days')::BIGINT AS checks_30d,
            COUNT(*) FILTER (WHERE purchase_time >= ? - INTERVAL '90 days')::BIGINT AS checks_90d,
            SUM(receipt_sum) FILTER (WHERE purchase_time >= ? - INTERVAL '90 days') AS spending_90d
        FROM receipts
        GROUP BY client_id
    """, [reference, reference, reference, reference]).df()
    features["spending_90d"] = features["spending_90d"].fillna(0)
    if not features.client_id.is_unique:
        raise AssertionError("Duplicate client_id in features output")
    features.to_csv(target, index=False)
    print(f"Raw history: {first_date} — {last_date}; {receipts_n:,} receipts")
    print(f"Recency reference (proxy, not SMS timestamp): {reference}")
    print(f"Features for {len(features):,} clients -> {target}")
    print("The official dataset contract says history is pre-campaign. "
          "Without individual SMS timestamps this cannot be independently verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/final_stats.csv")
    parser.add_argument("--memory-limit", default="4GB")
    args = parser.parse_args()
    build(args.input, args.output, args.memory_limit)
