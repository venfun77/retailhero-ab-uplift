#!/usr/bin/env python3
"""One-time, memory-friendly preparation of pre-campaign client purchase features.

Assumption (explicitly validated): purchases.csv.gz is sorted by client_id.
The buffer holds *all rows of the trailing client*, not just trailing receipt.
Therefore each client's receipts can be de-duplicated globally, even if a receipt
has non-contiguous product lines within that client's contiguous block.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def build(input_file: str, output_file: str, chunksize: int = 200_000) -> None:
    pieces = []
    buffer = pd.DataFrame()
    previous_last_id = None
    min_date, max_date = None, None
    total_rows = 0

    reader = pd.read_csv(
        input_file, chunksize=chunksize,
        dtype={'client_id': 'string', 'transaction_id': 'string'},
        usecols=['client_id', 'transaction_id', 'purchase_sum', 'transaction_datetime'],
        compression='infer',
    )
    for i, chunk in enumerate(reader):
        if chunk['client_id'].isna().any() or chunk['transaction_id'].isna().any():
            raise ValueError('Missing client_id / transaction_id; manual review required.')
        if not chunk['client_id'].is_monotonic_increasing:
            raise ValueError(f'Client rows are not sorted by client_id in chunk {i}.')
        if previous_last_id is not None and chunk['client_id'].iat[0] < previous_last_id:
            raise ValueError(f'Global client order breaks in chunk {i}.')
        previous_last_id = chunk['client_id'].iat[-1]

        total_rows += len(chunk)
        start = chunk['transaction_datetime'].min()
        end = chunk['transaction_datetime'].max()
        min_date = start if min_date is None else min(start, min_date)
        max_date = end if max_date is None else max(end, max_date)

        combined = pd.concat([buffer, chunk], ignore_index=True)
        last_id = combined['client_id'].iat[-1]
        trailing = combined['client_id'].eq(last_id)
        buffer = combined.loc[trailing].copy()
        ready = combined.loc[~trailing]
        if not ready.empty:
            receipt = ready.drop_duplicates(['client_id', 'transaction_id'])
            piece = receipt.groupby('client_id', sort=False)['purchase_sum'].agg(
                count='size', sum='sum'
            )
            pieces.append(piece)
        if (i+1) % 50 == 0:
            print(f'Chunks: {i+1}; product lines read: {total_rows:,}')

    if not buffer.empty:
        receipt = buffer.drop_duplicates(['client_id', 'transaction_id'])
        pieces.append(receipt.groupby('client_id')['purchase_sum'].agg(count='size', sum='sum'))
    if not pieces:
        raise ValueError('No receipt data found.')

    features = pd.concat(pieces).groupby(level=0)[['count','sum']].sum()
    features['count'] = features['count'].astype('int64')
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(target)
    print(f'Finished: {len(features):,} clients; {features["count"].sum():,} receipts')
    print(f'History interval: {min_date} — {max_date}')
    print(f'Output: {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to purchases.csv.gz')
    parser.add_argument('--output', default='data/final_stats.csv')
    parser.add_argument('--chunksize', type=int, default=200_000)
    args = parser.parse_args()
    build(args.input, args.output, args.chunksize)
