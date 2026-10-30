"""Small regression test for receipt-level RFM aggregation.

Run after: pip install duckdb pandas
    python -m unittest discover -s tests
"""
import gzip
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_rfm_features import build


class RFMFeaturesTest(unittest.TestCase):
    def test_receipt_lines_are_summed_and_order_is_irrelevant(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            purchases = folder / "tiny.csv.gz"
            rows = pd.DataFrame([
                ["a", "t1", "2020-01-01 09:00:00", 40],
                ["b", "t3", "2020-01-10 09:00:00", 120],
                ["a", "t1", "2020-01-01 09:00:00", 60],
                ["a", "t2", "2020-01-11 12:00:00", 30],
            ], columns=["client_id", "transaction_id",
                        "transaction_datetime", "purchase_sum"])
            with gzip.open(purchases, "wt", encoding="utf-8") as stream:
                rows.to_csv(stream, index=False)
            destination = folder / "final_stats.csv"
            build(str(purchases), str(destination), "1GB")
            res = pd.read_csv(destination).set_index("client_id")
            self.assertEqual(int(res.loc["a", "count"]), 2)
            self.assertAlmostEqual(float(res.loc["a", "sum"]), 130.0)
            self.assertAlmostEqual(float(res.loc["a", "mean_check"]), 65.0)
            self.assertEqual(int(res.loc["a", "recency_days"]), 1)
            self.assertEqual(int(res.loc["a", "active_span_days"]), 10)
            self.assertEqual(int(res.loc["a", "checks_30d"]), 2)
            self.assertEqual(int(res.loc["b", "count"]), 1)
            self.assertEqual(int(res.loc["b", "recency_days"]), 2)


if __name__ == "__main__":
    unittest.main()
