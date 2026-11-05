"""Small regression test for receipt-level RFM aggregation.

Run after: pip install duckdb pandas
    python -m unittest discover -s tests
"""
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_rfm_features import build


class RFMFeaturesTest(unittest.TestCase):
    @staticmethod
    def _write_purchases(folder, rows):
        purchases = folder / "tiny.csv.gz"
        columns = ["client_id", "transaction_id", "transaction_datetime",
                   "purchase_sum", "store_id"]
        with gzip.open(purchases, "wt", encoding="utf-8") as stream:
            pd.DataFrame(rows, columns=columns).to_csv(stream, index=False)
        return purchases

    def test_repeated_receipt_total_is_counted_once_and_order_is_irrelevant(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            purchases = self._write_purchases(folder, [
                ["a", "t1", "2020-01-01 09:00:00", 100, "s1"],
                ["b", "t3", "2020-01-10 09:00:00", 120, "s2"],
                ["a", "t1", "2020-01-01 09:00:00", 100, "s1"],
                ["a", "t2", "2020-01-11 12:00:00", 30, "s1"],
            ])
            destination = folder / "final_stats.csv"
            build(str(purchases), str(destination), "1GB")
            res = pd.read_csv(destination).set_index("client_id")
            self.assertEqual(int(res.loc["a", "count"]), 2)
            self.assertAlmostEqual(float(res.loc["a", "sum"]), 130.0)
            self.assertAlmostEqual(float(res.loc["a", "mean_check"]), 65.0)
            self.assertEqual(int(res.loc["a", "recency_days"]), 1)
            self.assertEqual(int(res.loc["a", "active_span_days"]), 10)
            self.assertEqual(int(res.loc["a", "checks_30d"]), 2)
            self.assertAlmostEqual(float(res.loc["a", "spending_90d"]), 130.0)
            metadata = json.loads(destination.with_suffix(".meta.json").read_text())
            self.assertEqual(metadata["feature_version"], "retailhero_receipt_total_once_v2")
            self.assertEqual(metadata["receipt_count"], 3)
            self.assertEqual(metadata["client_count"], 2)
            self.assertEqual(int(res.loc["b", "count"]), 1)
            self.assertEqual(int(res.loc["b", "recency_days"]), 2)

    def test_real_receipt_example_with_nineteen_product_lines(self):
        # Reconstructed from the audit: 19 lines, each with the same whole-check
        # total of 1007, must produce 1007 (never 19 * 1007).
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            purchases = self._write_purchases(folder, [
                ["000012768d", "7e3e2e3984", "2020-01-01 09:00:00", 1007, "s1"]
                for _ in range(19)
            ])
            destination = folder / "final_stats.csv"
            build(str(purchases), str(destination), "1GB")
            result = pd.read_csv(destination)
            self.assertEqual(int(result.loc[0, "count"]), 1)
            self.assertEqual(float(result.loc[0, "sum"]), 1007.0)

    def test_rejects_inconsistent_or_invalid_receipts(self):
        first = ["a", "t1", "2020-01-01 09:00:00", 100, "s1"]
        malformed = {
            "amount": ["a", "t1", "2020-01-01 09:00:00", 200, "s1"],
            "timestamp": ["a", "t1", "2020-01-02 09:00:00", 100, "s1"],
            "store": ["a", "t1", "2020-01-01 09:00:00", 100, "s2"],
            "missing key": ["", "t2", "2020-01-01 09:00:00", 50, "s1"],
            "bad amount": ["a", "t2", "2020-01-01 09:00:00", "oops", "s1"],
        }
        for case, bad_row in malformed.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                purchases = self._write_purchases(folder, [first, bad_row])
                with self.assertRaisesRegex(ValueError, "invalid values or inconsistent"):
                    build(str(purchases), str(folder / "final_stats.csv"), "1GB")


if __name__ == "__main__":
    unittest.main()
