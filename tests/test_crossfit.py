import unittest
import numpy as np
import pandas as pd
from scripts.validate_crossfit import top, crossfit, reports, paired_bootstrap

class TestCrossfit(unittest.TestCase):
    def test_top(self):
        selected=top(np.ones(101))
        self.assertEqual(selected.sum(),11)
        self.assertTrue(selected[:11].all())

    def test_oof_and_paired_bootstrap(self):
        rng=np.random.default_rng(19)
        n=1800
        t=rng.integers(0,2,n)
        money=rng.uniform(0,10,n)
        probability=.35+.005*money+t*(.02+.04*(money>6))
        y=rng.binomial(1,probability)
        data=pd.DataFrame({"client_id":[str(i) for i in range(n)],
            "treatment_flg":t,"target":y,"count":money,"sum":money*10,
            "recency_days":rng.uniform(1,100,n),"age_value":rng.uniform(18,80,n)})
        oof=crossfit(data,["count","sum","recency_days","age_value"])
        self.assertTrue(oof.uplift_score.notna().all())
        for _,frame in oof.groupby("fold"):
            self.assertEqual(frame.select_uplift.sum(),int(np.ceil(.10*len(frame))))
        summary,folds=reports(oof)
        self.assertIn("monetary",summary.policy.to_list())
        ci=paired_bootstrap(oof,repeats=200)
        self.assertIn("uplift_vs_random",ci.comparison.to_list())
        self.assertIn("uplift_vs_monetary",ci.comparison.to_list())
        self.assertTrue((ci.ci95_low_pp<=ci.ci95_high_pp).all())

if __name__=="__main__":unittest.main()
