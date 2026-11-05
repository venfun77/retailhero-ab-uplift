# Cross-fitting: historical retrospective evaluation

This is not a new A/B experiment or an untouched test; the dataset was previously inspected during feature/model selection. Each client's uplift score was fitted using other folds. Intervals condition on fixed scores and do not capture retraining or past model selection. Baselines: random, originally prespecified high spending; low spending was introduced AFTER reviewing this dataset and is exploratory, not a confirmatory test.

## Policy summary

fold       policy  selected  uplift_pp  gain_vs_random_pp
 all       random    200039      3.323              0.000
 all       uplift     20004      7.263              3.940
 all    frequency     20004      0.769             -2.554
 all     monetary     20004      1.461             -1.862
 all      recency     20004      2.206             -1.117
 all    rfm_score     20004      0.976             -2.347
 all low_spending     20004      7.251              3.928

## Bootstrap CI

            comparison  median_gain_pp  ci95_low_pp  ci95_high_pp  ci_contains_zero
      uplift_vs_random           3.951        2.707         5.223             False
   frequency_vs_random          -2.547       -3.301        -1.792             False
    monetary_vs_random          -1.860       -2.865        -0.837             False
     recency_vs_random          -1.120       -2.147        -0.129             False
   rfm_score_vs_random          -2.345       -3.046        -1.655             False
low_spending_vs_random           3.916        2.642         5.191             False
    uplift_vs_monetary           5.817        4.126         7.441             False
   uplift_vs_rfm_score           6.299        4.789         7.817             False
uplift_vs_low_spending           0.038       -1.483         1.594              True

## Per-fold stability

fold       policy  selected  uplift_pp  gain_vs_random_pp
   1       random     66680      3.324              0.000
   1       uplift      6668      5.994              2.670
   1    frequency      6668      0.365             -2.959
   1     monetary      6668      1.558             -1.766
   1      recency      6668      1.509             -1.814
   1    rfm_score      6668      1.431             -1.893
   1 low_spending      6668      7.043              3.719
   2       random     66680      3.324              0.000
   2       uplift      6668      8.835              5.511
   2    frequency      6668      0.793             -2.530
   2     monetary      6668      1.045             -2.278
   2      recency      6668      1.282             -2.042
   2    rfm_score      6668      0.211             -3.113
   2 low_spending      6668      7.141              3.817
   3       random     66679      3.322              0.000
   3       uplift      6668      7.040              3.718
   3    frequency      6668      1.147             -2.175
   3     monetary      6668      1.775             -1.547
   3      recency      6668      3.881              0.559
   3    rfm_score      6668      1.318             -2.004
   3 low_spending      6668      7.582              4.260

