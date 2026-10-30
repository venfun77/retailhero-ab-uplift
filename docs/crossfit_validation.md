# 3-fold cross-fitting: independent-of-training client evaluation

**Status:** implementation ready. No numeric results from this protocol have been computed on the full RetailHero dataset yet. This is retrospective validation, NOT a new A/B test and NOT a fully untouched test set.

## Predefined comparisons
- Random 10% of customers: baseline expected uplift equals the ATE in the historical A/B experiment.
- T-learner: top 10% selected by predicted uplift.
- Primary simple benchmark: top 10% by **pre-campaign historical spending**.
- Exploratory comparisons: highest frequency, most recent purchase, and composite RFM rank.

## Protocol
1. StratifiedKFold with **3** folds, fixed seed **20261030**, strata are historical treatment and binary outcome.
2. For each fold, fit separate control and treatment HistGradientBoostingClassifier models only on the other two folds; imputation medians fitted inside the training fold.
3. No customer target is used to train the models scoring that customer. Predictors are age/gender and pre-campaign RFM features; IDs, treatment and target are excluded.
4. Within each validation fold, select top 10% by model score and by each simple baseline.
5. Measure the **difference in conversion of treated vs control customers within each selected audience**, not the difference in raw conversion between two different audiences.
6. Compare uplift-minus-random and uplift-minus-spending, plus descriptive baselines, via a **paired nonparametric bootstrap stratified by fold** (2000 replications). Report 95% percentile intervals and effects separately for each fold.
7. Freeze model parameters and ranking rules before looking at the new results. Do not claim a robust advantage if interval contains zero or if simple targeting is comparable.

## Limits
Although every customer is scored out of fold, the dataset already informed feature/model development in earlier iterations. Cross-fitting does not erase prior adaptive feature selection or provide a genuinely independent final test. The bootstrap resamples clients with scores fixed and does not account for uncertainty due to retraining. A new experiment randomizing customers between targeting *policies* remains the gold-standard confirmation.

## Reproduce
Run the new notebook at notebooks/RetailHero_Crossfit_Validation.ipynb. It reuses /content/data/final_stats.csv if available and otherwise builds RFM from the original large purchase history. The output ZIP contains policy_summary.csv, bootstrap_intervals.csv, fold_results.csv and REPORT.md. Actual metrics will be added to README and CV only after the run.
