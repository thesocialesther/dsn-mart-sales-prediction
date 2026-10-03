# DSN Mart sales prediction

Open **DSN_Mart_Sales_Prediction.ipynb** for the step-by-step analysis, comments, charts, model evaluation, and test predictions. It follows the supplied `DSN_Bootcamp_MLTrack (1).pdf`, whose evaluation metric is RMSE.

## Files

- `DSN_Mart_Sales_Prediction.ipynb`: complete notebook; executed outputs are saved after running.
- `outputs/submission.csv`: predictions to upload to the competition (`id,total_sales`).
- `outputs/metrics.json`: selected model, measured scores, seed, and environment versions.
- `outputs/model_comparison.csv` and `fold_scores.csv`: five-fold development comparisons.
- `outputs/unseen_product_scores.csv`: grouped-product stress test.
- `outputs/holdout_predictions.csv` and `permutation_importance.csv`: error and feature diagnostics.
- `outputs/retail_sales_pipeline.pkl`: fitted model including preprocessing; load only trusted pickle files.
- `requirements.txt`: notebook dependencies.
- `requirements-lock.txt`: exact versions used for the delivered run.

## Run

Keep the original three CSVs beside the notebook. In VS Code, select the project's `.venv/Scripts/python.exe` notebook kernel and choose **Run All**. Another Python environment can install `requirements.txt`; use `requirements-lock.txt` with Python 3.14 to reproduce the delivered environment more closely.

For a new environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To execute headlessly in an environment with the dependencies installed:

```powershell
.\.venv\Scripts\python.exe execute_notebook.py
```

The notebook uses 20% of labelled rows as a final holdout. Candidate selection uses five-fold cross-validation on the other 80%. All learned preprocessing is fitted within training folds. After evaluation, the selected pipeline is retrained on all labelled rows to generate test predictions. No test labels or external datasets are used.

`build_notebook.py` is the notebook source generator. Running it replaces the notebook with a fresh unexecuted copy; ordinary users only need to open the existing notebook.

Local validation is not a leaderboard result. The notebook explains limitations and includes discussion points to help you understand and present the work. Upload `outputs/submission.csv` yourself; no online submission is performed by this project.

## Improvement experiments

`DSN_Mart_Improvement.ipynb` is a companion report with model comparisons, fold-level checks, and submission verification. It displays the complete commented training source from `improve_sales.py`. By default it reads saved experiment results; set `RERUN_TRAINING=True` in that notebook to rerun training.

The experiments compare the original model against more regularised CatBoost, histogram boosting, store-specific price splines, and an equal-weight ensemble. Model selection uses the original development folds. The original holdout has already been inspected and is explicitly treated as a diagnostic, not an untouched test.

Completed run: store-specific price splines with ridge alpha 1 won the development comparison (RMSE 1073.64 versus 1080.32 for the original), improving 3 of 5 folds. The diagnostic holdout worsened slightly (1070.02 versus 1067.24), so this is a candidate second submission, not a confirmed general improvement. Its Kaggle score is not yet known.

If an improvement is found, the new prediction file is `outputs/improvement/submission_improved.csv`. The original `outputs/submission.csv` is preserved. The new folder also contains `comparison.csv`, `fold_scores.csv`, `development_oof.csv`, `metrics.json`, and `model_bundle.pkl`. Local improvement does not guarantee a leaderboard improvement.

```powershell
.\.venv\Scripts\python.exe improve_sales.py
.\.venv\Scripts\python.exe execute_notebook.py DSN_Mart_Improvement.ipynb
```

## Normalization check

`DSN_Mart_Normalized.ipynb` is fully executed and tests StandardScaler within training folds, with 12 normalized spline/ridge settings against the previous unscaled model. `normalized_experiments.py` contains the same commented workflow. Three repetitions of five-fold development CV gave mean RMSE 1074.48 for the previous model and 1074.72 for the best normalized candidate. The normalization check did not justify a replacement, so no new submission CSV was created. Use the previous candidate `outputs/improvement/submission_improved.csv` and its matching `submission_description.txt` if submitting that version.

Full normalization comparisons, fold scores, a plot, and metrics are in `outputs/normalized/`. Scaling is fitted only on training rows; sales and RMSE remain in their original units. A zero leaderboard score is not established as attainable from these features, and normalization alone does not guarantee better generalisation.

## Product-history investigation

`DSN_Mart_Product_History.ipynb` is executed and tests whether product sales in other stores explain the remaining errors. It compares training-only shrunk product-residual corrections and CatBoost with explicit product identity against the prior spline baseline. It loads the locally saved prior models, so keep the existing outputs folders alongside the notebook.

The selected correction improved development RMSE only marginally: 1073.31 versus 1073.64. A second split gave 1073.93 versus 1074.39; the previously inspected holdout gave 1069.42. These results do not approach RMSE 50 or establish an irreducible error floor. Selection and repeat diagnostics reuse development data; a leaderboard improvement is unconfirmed.

Candidate: `outputs/product_history/submission_product_history.csv`. Matching description: `outputs/product_history/submission_description.txt`. All previous submissions remain unchanged. The new candidate must not be described as achieving the requested score threshold.

## Target normalization

`DSN_Mart_Target_Normalized.ipynb` is executed and wraps the product-history model in `TransformedTargetRegressor` with `StandardScaler`. Each scaler fits training labels only; predictions automatically return to original sales units before evaluation and export. The notebook loads the trusted local `outputs/product_history/pipeline.pkl`.

Target normalization produced equivalent predictions, with CV RMSE 1073.31 unchanged. It does not provide an expected leaderboard improvement. The demonstration submission is `outputs/target_normalized/submission_target_normalized.csv`; its matching description is `outputs/target_normalized/submission_description.txt`. All 1705 rows and inverse transformations were verified. Previous submissions are preserved.

## Source-code export
Author: Oluwaferanmi Esther Onifade. Competition datasets and fitted pickle models are excluded. Obtain train.csv, test.csv, and sample_submission.csv from your competition access and place them beside the notebooks. Notebook outputs were cleared. Reports that load saved models require those experiments to be rerun; the main sales-prediction notebook is the starting point. Local scores are not verified leaderboard scores.

## Dataset access
See [DATASETS.md](DATASETS.md) for included data, exact file manifests, and setup instructions.
