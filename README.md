# DSN Mart sales prediction

**Author: Oluwaferanmi Esther Onifade**

A machine-learning workflow for the DSN Bootcamp 2026 ML qualification task: predicting retail product sales from product and store information. The project covers data checks, preprocessing, model comparison, error analysis, and competition prediction generation.

## Data and methodology

The task contains 6,818 labelled training rows and 1,705 test rows. Inputs describe product identity, weight, fat content, shelf visibility, category, price, and store identity, age, size, location tier, and format. The target is `total_sales`; evaluation uses root mean squared error (RMSE).

The main workflow reserves 20% of labelled rows for holdout evaluation and compares models using five-fold cross-validation on the remaining development rows. Learned preprocessing is fitted within training folds. After evaluation, the selected pipeline is fitted on all labelled rows to generate test predictions.

## Included work

- `DSN_Mart_Sales_Prediction.ipynb`: main analysis and training workflow.
- Companion notebooks and scripts: spline/ridge improvements, feature normalization, product-history features, and target normalization.
- `outputs/`: saved metrics, model-comparison tables, and experiment reports.
- `check_data.py`: input-data checks.
- `execute_notebook.py`: headless notebook execution.
- `requirements.txt` and `requirements-lock.txt`: dependencies and recorded versions.
- [DATASETS.md](DATASETS.md): official dataset access and file verification.

Notebook outputs were cleared for export. Competition CSVs, fitted pickle models, and submission predictions are not included in the current source export. Some companion analyses load model bundles produced by earlier experiments.

## Setup and run

Download `train.csv`, `test.csv`, and `sample_submission.csv` through your competition access, accepting the applicable competition terms. Place them beside the notebooks.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_data.py
.\.venv\Scripts\python.exe execute_notebook.py
~~~

Alternatively, select that environment as the notebook kernel and run the main notebook. The lock file records the original Python 3.14 environment. Run the main workflow before companion experiments that depend on its generated artifacts. `build_notebook.py` regenerates the main notebook and replaces its existing contents.

## Recorded results

The saved baseline report selects CatBoost: development out-of-fold RMSE **1080.32**, holdout RMSE **1067.24**, holdout MAE **760.89**, and R² **0.6131**. A mean-prediction baseline has holdout RMSE **1716.87**.

Later experiments achieved development RMSE 1073.64 with store-specific price splines and 1073.31 with a product-history correction. These experiments reused previously inspected data for diagnostics; their results do not establish improvement on an untouched final test set or the competition leaderboard.

To complete the reproducible release, add the chosen final model bundle, matching submission file, and any verified leaderboard evidence. Keep each exported submission paired with its producing model and experiment.
