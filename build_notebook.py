"""Build the documented notebook. Run with Python; no third-party dependencies required."""
import json
from pathlib import Path
from textwrap import dedent

cells = []
def md(source):
    cells.append(dict(cell_type='markdown', metadata={}, source=dedent(source).strip()+'\n'))
def code(source):
    cells.append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=dedent(source).strip()+'\n'))

md('''
# DSN Mart: product–store sales prediction
## DSN AI Bootcamp Qualification Hackathon 2026 — ML Track

This notebook follows **DSN_Bootcamp_MLTrack (1).pdf**: understand the data, analyse sales factors,
clean the data, explore patterns, engineer features, evaluate regression models, and predict test sales.
The target is `total_sales`. The official metric in the supplied brief is **root mean squared error
(RMSE): lower is better**. No external training data is used.

**Run instructions:** Keep this notebook beside `train.csv`, `test.csv`, and `sample_submission.csv`.
Select a Python environment with the libraries in `requirements.txt`, then **Run All**.
The optional installation cell below can be uncommented in a fresh Jupyter/Colab environment.
Training uses the CPU and fixed random seeds; runtime depends on your machine.
Outputs go to `outputs/`, including `submission.csv`, model comparisons, diagnostic tables, and a saved model.

**Validation plan:** reserve 20% of labelled rows before target-based exploration. Explore only the
remaining development data; compare candidates using five identical development folds; select the
lowest cross-validated RMSE; evaluate that choice once on the holdout; then refit on every labelled row.
The holdout is not used for early stopping or selecting hyperparameters. Test labels are unavailable,
and the sample submission's numbers are placeholders, not training labels.
''')
code('''
# Uncomment only when your notebook environment needs the dependencies.
# %pip install -r requirements.txt
''')
md('''## 1. Imports and reproducibility
RMSE penalises large mistakes strongly and is measured in the same units as sales. We also report
MAE (typical absolute error) and R² (improvement over predicting the mean). The brief does not specify
the currency or sales period, so we do not invent those units.
''')
code('''
from pathlib import Path
import json, sys, platform, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from catboost import CatBoostRegressor
import cloudpickle

SEED = 42
DATA_DIR = Path('.')
OUT = DATA_DIR / 'outputs'
OUT.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid', palette='deep')
pd.set_option('display.max_columns', 30)
versions = {p: importlib.metadata.version(p) for p in
            ['numpy', 'pandas', 'scikit-learn', 'catboost', 'matplotlib', 'seaborn', 'cloudpickle']}
print('Python:', platform.python_version())
display(pd.Series(versions, name='installed_version'))
''')
md('''## 2. Load and validate the supplied files
`id` identifies a submission row. `product_code` and `store_code` identify entities; they are not quantities.
Product attributes describe weight, fat content, visibility, category, and price. Outlet attributes describe
age, size, location tier, and format. `store_age_years` is already an age; do not subtract it from a calendar year.
Location tiers are labels; their precise socioeconomic interpretation is not supplied.
''')
code('''
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
TARGET = 'total_sales'
ID = 'id'
expected_features = ['product_code', 'product_weight_kg', 'fat_content', 'shelf_visibility',
    'product_category', 'product_price', 'store_code', 'store_age_years', 'store_size',
    'store_location_tier', 'store_format']
assert set(train.columns) == set([ID, TARGET] + expected_features)
assert set(test.columns) == set([ID] + expected_features)
assert list(sample.columns) == [ID, TARGET]
assert train[ID].is_unique and test[ID].is_unique and sample[ID].is_unique
assert set(train[ID]).isdisjoint(set(test[ID]))
assert set(sample[ID]) == set(test[ID])
assert np.isfinite(train[TARGET]).all() and train[TARGET].ge(0).all()
assert not train.duplicated(['product_code', 'store_code']).any(), 'Reconsider grouped splitting for duplicate pairs.'
print(f'Train: {train.shape}; test: {test.shape}; submission template: {sample.shape}')
display(train.head())
display(pd.DataFrame({'dtype': train.dtypes.astype(str), 'missing_train': train.isna().sum(),
    'missing_test': test.isna().sum(), 'unique_train': train.nunique()}))
print('Duplicate full rows:', train.duplicated().sum())
print('Test products absent from training:', (~test.product_code.isin(train.product_code)).sum())
print('Test stores absent from training:', (~test.store_code.isin(train.store_code)).sum())
''')
md('''## 3. Set aside the final validation set
A shuffled row split matches a snapshot prediction problem with products and stores recurring across
rows. It estimates performance on new product–store combinations in the observed retail network.
It does **not** establish future-period or new-store accuracy. We also run a separate grouped-product
stress test later. `id` and the opaque product identifier are excluded as model predictors; store identity
is retained because the task includes known outlets. Product identity is used only for training-derived
weight/visibility statistics, never target averages.
''')
code('''
X = train[expected_features].copy()
y = train[TARGET].copy()
X_dev, X_hold, y_dev, y_hold = train_test_split(X, y, test_size=0.20, random_state=SEED)
print(f'Development rows: {len(X_dev)}; untouched holdout rows: {len(X_hold)}')
''')
md('''## 4. Clean labels and explore the development data
Category case differences are formatting errors, not distinct categories. Missing weights are retained
until training-only imputation. Missing store sizes become an explicit `unknown` category: guessing an
outlet's size from its sales would leak the target. Zero visibility may be a real value or an unrecorded
measurement, so we preserve it and add an indicator rather than silently treating it as missing.
''')
code('''
categorical_columns = ['product_code', 'fat_content', 'product_category', 'store_code',
                       'store_size', 'store_location_tier', 'store_format']
numeric_columns = ['product_weight_kg', 'shelf_visibility', 'product_price', 'store_age_years']

def clean_labels(frame):
    # This function learns nothing from other rows and is safe before splitting.
    z = frame.copy()
    for col in categorical_columns:
        z[col] = z[col].astype('string').str.strip().str.lower().replace('', pd.NA).fillna('unknown').astype(object)
    z['fat_content'] = z['fat_content'].replace({'lf': 'low fat', 'lowfat': 'low fat', 'reg': 'regular'})
    for col in numeric_columns:
        z[col] = pd.to_numeric(z[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    # Physically invalid measurements become missing; plausible high values remain.
    for col in ['product_weight_kg', 'product_price']:
        z.loc[z[col] <= 0, col] = np.nan
    for col in ['shelf_visibility', 'store_age_years']:
        z.loc[z[col] < 0, col] = np.nan
    return z

eda = clean_labels(X_dev).assign(total_sales=y_dev)
display(eda.describe(include='all').T)
print('Raw category labels:', X_dev.product_category.nunique(),
      '| cleaned category labels:', eda.product_category.nunique())
print('Zero-visibility rows:', int(eda.shelf_visibility.eq(0).sum()))
display(eda.groupby('store_code')[['store_format', 'store_size', 'store_location_tier', 'store_age_years']].nunique())
''')
code('''
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
sns.histplot(eda[TARGET], bins=40, ax=axes[0])
axes[0].set_title('Sales distribution (development only)')
sns.scatterplot(data=eda, x='product_price', y=TARGET, hue='store_format', alpha=.25, s=12, ax=axes[1])
axes[1].set_title('Price and sales by store format')
axes[1].legend(fontsize=7)
sns.scatterplot(data=eda, x='shelf_visibility', y=TARGET, alpha=.2, s=12, ax=axes[2])
axes[2].set_title('Visibility and sales')
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, col in zip(axes, ['store_format', 'store_location_tier', 'store_size']):
    sns.boxplot(data=eda, x=col, y=TARGET, ax=ax, showfliers=False)
    ax.tick_params(axis='x', rotation=35)
    ax.set_title(f'Sales by {col}')
plt.tight_layout(); plt.show()

category_summary = eda.groupby('product_category')[TARGET].agg(['count', 'mean', 'median']).sort_values('mean', ascending=False)
display(category_summary)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
category_summary['mean'].sort_values().plot.barh(ax=axes[0], title='Mean sales by category')
sns.heatmap(eda[numeric_columns + [TARGET]].corr(), annot=True, fmt='.2f', cmap='vlag', center=0, ax=axes[1])
axes[1].set_title('Numeric Pearson correlations')
plt.tight_layout(); plt.show()
display(eda.groupby(['store_format', 'store_location_tier'])[TARGET].agg(['count', 'mean', 'median']))
''')
code('''
format_summary = eda.groupby('store_format')[TARGET].agg(['count', 'mean', 'median']).sort_values('mean', ascending=False)
price_corr = eda[['product_price', TARGET]].corr().iloc[0, 1]
display(Markdown(f"""### Evidence from this development sample
- Mean sales are highest for **{format_summary.index[0]}** ({format_summary.iloc[0]['mean']:,.1f})
  and lowest for **{format_summary.index[-1]}** ({format_summary.iloc[-1]['mean']:,.1f}).
- Price–sales Pearson correlation is **{price_corr:.3f}**. This motivates nonlinear price effects
  and price-by-format interactions; it does not show that raising prices will increase demand.
- There are **{eda.store_code.nunique()} stores** in this sample. Format, age, size, location, and
  store identity overlap, so their effects cannot be interpreted as independent causal effects.
- Compare category means with their counts and medians; small categories can give unstable averages.
- High sales are retained: RMSE is sensitive to large errors and the brief does not identify outliers as mistakes.
"""))
''')
md('''## 5. Engineer features without leakage
Every pipeline fits its own transformer on its training rows. Weight filling uses the product median,
then category median, then global median. Other numeric missing values use training medians. The model
also receives missingness indicators, zero visibility, price per kilogram, log price/visibility,
price relative to its category, visibility relative to its product, and category–format/format–tier
interactions. Ratios use protected denominators. No feature uses sales, row-number patterns, or test-set statistics.

Tree models learn further nonlinear interactions. These engineered features are hypotheses whose value
must be checked by validation, not assumed from business intuition alone.
''')
code('''
class RetailFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, engineered=True):
        self.engineered = engineered

    def fit(self, X, y=None):
        z = clean_labels(X)
        self.product_weight_ = z.groupby('product_code')['product_weight_kg'].median()
        self.category_weight_ = z.groupby('product_category')['product_weight_kg'].median()
        self.medians_ = z[numeric_columns].median().fillna(0)
        self.category_price_ = z.groupby('product_category')['product_price'].median()
        self.product_visibility_ = z.groupby('product_code')['shelf_visibility'].mean()
        return self

    def transform(self, X):
        z = clean_labels(X)
        for col in numeric_columns:
            z[col + '_missing'] = z[col].isna().astype(int)
        z['product_weight_kg'] = (z.product_weight_kg
            .fillna(z.product_code.map(self.product_weight_))
            .fillna(z.product_category.map(self.category_weight_)))
        z[numeric_columns] = z[numeric_columns].fillna(self.medians_)
        if self.engineered:
            z['visibility_zero'] = z.shelf_visibility.eq(0).astype(int)
            z['log_price'] = np.log1p(z.product_price)
            z['log_visibility'] = np.log1p(z.shelf_visibility)
            z['price_per_kg'] = z.product_price / z.product_weight_kg.clip(lower=.1)
            category_price = z.product_category.map(self.category_price_).fillna(self.medians_['product_price'])
            z['relative_category_price'] = z.product_price / category_price.clip(lower=.1)
            product_visibility = z.product_code.map(self.product_visibility_).fillna(self.medians_['shelf_visibility'])
            z['relative_product_visibility'] = z.shelf_visibility / product_visibility.clip(lower=.001)
            z['category_format'] = z.product_category + ' | ' + z.store_format
            z['format_tier'] = z.store_format + ' | ' + z.store_location_tier
        # Keep categorical dtypes explicit across pandas versions, including pandas 3.
        for col in categorical_columns + (['category_format', 'format_tier'] if self.engineered else []):
            z[col] = z[col].astype(object)
        return z.drop(columns='product_code')

feature_preview = RetailFeatures().fit(X_dev).transform(X_dev)
assert not feature_preview.isna().any().any()
display(feature_preview.head())
''')
md('''## 6. Compare baseline, linear, and tree models
The mean baseline establishes a minimum standard. Ridge uses scaled numeric values and one-hot
categoricals. Extra Trees captures nonlinearities. CatBoost handles categorical features natively.
We try a small, predefined set of CatBoost settings and a raw-feature ablation (same imputation, fewer
engineered features). All candidates see identical five-fold splits. There is no holdout-driven tuning.
Predictions are clipped at zero because the target is nonnegative; this cannot worsen squared error
for a nonnegative true value. Models optimise sales on the original scale, matching RMSE.
''')
code('''
encoder = ColumnTransformer([
    ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]),
     make_column_selector(dtype_include=np.number)),
    ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
     make_column_selector(dtype_include=object)),
])
def numeric_pipeline(model):
    return Pipeline([('features', RetailFeatures()), ('encode', clone(encoder)), ('model', model)])

def cat_pipeline(depth, iterations, l2, engineered=True):
    categoricals = ['fat_content', 'product_category', 'store_code', 'store_size',
                    'store_location_tier', 'store_format']
    if engineered:
        categoricals += ['category_format', 'format_tier']
    return Pipeline([('features', RetailFeatures(engineered=engineered)), ('model', CatBoostRegressor(
        iterations=iterations, depth=depth, learning_rate=.035, l2_leaf_reg=l2,
        loss_function='RMSE', cat_features=tuple(categoricals), random_seed=SEED,
        verbose=False, allow_writing_files=False, thread_count=4))])

candidates = {
    'Mean baseline': numeric_pipeline(DummyRegressor(strategy='mean')),
    'Ridge': numeric_pipeline(Ridge(alpha=25)),
    'Extra Trees': numeric_pipeline(ExtraTreesRegressor(n_estimators=350, min_samples_leaf=8,
                         max_features=.85, random_state=SEED, n_jobs=4)),
    'CatBoost depth 4': cat_pipeline(4, 650, 8),
    'CatBoost depth 5': cat_pipeline(5, 900, 12),
    'CatBoost depth 6': cat_pipeline(6, 650, 15),
    'CatBoost basic features': cat_pipeline(4, 650, 8, engineered=False),
}
folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X_dev))
records, oof_predictions = [], {}
for name, template in candidates.items():
    oof = np.empty(len(X_dev))
    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        model = clone(template)
        model.fit(X_dev.iloc[fit_idx], y_dev.iloc[fit_idx])
        pred = np.maximum(0, model.predict(X_dev.iloc[val_idx]))
        oof[val_idx] = pred
        records.append({'model': name, 'fold': fold,
            'rmse': root_mean_squared_error(y_dev.iloc[val_idx], pred),
            'mae': mean_absolute_error(y_dev.iloc[val_idx], pred)})
    oof_predictions[name] = oof
    print(f'{name}: OOF RMSE = {root_mean_squared_error(y_dev, oof):,.3f}', flush=True)

fold_scores = pd.DataFrame(records)
comparison = fold_scores.groupby('model').agg(mean_rmse=('rmse', 'mean'),
    std_rmse=('rmse', 'std'), mean_mae=('mae', 'mean'))
comparison['oof_rmse'] = pd.Series({name: root_mean_squared_error(y_dev, p) for name, p in oof_predictions.items()})
comparison = comparison.sort_values('oof_rmse')
best_name = comparison.index[0]
display(comparison)
comparison.to_csv(OUT / 'model_comparison.csv')
fold_scores.to_csv(OUT / 'fold_scores.csv', index=False)
print('Selected exclusively by development OOF RMSE:', best_name)
''')
md('''## 7. Stress-test generalisation to unseen products
GroupKFold keeps each product entirely on one side of a fold. This is a secondary diagnostic for the
already-selected model; it is not used to reselect the winner. Store overlap is still allowed. It does
not measure new-store performance, and its score is not directly comparable to the competition leaderboard.
''')
code('''
group_scores = []
for fold, (fit_idx, val_idx) in enumerate(GroupKFold(n_splits=3).split(X_dev, y_dev, groups=X_dev.product_code), 1):
    assert set(X_dev.iloc[fit_idx].product_code).isdisjoint(X_dev.iloc[val_idx].product_code)
    model = clone(candidates[best_name]).fit(X_dev.iloc[fit_idx], y_dev.iloc[fit_idx])
    p = np.maximum(0, model.predict(X_dev.iloc[val_idx]))
    group_scores.append({'fold': fold, 'rmse': root_mean_squared_error(y_dev.iloc[val_idx], p)})
group_scores = pd.DataFrame(group_scores)
display(group_scores)
print('Unseen-product mean RMSE:', group_scores.rmse.mean())
group_scores.to_csv(OUT / 'unseen_product_scores.csv', index=False)
''')
md('''## 8. One final holdout evaluation and error analysis
The selected pipeline now trains on all development rows. We inspect errors after selection and do
not use these results to tune it. If a future iteration responds to these diagnostics, this holdout
will no longer be an untouched estimate and a new evaluation design will be needed.
''')
code('''
selected = clone(candidates[best_name]).fit(X_dev, y_dev)
hold_pred = np.maximum(0, selected.predict(X_hold))
hold_metrics = {'rmse': float(root_mean_squared_error(y_hold, hold_pred)),
                'mae': float(mean_absolute_error(y_hold, hold_pred)),
                'r2': float(r2_score(y_hold, hold_pred))}
baseline_rmse = root_mean_squared_error(y_hold, np.full(len(y_hold), y_dev.mean()))
display(pd.Series(hold_metrics, name='holdout'))
print(f'Mean baseline holdout RMSE: {baseline_rmse:,.3f}')
print(f'RMSE improvement over mean: {100 * (1 - hold_metrics["rmse"] / baseline_rmse):.1f}%')

errors = clean_labels(X_hold).assign(actual=y_hold, predicted=hold_pred)
errors['residual'] = errors.actual - errors.predicted  # Positive = underprediction.
errors['squared_error'] = errors.residual ** 2
errors['absolute_error'] = errors.residual.abs()
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.scatterplot(data=errors, x='actual', y='predicted', alpha=.35, ax=axes[0])
limit = max(errors.actual.max(), errors.predicted.max())
axes[0].plot([0, limit], [0, limit], 'r--'); axes[0].set_title('Actual versus predicted')
sns.scatterplot(data=errors, x='predicted', y='residual', alpha=.35, ax=axes[1])
axes[1].axhline(0, color='red', linestyle='--'); axes[1].set_title('Residuals versus prediction')
sns.histplot(errors.residual, bins=40, ax=axes[2]); axes[2].set_title('Residual distribution')
plt.tight_layout(); plt.show()
for col in ['store_format', 'store_location_tier', 'product_category']:
    segment = errors.groupby(col).agg(count=('actual', 'size'), mean_sales=('actual', 'mean'),
        mse=('squared_error', 'mean'), mae=('absolute_error', 'mean'), bias=('residual', 'mean'))
    segment['rmse'] = np.sqrt(segment.pop('mse'))
    display(segment.sort_values('rmse', ascending=False))
errors.assign(id=train.loc[X_hold.index, ID]).to_csv(OUT / 'holdout_predictions.csv', index=False)
''')
md('''## 9. Interpret predictive factors cautiously
Permutation importance shuffles one raw input at a time on the holdout and measures the increase in
RMSE. Larger positive increases indicate useful predictive information; negative values can arise from
noise. Correlated store attributes can substitute for each other and dilute importance. These results
explain model dependence, not causal effects or expected returns on store investment.
''')
code('''
importance = permutation_importance(selected, X_hold, y_hold, scoring='neg_root_mean_squared_error',
    n_repeats=5, random_state=SEED, n_jobs=1)
importance_table = pd.DataFrame({'feature': X_hold.columns, 'rmse_increase': importance.importances_mean,
    'std': importance.importances_std}).sort_values('rmse_increase', ascending=False)
display(importance_table)
importance_table.sort_values('rmse_increase').plot.barh(x='feature', y='rmse_increase', xerr='std',
    figsize=(9, 6), legend=False, title='Holdout permutation importance: increase in RMSE')
plt.tight_layout(); plt.show()
importance_table.to_csv(OUT / 'permutation_importance.csv', index=False)
''')
md('''## 10. Refit on all labelled rows and generate the submission
Once evaluation is complete, all labelled rows can improve the final fit. Test data is transformed
using statistics learned from training only. We join predictions by `id` to preserve the sample
submission's exact order, and check shape, columns, uniqueness, numeric finiteness, and nonnegativity.
The CSV has no index column. Nothing is submitted online automatically.
''')
code('''
final_model = clone(candidates[best_name]).fit(X, y)
test_prediction = np.maximum(0, final_model.predict(test[expected_features]))
by_id = pd.Series(test_prediction, index=test[ID], name=TARGET)
submission = sample[[ID]].copy()
submission[TARGET] = submission[ID].map(by_id)
assert list(submission.columns) == list(sample.columns)
assert len(submission) == len(test)
assert submission[ID].equals(sample[ID]) and submission[ID].is_unique
assert np.isfinite(submission[TARGET]).all() and submission[TARGET].ge(0).all()
submission.to_csv(OUT / 'submission.csv', index=False)
roundtrip = pd.read_csv(OUT / 'submission.csv')
assert roundtrip.shape == sample.shape and roundtrip[ID].equals(sample[ID])
assert np.allclose(roundtrip[TARGET], test_prediction[test.set_index(ID).index.get_indexer(sample[ID])])
display(submission.head(10))
display(submission[TARGET].describe())
print('Validated submission:', (OUT / 'submission.csv').resolve())

# Cloudpickle captures the notebook-defined transformer along with the fitted pipeline.
# Only load model files you trust; use the recorded package versions for compatibility.
with (OUT / 'retail_sales_pipeline.pkl').open('wb') as f:
    cloudpickle.dump(final_model, f)
with (OUT / 'retail_sales_pipeline.pkl').open('rb') as f:
    restored = cloudpickle.load(f)
assert np.allclose(final_model.predict(test[expected_features].head()), restored.predict(test[expected_features].head()))
report = {'selected_model': best_name, 'development_oof_rmse': float(comparison.loc[best_name, 'oof_rmse']),
          'holdout_metrics': hold_metrics, 'holdout_baseline_rmse': float(baseline_rmse),
          'unseen_product_mean_rmse': float(group_scores.rmse.mean()),
          'seed': SEED, 'train_rows': len(train), 'test_rows': len(test),
          'features': expected_features, 'versions': versions, 'python': platform.python_version()}
(OUT / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
''')
md('''## 11. Communicate results and limitations
''')
code('''
top_features = ', '.join(importance_table.head(3).feature)
display(Markdown(f"""**Selected model:** {best_name}. Development out-of-fold RMSE:
**{comparison.loc[best_name, 'oof_rmse']:,.2f}**; final holdout RMSE: **{hold_metrics['rmse']:,.2f}**;
MAE: **{hold_metrics['mae']:,.2f}**; R²: **{hold_metrics['r2']:.3f}**.

The strongest raw-input permutation signals in this run are **{top_features}**.
Use predictions to prioritise stock-review conversations and compare expected sales across the observed
product–store combinations. Sales value is not unit demand; ordering quantities also require inventory,
unit sales, lead times, margins, and stockout information.

**Limits:** this snapshot has no dates, promotions, or stockout records. It cannot establish seasonal
trends, future accuracy, or causal price effects. Only ten outlets are represented in the full supplied
training file, so store-level patterns may not generalise to new locations. The grouped-product check
tests one additional generalisation scenario; it does not remove these limitations. Cross-validation
variation is descriptive, not a confidence interval. Local scores are not leaderboard scores, and no
rank or bootcamp selection is guaranteed.

**Deliverable:** `outputs/submission.csv` contains {len(submission):,} predictions and the exact required
`id,total_sales` columns. Review this notebook and upload that CSV yourself to the competition.
"""))
''')
md('''### Application discussion checklist
- Explain why RMSE, original-scale training, and a mean baseline are appropriate.
- Walk through missing-value handling and why it happens inside each fold.
- Explain the row split and what the grouped-product stress test measures.
- Compare the engineered and basic CatBoost variants using the actual results above.
- Discuss underprediction of unusually high sales and the distinction between association and causation.
- Describe any assistance used accurately and follow the competition's applicable submission rules.

**Source:** the local competition brief `DSN_Bootcamp_MLTrack (1).pdf` and the three supplied CSV files.
''')

notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
    language_info=dict(name='python', version='3.14')), nbformat=4, nbformat_minor=5)
for i, cell in enumerate(cells):
    cell['id'] = f'cell-{i:03d}'
Path('DSN_Mart_Sales_Prediction.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print(f'Created notebook with {len(cells)} cells.')
