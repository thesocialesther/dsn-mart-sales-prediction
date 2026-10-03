# %% [markdown]
# # DSN Mart: model improvement experiments
# This follow-up preserves the original submission and compares simpler, more strongly
# regularised models with the original CatBoost pipeline. All experiments use the original
# development partition and the same five folds. The previously inspected holdout is only
# a diagnostic. Candidate selection scores are optimistic after trying multiple models;
# a lower local RMSE does not guarantee a higher Kaggle rank.
# No external data or leaderboard labels are used. Run beside the original CSV files.
# %%
import os
os.environ['OMP_NUM_THREADS'] = '4'
from pathlib import Path
import json
import cloudpickle
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
from threadpoolctl import threadpool_limits

SEED = 42
OUT = Path('outputs/improvement')
OUT.mkdir(parents=True, exist_ok=True)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')
with open('outputs/retail_sales_pipeline.pkl', 'rb') as f:
    original = cloudpickle.load(f)
features = [c for c in test if c != 'id']
X, y = train[features], train.total_sales
Xd, Xh, yd, yh = train_test_split(X, y, test_size=.2, random_state=SEED)
folds = list(KFold(5, shuffle=True, random_state=SEED).split(Xd))
print('Development:', len(Xd), '| previously inspected holdout:', len(Xh), flush=True)

# %% [markdown]
# ## Hypotheses and candidate models
# Store format and price dominate this problem. Shallow boosting and smooth price curves
# may generalise better than detailed product-level splits. The spline model fits a smooth
# price-to-sales curve for each store with ridge regularisation. It does not divide the
# target by price, change the RMSE objective, or use target-derived features.
# HistGradientBoosting tests another boosting implementation. Category encoders are fitted
# within each fold. Unknown categorical values are handled without consulting test data.
# %%
class StorePriceSpline(TransformerMixin, BaseEstimator):
    def __init__(self, n_knots=4):
        self.n_knots = n_knots

    def fit(self, X, y=None):
        self.spline_ = SplineTransformer(n_knots=self.n_knots, degree=2,
            include_bias=True, extrapolation='linear').fit(X[['product_price']])
        self.stores_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(X[['store_code']])
        return self

    def transform(self, X):
        price = self.spline_.transform(X[['product_price']])
        store = self.stores_.transform(X[['store_code']])
        # Global price curve provides a fallback; store-specific curves capture interactions.
        return np.column_stack([price, (price[:, :, None] * store[:, None, :]).reshape(len(X), -1)])

cat_cols = ['fat_content', 'product_category', 'store_code', 'store_size', 'store_location_tier', 'store_format']
num_cols = ['product_weight_kg', 'shelf_visibility', 'product_price', 'store_age_years']
encode = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)])
candidates = {'Original CatBoost': clone(original)}
for depth, iterations, l2 in [(3, 450, 20), (4, 350, 30)]:
    p = clone(original)
    p.set_params(model__depth=depth, model__iterations=iterations, model__l2_leaf_reg=l2,
                 model__thread_count=4)
    candidates[f'CatBoost d{depth} n{iterations}'] = p
for depth, iterations in [(2, 250), (3, 200), (4, 150)]:
    candidates[f'Histogram d{depth}'] = Pipeline([
        ('features', clone(original.named_steps['features'])), ('encode', clone(encode)),
        ('model', HistGradientBoostingRegressor(max_depth=depth, max_iter=iterations,
            learning_rate=.04, l2_regularization=20, min_samples_leaf=35,
            early_stopping=False, random_state=SEED))])
for alpha in [1., 10.]:
    candidates[f'Store-price spline alpha {alpha}'] = Pipeline([
        ('features', clone(original.named_steps['features'])),
        ('spline', StorePriceSpline()), ('model', Ridge(alpha=alpha))])

# %% [markdown]
# ## Five-fold comparison
# Each validation row is predicted by a model that never saw that row's target.
# The additional ensemble is a predefined equal average of the three best individual
# candidates. Its member selection uses development results, so its OOF score is a
# model-selection estimate, not an unbiased final test. No holdout-driven selection occurs.
# %%
oof, fold_records = {}, []
with threadpool_limits(limits=4):
    for name, candidate in candidates.items():
        prediction = np.zeros(len(Xd))
        for fold, (a, b) in enumerate(folds, 1):
            fitted = clone(candidate).fit(Xd.iloc[a], yd.iloc[a])
            prediction[b] = np.maximum(0, fitted.predict(Xd.iloc[b]))
            score = root_mean_squared_error(yd.iloc[b], prediction[b])
            fold_records.append({'model': name, 'fold': fold, 'rmse': float(score)})
            print(f'{name} | fold {fold}: {score:.3f}', flush=True)
        oof[name] = prediction
        print(f'OOF {name}: {root_mean_squared_error(yd, prediction):.3f}', flush=True)

scores = {name: float(root_mean_squared_error(yd, p)) for name, p in oof.items()}
top_three = sorted(scores, key=scores.get)[:3]
blend_name = 'Equal blend of top three'
oof[blend_name] = np.mean([oof[name] for name in top_three], axis=0)
scores[blend_name] = float(root_mean_squared_error(yd, oof[blend_name]))
for fold, (_, b) in enumerate(folds, 1):
    fold_records.append({'model': blend_name, 'fold': fold,
        'rmse': float(root_mean_squared_error(yd.iloc[b], oof[blend_name][b]))})
comparison = pd.DataFrame({'model': list(scores), 'oof_rmse': list(scores.values())}).sort_values('oof_rmse')
winner = comparison.iloc[0]['model']
members = top_three if winner == blend_name else [winner]
fold_table = pd.DataFrame(fold_records)
display(comparison)
comparison.to_csv(OUT / 'comparison.csv', index=False)
fold_table.to_csv(OUT / 'fold_scores.csv', index=False)
pd.DataFrame(oof, index=Xd.index).assign(actual=yd).to_csv(OUT / 'development_oof.csv', index_label='train_index')
paired = fold_table.pivot(index='fold', columns='model', values='rmse')
paired['improvement_over_original'] = paired['Original CatBoost'] - paired[winner]
display(paired[['Original CatBoost', winner, 'improvement_over_original']])
print('Selected:', winner, '| members:', members, flush=True)

# %% [markdown]
# ## Diagnostic evaluation, final fit, and submission checks
# The winner is fixed before inspecting these diagnostic results. We do not tune in response
# to them. The final ensemble, if selected, uses the same member weights as validation.
# The original `outputs/submission.csv` is never overwritten.
# %%
with threadpool_limits(limits=4):
    hold_models = [clone(candidates[name]).fit(Xd, yd) for name in members]
    hold_prediction = np.mean([np.maximum(0, m.predict(Xh)) for m in hold_models], axis=0)
    hold_rmse = float(root_mean_squared_error(yh, hold_prediction))
    print(f'Previously inspected holdout RMSE: {hold_rmse:.3f}', flush=True)
    improved = scores[winner] < scores['Original CatBoost']
    final_models = [clone(candidates[name]).fit(X, y) for name in members]
    prediction = np.mean([np.maximum(0, m.predict(test[features])) for m in final_models], axis=0)

submission = sample[['id']].copy()
submission['total_sales'] = submission.id.map(pd.Series(prediction, index=test.id))
assert len(submission) == len(test) and submission.id.equals(sample.id)
assert submission.id.is_unique and np.isfinite(submission.total_sales).all()
assert submission.total_sales.ge(0).all()
if improved:
    submission.to_csv(OUT / 'submission_improved.csv', index=False)
else:
    print('No CV improvement: retain original submission.', flush=True)
with (OUT / 'model_bundle.pkl').open('wb') as f:
    cloudpickle.dump({'models': final_models, 'members': members, 'features': features}, f)
report = {'selected': winner, 'members': members, 'cv_rmse': scores[winner],
    'original_cv_rmse': scores['Original CatBoost'],
    'cv_improvement': scores['Original CatBoost'] - scores[winner],
    'folds_improved': int(paired.improvement_over_original.gt(0).sum()),
    'diagnostic_holdout_rmse': hold_rmse,
    'diagnostic_holdout_mae': float(mean_absolute_error(yh, hold_prediction)),
    'original_holdout_rmse': 1067.239452009752, 'new_submission_created': bool(improved),
    'note': 'Previously inspected holdout; no claimed untouched test or guaranteed leaderboard gain.'}
(OUT / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
display(pd.Series(report))
display(submission.head())
print('Finished. Original submission preserved.', flush=True)
