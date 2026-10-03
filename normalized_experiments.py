# %% [markdown]
# # DSN Mart: normalization and regularisation experiments
# RMSE remains in original sales units. Scaling target values and reporting the scaled
# error would not improve the competition result. Here StandardScaler learns feature
# means and standard deviations from training folds only. Test rows never fit a scaler.
#
# Normalization affects the regularisation of linear models; it does not create missing
# predictive information. A displayed leaderboard zero is not evidence that normalization
# can achieve zero error. We do not use external targets or leaderboard answers.
#
# We compare matched unscaled/scaled price-spline models, then test a small predefined grid
# of spline complexity and ridge strength using three repetitions of five-fold CV. These
# are model-selection estimates, not independent test scores. Previously inspected holdout
# data is used only after selection as a diagnostic. Prior submissions remain unchanged.
# %%
import os
os.environ['OMP_NUM_THREADS'] = '4'
from pathlib import Path
import json
import cloudpickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import root_mean_squared_error
from threadpoolctl import threadpool_limits

OUT = Path('outputs/normalized')
OUT.mkdir(parents=True, exist_ok=True)
train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')
with open('outputs/improvement/model_bundle.pkl', 'rb') as f:
    previous = cloudpickle.load(f)
baseline = previous['models'][0]
assert 'spline' in baseline.named_steps, 'This experiment expects the prior spline model.'
features = previous['features']
X, y = train[features], train.total_sales
Xd, Xh, yd, yh = train_test_split(X, y, test_size=.2, random_state=42)

# %% [markdown]
# ## 1. Define candidates before evaluating
# Scaling applies to the spline and store-interaction design matrix, placing columns on
# comparable scales before Ridge penalises their coefficients. The same scaler is then
# applied to validation and test rows. One candidate changes scaling alone at the prior
# alpha=1 and four knots, isolating its effect. Other candidates retune regularisation
# because the same alpha has a different effect after scaling.
# %%
candidates = {'Previous unscaled spline': clone(baseline)}
for knots in [3, 4, 6]:
    for alpha in [1., 10., 100., 1000.]:
        spline = clone(baseline.named_steps['spline']).set_params(n_knots=knots)
        candidates[f'Normalized spline k{knots} alpha{alpha:g}'] = Pipeline([
            ('features', clone(baseline.named_steps['features'])),
            ('spline', spline), ('normalize', StandardScaler()),
            ('model', Ridge(alpha=alpha))])
seeds = [42, 17, 93]
records, predictions = [], {}

# %% [markdown]
# ## 2. Repeated cross-validation
# Each candidate is fitted 15 times. For each seed we compute RMSE across all held-out
# predictions, then average these three RMSEs for selection. Fold improvements are paired
# against the unscaled reference. Folds/repetitions overlap, so their spread is descriptive,
# not an independent confidence interval. No model is selected using holdout performance.
# %%
with threadpool_limits(limits=4):
    for name, template in candidates.items():
        seed_predictions = []
        for seed in seeds:
            pred = np.zeros(len(Xd))
            for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=seed).split(Xd), 1):
                fitted = clone(template).fit(Xd.iloc[a], yd.iloc[a])
                pred[b] = np.maximum(0, fitted.predict(Xd.iloc[b]))
                records.append({'model': name, 'seed': seed, 'fold': fold,
                    'rmse': float(root_mean_squared_error(yd.iloc[b], pred[b]))})
            seed_predictions.append(pred)
        predictions[name] = np.asarray(seed_predictions)
        print(name, [round(root_mean_squared_error(yd, p), 3) for p in seed_predictions], flush=True)

summary = []
for name, preds in predictions.items():
    scores = [float(root_mean_squared_error(yd, p)) for p in preds]
    summary.append({'model': name, 'mean_repeat_rmse': np.mean(scores),
                    **{f'rmse_seed_{s}': v for s, v in zip(seeds, scores)}})
comparison = pd.DataFrame(summary).sort_values('mean_repeat_rmse').reset_index(drop=True)
winner = comparison.loc[0, 'model']
fold_scores = pd.DataFrame(records)
paired = fold_scores.pivot(index=['seed', 'fold'], columns='model', values='rmse')
gain = paired['Previous unscaled spline'] - paired[winner]
display(comparison)
comparison.to_csv(OUT / 'comparison.csv', index=False)
fold_scores.to_csv(OUT / 'fold_scores.csv', index=False)
np.savez_compressed(OUT / 'oof_predictions.npz', **predictions)
comparison.sort_values('mean_repeat_rmse', ascending=False).plot.barh(
    x='model', y='mean_repeat_rmse', legend=False, figsize=(10, 7), title='Repeated CV RMSE: lower is better')
plt.tight_layout(); plt.savefig(OUT / 'comparison.png'); plt.show()

# %% [markdown]
# ## 3. Verify scaling and inspect the previously seen holdout
# We explicitly verify that the scaler's means match the training-only design matrix.
# The chosen model is fixed before this diagnostic. No target scaling is used, so there
# is no inverse-target transformation to forget when exporting sales predictions.
# %%
best_normalized = comparison[comparison.model.str.startswith('Normalized')].iloc[0]['model']
with threadpool_limits(limits=4):
    normalized_audit = clone(candidates[best_normalized]).fit(Xd, yd)
    design = normalized_audit[:-2].transform(Xd)
    scaler = normalized_audit.named_steps['normalize']
    assert np.allclose(scaler.mean_, design.mean(axis=0))
    assert np.allclose(scaler.transform(design).mean(axis=0), 0, atol=1e-10)
    selected = clone(candidates[winner]).fit(Xd, yd)
    old = clone(baseline).fit(Xd, yd)
    hold_rmse = float(root_mean_squared_error(yh, np.maximum(0, selected.predict(Xh))))
    old_hold_rmse = float(root_mean_squared_error(yh, np.maximum(0, old.predict(Xh))))
    final = clone(candidates[winner]).fit(X, y)
    pred = np.maximum(0, final.predict(test[features]))

# %% [markdown]
# ## 4. Export a candidate only if repeated CV improves
# Preserve all previous files. Align predictions by ID, check the CSV schema, and reload
# the saved pipeline to reproduce every prediction. Every exported submission receives
# a matching ready-to-paste description with measured, original-unit RMSE.
# %%
baseline_score = float(comparison.set_index('model').loc['Previous unscaled spline', 'mean_repeat_rmse'])
best_score = float(comparison.loc[0, 'mean_repeat_rmse'])
improved = winner != 'Previous unscaled spline' and best_score < baseline_score
report = {'selected': winner, 'repeated_cv_rmse': best_score,
    'previous_repeated_cv_rmse': baseline_score, 'cv_improvement': baseline_score - best_score,
    'seed42_rmse': float(comparison.loc[0, 'rmse_seed_42']),
    'folds_improved_out_of_15': int(gain.gt(0).sum()),
    'diagnostic_holdout_rmse': hold_rmse, 'previous_diagnostic_holdout_rmse': old_hold_rmse,
    'normalization_selected': 'normalize' in final.named_steps,
    'new_submission_created': bool(improved), 'seeds': seeds,
    'validation_note': 'Repeated development CV for selection; holdout already inspected; leaderboard gain unconfirmed.'}
with (OUT / 'pipeline.pkl').open('wb') as f:
    cloudpickle.dump(final, f)
with (OUT / 'pipeline.pkl').open('rb') as f:
    restored = cloudpickle.load(f)
assert np.allclose(pred, np.maximum(0, restored.predict(test[features])))
if improved:
    submission = sample[['id']].copy()
    submission['total_sales'] = submission.id.map(pd.Series(pred, index=test.id))
    assert submission.shape == sample.shape and submission.id.equals(sample.id)
    assert submission.id.is_unique and np.isfinite(submission.total_sales).all()
    assert submission.total_sales.ge(0).all()
    submission.to_csv(OUT / 'submission_normalized.csv', index=False)
    description = (f'Standardized store-specific price-spline regression with ridge regularisation. '
        f'Preprocessing fitted within training folds. Selected using three repetitions of five-fold CV; '
        f'mean RMSE {best_score:.2f}. Refit on all labelled data. No external data used.')
    (OUT / 'submission_description.txt').write_text(description, encoding='utf-8')
    display(submission.head())
    print('Submission description:', description)
else:
    print('Normalization did not improve repeated CV; no new submission created.')
(OUT / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
display(pd.Series(report))
display(Markdown('**Interpretation:** compare both validation results above. Normalization is a tested '
    'preprocessing choice, not a guarantee of generalisation or zero RMSE. Small differences may not '
    'persist on Kaggle. The displayed public leaderboard zero has not been independently verified.'))
