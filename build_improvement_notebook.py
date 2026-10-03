"""Create the companion notebook; training is reproducible through improve_sales.py."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
md('''# DSN Mart — improvement experiments and second submission

This companion to `DSN_Mart_Sales_Prediction.ipynb` reports a completed improvement run.
The full, commented training implementation is in `improve_sales.py`, split into numbered-workflow-style
code sections with explanations. No external training data is used.

**Important:** the old holdout has already been inspected. Its score is a diagnostic, not a new
untouched performance estimate. Selection uses the original five development folds. Trying several
models introduces selection optimism. A local improvement does not guarantee a Kaggle improvement.

The original `outputs/submission.csv` is preserved. New outputs are under `outputs/improvement/`.
'''),
code('''from pathlib import Path
import json, runpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle
from IPython.display import display, Markdown, Code
OUT = Path('outputs/improvement')
'''),
md('''## 1. Review and optionally rerun training

The experiment compares the original CatBoost pipeline, two more regularised CatBoost settings,
three shallow histogram boosting models, two ridge-regularised store-specific price splines,
and an equal-weight average of the three best individual candidates. Preprocessing is fitted
inside every training fold. Scores are compared against the original model on identical rows.

The cell below displays the actual training source. The next cell defaults to reading **saved results**
to avoid repeating the multi-minute training run. Set `RERUN_TRAINING = True` to regenerate them.
'''),
code("display(Code(Path('improve_sales.py').read_text(encoding='utf-8'), language='python'))"),
code('''RERUN_TRAINING = False
if RERUN_TRAINING:
    runpy.run_path('improve_sales.py', run_name='__main__')
assert (OUT / 'metrics.json').exists(), 'Set RERUN_TRAINING=True to generate experiment results.'
report = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
comparison = pd.read_csv(OUT / 'comparison.csv')
folds = pd.read_csv(OUT / 'fold_scores.csv')
'''),
md('''## 2. Compare development RMSE

Lower is better. The winner is selected before looking at the diagnostic holdout. The ensemble's
members are also chosen using development results, so its score is a selection estimate.
'''),
code('''display(comparison)
comparison.sort_values('oof_rmse', ascending=False).plot.barh(
    x='model', y='oof_rmse', figsize=(10, 5), legend=False, title='Development out-of-fold RMSE')
plt.xlabel('RMSE (lower is better)')
plt.tight_layout(); plt.show()
paired = folds.pivot(index='fold', columns='model', values='rmse')
winner = report['selected']
check = paired[['Original CatBoost', winner]].copy()
check['improvement'] = paired['Original CatBoost'] - paired[winner]
display(check)
'''),
md('''## 3. Interpret the measured improvement

Check the magnitude and fold consistency rather than assuming the best mean score is decisive.
The diagnostic holdout helps identify a contradictory result, but is not used for further tuning.
'''),
code('''display(pd.Series(report))
display(Markdown(f"""Selected: **{report['selected']}**.
Development RMSE changed from **{report['original_cv_rmse']:,.3f}** to
**{report['cv_rmse']:,.3f}**, improving **{report['folds_improved']} of 5 folds**.
Previously inspected holdout RMSE: **{report['diagnostic_holdout_rmse']:,.3f}**
(original: **{report['original_holdout_rmse']:,.3f}**).
These are local results, not a new leaderboard score."""))
'''),
md('''## 4. Verify the second submission and saved model

The fitted bundle stores one model or an equal-weight ensemble. The verification below loads it
and recomputes all test predictions. The required ID order comes from the sample submission.
Only load this locally created, trusted pickle file.
'''),
code('''test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')
assert report['new_submission_created'], 'No improved candidate; keep the original submission.'
submission = pd.read_csv(OUT / 'submission_improved.csv')
assert submission.columns.tolist() == ['id', 'total_sales']
assert submission.shape == sample.shape and submission.id.equals(sample.id)
assert submission.id.is_unique and np.isfinite(submission.total_sales).all()
assert submission.total_sales.ge(0).all()
with (OUT / 'model_bundle.pkl').open('rb') as f:
    bundle = cloudpickle.load(f)
pred = np.mean([np.maximum(0, m.predict(test[bundle['features']])) for m in bundle['models']], axis=0)
aligned = submission.id.map(pd.Series(pred, index=test.id))
assert np.allclose(submission.total_sales, aligned)
display(submission.head())
print(f'PASS: {len(submission)} predictions; saved model reproduces every value.')
print('Second submission:', (OUT / 'submission_improved.csv').resolve())
'''),
md('''## 5. Submit and compare

Upload `outputs/improvement/submission_improved.csv` as a new Kaggle submission. Keep the original
submission available. Record the new public score and compare it with **1075.95433**, the original
score shown in your screenshot. Do not claim an improvement until Kaggle has scored the new file.
Avoid repeatedly tuning to small public-leaderboard differences; they can reflect sampling noise.
''')]
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nbf.write(nb, 'DSN_Mart_Improvement.ipynb')
print('Created improvement notebook.')
