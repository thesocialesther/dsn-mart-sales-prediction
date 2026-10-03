"""Check executed notebook, submission alignment, and saved-model predictions."""
from pathlib import Path
import json
import cloudpickle
import nbformat
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
nb = nbformat.read(root / 'DSN_Mart_Sales_Prediction.ipynb', as_version=4)
nbformat.validate(nb)
code_cells = [c for c in nb.cells if c.cell_type == 'code']
assert all(c.execution_count is not None for c in code_cells)
assert not [o for c in code_cells for o in c.outputs if o.output_type == 'error']
charts = sum('image/png' in o.get('data', {}) for c in code_cells for o in c.outputs)
assert charts >= 5
out = root / 'outputs'
report = json.loads((out / 'metrics.json').read_text())
test = pd.read_csv(root / 'test.csv')
sample = pd.read_csv(root / 'sample_submission.csv')
submission = pd.read_csv(out / 'submission.csv')
assert submission.shape == sample.shape == (len(test), 2)
assert submission.columns.tolist() == ['id', 'total_sales']
assert submission.id.equals(sample.id) and submission.id.is_unique
assert np.isfinite(submission.total_sales).all() and submission.total_sales.ge(0).all()
folds = pd.read_csv(out / 'fold_scores.csv')
assert folds.groupby('model').size().eq(5).all()
with (out / 'retail_sales_pipeline.pkl').open('rb') as f:
    model = cloudpickle.load(f)
pred = pd.Series(np.maximum(0, model.predict(test[report['features']])), index=test.id)
assert np.allclose(submission.total_sales, submission.id.map(pred))
print(f'PASS: {len(code_cells)} executed code cells, {charts} charts, {len(submission)} valid predictions.')
print('PASS: model reload in a fresh process reproduces the entire submission.')
print(json.dumps(report, indent=2))
