"""Execute and save all notebook outputs using the current project interpreter."""
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
for key, folder in [('JUPYTER_RUNTIME_DIR', '.jupyter/runtime'),
                    ('IPYTHONDIR', '.jupyter/ipython'), ('MPLCONFIGDIR', '.jupyter/matplotlib')]:
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
path = root / (sys.argv[1] if len(sys.argv) > 1 else 'DSN_Mart_Sales_Prediction.ipynb')
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
client = NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(root)}})
def progress(cell, cell_index, **kwargs):
    print(f'Executing cell {cell_index}: {cell.source.splitlines()[0][:95]}', flush=True)
client.on_cell_execute = progress
def checkpoint(cell, cell_index, **kwargs):
    nbformat.write(nb, path)
client.on_cell_executed = checkpoint
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print('Notebook execution complete.', flush=True)
