"""Convert the commented experiment sections into a runnable notebook."""
from pathlib import Path
import nbformat as nbf
source = Path('normalized_experiments.py').read_text(encoding='utf-8')
cells = []
for section in source.split('# %%'):
    if not section.strip():
        continue
    if section.startswith(' [markdown]'):
        lines = section.splitlines()[1:]
        cells.append(nbf.v4.new_markdown_cell('\n'.join(line[2:] if line.startswith('# ') else line[1:] if line.startswith('#') else line for line in lines).strip()))
    else:
        cells.append(nbf.v4.new_code_cell(section.strip()))
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'display_name':'Python 3','language':'python','name':'python3'}
nbf.write(nb, 'DSN_Mart_Normalized.ipynb')
