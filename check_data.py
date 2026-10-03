import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
for item in json.loads((root / 'dataset-manifest.json').read_text()):
    path = root / item['file']
    if not path.exists():
        raise SystemExit('Download missing competition file: ' + item['file'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        raise SystemExit('Dataset version differs: ' + item['file'])
    print('OK:', item['file'])
