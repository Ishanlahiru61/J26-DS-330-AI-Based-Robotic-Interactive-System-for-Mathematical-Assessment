"""Record reviewable source changes without accessing Git."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs' / 'integration_evidence'
IGNORE = {'.venv', '__pycache__', 'integration_evidence', 'runtime'}

def snapshot():
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ROOT.rglob('*') if p.is_file() and not IGNORE.intersection(p.relative_to(ROOT).parts)
            and p.suffix not in {'.db', '.sqlite', '.sqlite3', '.pyc'}}

OUTPUT.mkdir(parents=True, exist_ok=True)
before = OUTPUT / 'source_before.json'
if sys.argv[1:] == ['baseline']:
    if before.exists():
        raise SystemExit('Baseline already exists')
    before.write_text(json.dumps(snapshot(), indent=2), encoding='utf-8')
else:
    old = json.loads(before.read_text(encoding='utf-8'))
    new = snapshot()
    changed = {'new': sorted(new.keys()-old.keys()),
               'modified': sorted(k for k in old.keys() & new.keys() if old[k] != new[k]),
               'removed': sorted(old.keys()-new.keys())}
    # This helper was created to capture the baseline, before any implementation edits.
    helper = 'tools/change_inventory.py'
    changed['new'] = sorted(set(changed['new']) | {helper})
    changed['modified'] = [name for name in changed['modified'] if name != helper]
    changed['generated_evidence'] = sorted(str(p.relative_to(ROOT)).replace('\\', '/')
                                           for p in OUTPUT.iterdir() if p.is_file())
    (OUTPUT / 'changed_files.json').write_text(json.dumps(changed, indent=2), encoding='utf-8')
    print(json.dumps(changed, indent=2))
