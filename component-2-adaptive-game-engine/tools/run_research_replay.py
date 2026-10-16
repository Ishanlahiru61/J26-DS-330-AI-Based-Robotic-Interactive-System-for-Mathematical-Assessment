import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from component2_app.research_replay import replay

parser = argparse.ArgumentParser(description='Frozen observed-task inference and support decision replay; no training')
parser.add_argument('--workspace', required=True, type=Path)
args = parser.parse_args()
result = replay(args.workspace)
output = ROOT / 'docs/integration_evidence/research_replay.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
