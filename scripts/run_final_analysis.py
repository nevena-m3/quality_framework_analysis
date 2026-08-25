from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_analysis.final_pipeline import run_final_analysis


if __name__ == "__main__":
    result = run_final_analysis(ROOT)
    print(json.dumps(result, indent=2))
