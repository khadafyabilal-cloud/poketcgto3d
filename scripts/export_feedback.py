"""Export user ratings as a training set.

    PYTHONPATH=. python scripts/export_feedback.py [out_dir]      (default: feedback_export)

Writes <out_dir>/labels.jsonl (one JSON object per rated result) and <out_dir>/images/*.jpg (the straightened card
pictures). `label_dex` is the best guess at the truth: a user's correction, else the app's answer when net up-voted,
else null (unknown - skip those rows when training). Use it to fine-tune a classifier OFFLINE; the running app does
not retrain itself (see feedback.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feedback import FeedbackStore

root = Path(__file__).resolve().parent.parent
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "feedback_export"
n = FeedbackStore(root / "feedback").export(out)
print(f"wrote {n} rated results to {out}")
