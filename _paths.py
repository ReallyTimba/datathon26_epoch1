"""Project layout. Imported first by every entry point (run_rl.py, pretrain_rl.py).

  * makes the modules in rl/, models/ and recommenders/ importable by their plain names;
  * moves the working directory to starter/, so .env is always found, whatever folder you run from;
  * tells hybrid_plus where the model files are (models/) and the agent where its state goes (state/).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ("rl", "models", "recommenders"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(ROOT)
os.environ.setdefault("HYBRID_DIR", str(ROOT / "models"))
os.environ.setdefault("RL_STATE", str(ROOT / "state" / "rl_state.npz"))

_missing = [f for f in ("meta.json", "sasrec_ctx.pt", "xgb_ranker.json") if not (ROOT / "models" / f).exists()]
if _missing:
    sys.exit(f"Model files missing in {ROOT / 'models'}: {', '.join(_missing)}. Copy them there (they are the teammate's trained weights).")
if not (ROOT / ".env").exists() and "TEAM_ID" not in os.environ:
    sys.exit(f"No .env in {ROOT}: create it with TEAM_ID=... and RAWTREE_API_KEY=...")
