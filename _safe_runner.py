"""
Safe runner: executes a target script but intercepts all file writes
(DataFrame.to_csv, plt.savefig, fig.savefig, json.dump with file handles)
and redirects them to data/new_results/NEW_<original_filename>.

Usage:  python _safe_runner.py <target_script.py>
"""
import sys, os, re, pathlib, builtins, json
from pathlib import Path

TARGET = sys.argv[1]
SAFE_DIR = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data/new_results")
SAFE_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR_STR = "/Users/adelinewen/Desktop/pre-airdrop-detection/data"

def redirect_path(p):
    """Return a NEW_-prefixed path in safe dir if p is in data/, else unchanged."""
    s = str(p)
    if DATA_DIR_STR in s:
        fname = pathlib.Path(s).name
        new_p = SAFE_DIR / f"NEW_{fname}"
        return new_p
    return p

# ---------- Patch pandas -------------------------------------------------
import pandas as pd
_orig_df_to_csv = pd.DataFrame.to_csv
def _patched_df_to_csv(self, path_or_buf=None, **kwargs):
    if path_or_buf is not None and not hasattr(path_or_buf, 'write'):
        new_path = redirect_path(path_or_buf)
        print(f"[REDIRECT] {path_or_buf} → {new_path}")
        return _orig_df_to_csv(self, new_path, **kwargs)
    return _orig_df_to_csv(self, path_or_buf, **kwargs)
pd.DataFrame.to_csv = _patched_df_to_csv

# ---------- Patch matplotlib --------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_orig_plt_savefig = plt.savefig
def _patched_plt_savefig(fname, *args, **kwargs):
    new_fname = redirect_path(fname)
    print(f"[REDIRECT] {fname} → {new_fname}")
    return _orig_plt_savefig(new_fname, *args, **kwargs)
plt.savefig = _patched_plt_savefig

# Also patch Figure.savefig (instance method used as fig.savefig)
import matplotlib.figure
_orig_fig_savefig = matplotlib.figure.Figure.savefig
def _patched_fig_savefig(self, fname, *args, **kwargs):
    new_fname = redirect_path(fname)
    print(f"[REDIRECT fig] {fname} → {new_fname}")
    return _orig_fig_savefig(self, new_fname, *args, **kwargs)
matplotlib.figure.Figure.savefig = _patched_fig_savefig

# ---------- Patch builtins.open for json writes -------------------------
_orig_open = builtins.open
def _patched_open(file, mode='r', *args, **kwargs):
    if ('w' in str(mode)) and isinstance(file, (str, pathlib.Path)):
        new_file = redirect_path(file)
        if new_file != file:
            print(f"[REDIRECT open] {file} → {new_file}")
            file = new_file
    return _orig_open(file, mode, *args, **kwargs)
builtins.open = _patched_open

# ---------- Execute target script ---------------------------------------
print(f"\n{'='*60}")
print(f"Running: {TARGET}")
print(f"Outputs → {SAFE_DIR}/NEW_*")
print(f"{'='*60}\n")

script_globals = {
    '__name__': '__main__',
    '__file__': TARGET,
    'pd': pd,
    'plt': plt,
}

with _orig_open(TARGET, 'r') as f:
    code = f.read()

exec(compile(code, TARGET, 'exec'), script_globals)
print(f"\n[DONE] {TARGET}")
