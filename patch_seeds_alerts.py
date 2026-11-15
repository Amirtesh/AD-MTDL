#!/usr/bin/env python3
"""patch_seeds_alerts.py - make select_seeds_v2.py skip seeds that would score 0 from step 1 (they match the scoring function's own
CUSTOM_ALERTS, read from optimize.py) and, by default, PAINS. Backup: select_seeds_v2.py.bak. Run once from ~/RL."""
import shutil, sys
from pathlib import Path

def sub(src, old, new):
    assert src.count(old) == 1, f"anchor not found exactly once ({src.count(old)}): {old[:70]!r}"
    return src.replace(old, new)

p = Path("select_seeds_v2.py"); s = p.read_text()
if "CUSTOM_ALERTS" in s:
    sys.exit("already patched")
shutil.copy(p, "select_seeds_v2.py.bak")
s = sub(s, 'ap.add_argument("--n", type=int, default=3); ap.add_argument("--out", default="campaign_seeds_v2.csv")\n',
        'ap.add_argument("--n", type=int, default=3); ap.add_argument("--out", default="campaign_seeds_v2.csv")\n'
        'ap.add_argument("--allow-pains", action="store_true", help="do not exclude PAINS matches")\n')
s = sub(s, '''def parent_smiles(s):
    mol = Chem.MolFromSmiles(s)
    if mol is None: return None
    p = parent_mol(mol)
    return Chem.MolToSmiles(p) if 200 <= Descriptors.MolWt(p) <= 550 else None
''', '''import ast
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
_tree = ast.parse(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimize.py")).read())
ALERTS = [Chem.MolFromSmarts(a) for a in next(ast.literal_eval(n.value) for n in _tree.body
          if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CUSTOM_ALERTS")]
_pp = FilterCatalogParams(); _pp.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS); PAINS = FilterCatalog(_pp)
n_alert = n_pains = 0
def parent_smiles(s):
    global n_alert, n_pains
    mol = Chem.MolFromSmiles(s)
    if mol is None: return None
    p = parent_mol(mol)
    if not 200 <= Descriptors.MolWt(p) <= 550: return None
    if any(p.HasSubstructMatch(a) for a in ALERTS): n_alert += 1; return None     # would score 0 from step 1
    if not args.allow_pains and PAINS.HasMatch(p): n_pains += 1; return None
    return Chem.MolToSmiles(p)
''')
s = sub(s, "universe = sorted({v for v in cache.values() if v})\n",
        'print(f"excluded as seeds: {n_alert} match the scoring function\'s structural alerts, {n_pains} match PAINS")\n'
        "universe = sorted({v for v in cache.values() if v})\n")
p.write_text(s); print("patched select_seeds_v2.py (backup select_seeds_v2.py.bak)")
