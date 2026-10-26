"""Führt Notebooks headless aus, mit Fortschritt pro Zelle, und speichert die Outputs.

Anders als `jupyter nbconvert --execute` zeigt dieser Runner, welche Zelle gerade läuft
und wie lange sie gebraucht hat. Bei einem Fehler wird das Notebook inklusive
Fehlermeldung gespeichert und der Lauf abgebrochen.

    python scripts/run_notebooks.py notebooks/03_baseline_rf.ipynb notebooks/04_xgboost.ipynb
    python scripts/run_notebooks.py --threads notebooks/03_baseline_rf.ipynb   # siehe --help
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

LIBOMP_CANDIDATES = ["/opt/homebrew/opt/libomp/lib/libomp.dylib", "/usr/local/opt/libomp/lib/libomp.dylib"]


def ensure_libomp() -> None:
    """XGBoost braucht auf macOS libomp. Fehlt Homebrews libomp, wird die von
    scikit-learn mitgelieferte Kopie als Fallback eingetragen (gilt für die Kernel)."""
    if sys.platform != "darwin" or any(Path(p).exists() for p in LIBOMP_CANDIDATES):
        return
    import sklearn

    dylibs = Path(sklearn.__file__).parent / ".dylibs"
    if (dylibs / "libomp.dylib").exists():
        current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib")
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{dylibs}:{current}"
        print(f"Hinweis: libomp nicht über Homebrew installiert – nutze {dylibs / 'libomp.dylib'}\n"
              f"         (dauerhafte Lösung: brew install libomp)")


def fmt(sec: float) -> str:
    return f"{sec:5.1f}s" if sec < 120 else f"{sec / 60:4.1f}min"


def run(path: Path, threads: bool) -> bool:
    nb = nbformat.read(path, as_version=4)
    code_idx = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    t_nb, t_cell = time.time(), {}

    def on_start(cell, cell_index, **_):
        t_cell[cell_index] = time.time()
        n = code_idx.index(cell_index) + 1 if cell_index in code_idx else "?"
        first = next((ln for ln in cell.source.splitlines() if ln.strip() and not ln.startswith("#")), "")
        print(f"  [{time.strftime('%H:%M:%S')}] Zelle {n}/{len(code_idx)}  {first[:60]}", flush=True)

    def on_complete(cell, cell_index, **_):
        print(f"             ✓ {fmt(time.time() - t_cell.get(cell_index, time.time()))}", flush=True)

    extra = []
    if threads:  # für Umgebungen, in denen joblib keine Prozesse starten darf (z. B. Sandboxes)
        extra = ["--IPKernelApp.exec_lines=from joblib.parallel import register_parallel_backend, "
                 "ThreadingBackend; register_parallel_backend('threading', ThreadingBackend, make_default=True)"]

    client = NotebookClient(
        nb, timeout=None, kernel_name="python3", extra_arguments=extra,
        resources={"metadata": {"path": str(path.parent)}},
        on_cell_start=on_start, on_cell_complete=on_complete,
    )
    print(f"▶ {path.name}", flush=True)
    ok = True
    try:
        client.execute()
    except CellExecutionError as err:
        ok = False
        print("\n✗ Fehler – Notebook wird mit Fehlermeldung gespeichert:\n" + str(err)[-2000:], flush=True)
    finally:
        nbformat.write(nb, path)
    print(f"{'✓' if ok else '✗'} {path.name} in {fmt(time.time() - t_nb)}\n", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebooks", nargs="+", type=Path)
    ap.add_argument("--threads", action="store_true",
                    help="joblib-Threads statt Prozesse (nur nötig, wenn Prozess-Pools blockiert sind)")
    args = ap.parse_args()
    ensure_libomp()
    for nb in args.notebooks:
        if not run(nb.resolve(), args.threads):
            sys.exit(1)


if __name__ == "__main__":
    main()
