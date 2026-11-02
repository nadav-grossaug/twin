"""Execute a notebook in place: python run_nb.py in.ipynb out.ipynb (cwd = geolife/inside/notebooks)."""
import sys

import nbformat
from nbclient import NotebookClient

nb = nbformat.read(sys.argv[1], 4)
NotebookClient(nb, timeout=3000, kernel_name="python3",
               resources={"metadata": {"path": "geolife/inside/notebooks"}}).execute()
nbformat.write(nb, sys.argv[2])
