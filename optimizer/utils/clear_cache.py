import os
import shutil

PLOT_DIR = os.path.join(
    "static",
    "optimizer",
)

def clear_plot_cache():
    if os.path.exists(PLOT_DIR):
        shutil.rmtree(PLOT_DIR)
    os.makedirs(PLOT_DIR, exist_ok=True)