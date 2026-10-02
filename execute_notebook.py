"""
Script to execute notebooks/mnist_neural_network.ipynb cell-by-cell in the current Python environment
and embed real inputs, outputs, charts (PNG base64), and execution counts into the notebook.
"""
import os
import sys
import io
import json
import base64
import contextlib
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["KERAS_HOME"] = os.path.join(BASE_DIR, ".keras_cache")
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".mpl_cache")
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nb_path = os.path.join(BASE_DIR, "notebooks/mnist_neural_network.ipynb")
with open(nb_path, "r") as f:
    nb = json.load(f)

# Global execution scope
exec_globals = {
    "__name__": "__main__",
    "RUN_TRAINING": False
}

execution_count = 1

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    
    code = "".join(cell["source"])
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    cell["outputs"] = []
    cell["execution_count"] = execution_count
    
    # Intercept plt.show to capture generated figures
    captured_figs = []
    orig_show = plt.show
    def custom_show(*args, **kwargs):
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            captured_figs.append(img_b64)
            plt.close(fig)
    plt.show = custom_show
    
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, exec_globals)
            # Catch any remaining figures
            if plt.get_fignums():
                custom_show()
    except Exception as e:
        tb = traceback.format_exc()
        cell["outputs"].append({
            "output_type": "error",
            "ename": type(e).__name__,
            "evalue": str(e),
            "traceback": tb.split("\n")
        })
        print(f"Error executing cell {execution_count}:\n{tb}")
    finally:
        plt.show = orig_show
        
    out_text = stdout_capture.getvalue()
    err_text = stderr_capture.getvalue()
    
    if out_text:
        cell["outputs"].append({
            "output_type": "stream",
            "name": "stdout",
            "text": [line + "\n" for line in out_text.rstrip("\n").split("\n")]
        })
    if err_text:
        # filter harmless warnings
        filtered_err = [line for line in err_text.split("\n") if "UserWarning" not in line and "IPython parent" not in line]
        if "".join(filtered_err).strip():
            cell["outputs"].append({
                "output_type": "stream",
                "name": "stderr",
                "text": [line + "\n" for line in filtered_err]
            })
            
    for img_b64 in captured_figs:
        cell["outputs"].append({
            "output_type": "display_data",
            "data": {
                "image/png": img_b64,
                "text/plain": ["<Figure size ...>"]
            },
            "metadata": {}
        })
        
    print(f"Executed cell {execution_count}: {len(cell['outputs'])} output items generated.")
    execution_count += 1

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=2)

print("\n✓ Notebook execution and output embedding completed successfully.")
