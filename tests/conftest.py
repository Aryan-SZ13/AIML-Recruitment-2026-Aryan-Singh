import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["KERAS_HOME"] = os.path.join(BASE_DIR, ".keras_cache")
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".mpl_cache")
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
