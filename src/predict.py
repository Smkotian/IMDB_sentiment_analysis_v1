"""Optional demo: python src/predict.py "I loved this movie" "Not a good film at all" """
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *

if __name__ == "__main__":
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    texts = sys.argv[1:] or ["An absolutely wonderful film, I loved it.", "Not good. A boring waste of time."]
    for t, p in zip(texts, model.predict(texts)):
        print("positive" if p == 1 else "negative", "|", t)
