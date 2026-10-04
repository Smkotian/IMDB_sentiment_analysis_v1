"""Stage 1 - read the raw IMDb files, apply preprocessing, cache to data/cache/.
Preprocessing is stateless (no statistics are learned), so doing it for train and test here is leak-free."""
import sys
import joblib
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *

def main():
    for split in ("train", "test"):
        texts, y = read_raw_split(split)
        print(f"  {split}: {len(texts)} reviews  (pos={int(y.sum())}, neg={int((y == 0).sum())})")
        if split == "train" and len(texts) != 25000:
            print("  WARNING: expected 25000 training reviews - is this the full Stanford aclImdb dataset?")
        for name, fn in PREPROCESSORS.items():
            joblib.dump(([fn(t) for t in texts], y), text_cache_path(split, name))
    ex = "This movie was not very good, it isn't funny.<br />Really."
    print("  example basic   :", basic_preprocess(ex))
    print("  example negation:", negation_preprocess(ex))
    print("  cached to", CACHE_DIR)

if __name__ == "__main__":
    main()
