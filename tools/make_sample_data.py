from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main(n=200, seed=42):
    src = ROOT / "data" / "raw" / "train.csv"
    dst = ROOT / "tests" / "data" / "sample_train.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    sample.to_csv(dst, index=False)
    print(f"Saved: {dst}  shape={sample.shape}")

if __name__ == "__main__":
    main()
