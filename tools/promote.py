# tools/promote.py
import argparse
from src.registry import set_alias

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--alias", required=True, choices=["production", "staging", "best", "latest"])
    ap.add_argument("--run-id", required=False, default=None)
    args = ap.parse_args()

    set_alias(args.model, args.alias, args.run_id)
    print(f"[OK] set {args.model}:{args.alias} = {args.run_id}")

if __name__ == "__main__":
    main()
