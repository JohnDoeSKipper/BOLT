"""
Generate extended (synthetic + real) datasets for training.

Run from project root:
    python scripts/generate_synthetic_data.py

Outputs CSVs in data/synthetic/ — these have ~365 days of data each,
combining synthetic days (going back in time) with the real meter data.
"""
import sys
from pathlib import Path

# Make `src` importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import auto_load
from src.synth_data import synthesize_extended, quality_report


REAL_FILES = [
    ("data/1__Load_Profile_With_Solar_Installed_SoL_copy.xlsx",
     "SoL_With_Solar_944kWp", 944.88),
    ("data/2__Load_Profile_No_Solar_E_copy.xlsx",
     "E_No_Solar", 0.0),
    ("data/3__Load_Profile_No_Solar_SuN_copy.xlsx",
     "SuN_No_Solar", 0.0),
    ("data/4__Load_Profile_With_Solar_Mi2_copy.xlsx",
     "Mi2_With_Solar", 500.0),
]

OUTPUT_DIR = Path("data/synthetic")
TARGET_DAYS = 365
NOISE_STD = 0.05
ANOMALY_RATE = 0.02
SEED = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}")
    print(f"  SYNTHETIC DATA GENERATION  →  {TARGET_DAYS} days target")
    print(f"{'='*72}\n")

    for file_path, label, capacity in REAL_FILES:
        print(f"--- {label} ---")
        path = Path(file_path)
        if not path.exists():
            print(f"  SKIP: {file_path} not found\n")
            continue

        df = auto_load(path)
        real_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
        print(f"  Real:   {len(df):>5} samples  ({real_days} days)")

        extended = synthesize_extended(
            df,
            target_days=TARGET_DAYS,
            noise_std=NOISE_STD,
            anomaly_rate=ANOMALY_RATE,
            seed=SEED,
        )
        ext_days = (extended["timestamp"].max() - extended["timestamp"].min()).days + 1

        out_path = OUTPUT_DIR / f"extended_{label}.csv"
        extended.to_csv(out_path, index=False)

        synth_n = len(extended) - len(df)
        qr = quality_report(df, extended)
        print(f"  Synth:  {synth_n:>5} samples added")
        print(f"  Total:  {len(extended):>5} samples  ({ext_days} days)")
        print(f"  Range:  {extended['timestamp'].min()} → {extended['timestamp'].max()}")
        print(f"  Quality: mean ratio={qr['mean_ratio']:.3f}, std ratio={qr['std_ratio']:.3f}")
        print(f"  Saved:  {out_path}\n")

    print(f"{'='*72}")
    print("  Done. Use these CSVs in the Streamlit app's 'Historical Data' uploader.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
