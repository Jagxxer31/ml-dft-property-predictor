import pandas as pd
import numpy as np

CSV_FILE = "pbe_hse_dataset.csv"

# Tolerances
GAP_TOL = 0.05      # eV (numerical tolerance)
MAX_GAP = 20.0      # eV (hard sanity cap)

# Required columns
REQUIRED_COLUMNS = [
    "molecule",
    "num_atoms",
    "pbe_energy_hartree",
    "pbe_homo_hartree",
    "pbe_lumo_hartree",
    "pbe_gap_ev",
    "pbe_dipole_debye",
    "hse_gap_ev",
    "delta_gap_ev",
]

def fail(msg):
    print(f"[FAIL] {msg}")
    raise SystemExit(1)

def warn(msg):
    print(f"[WARN] {msg}")

def ok(msg):
    print(f"[OK] {msg}")

# ---------------- Load CSV ----------------
try:
    df = pd.read_csv(CSV_FILE)
except Exception as e:
    fail(f"Could not read CSV: {e}")

ok(f"Loaded CSV with {len(df)} molecules")

# ---------------- Column checks ----------------
missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
if missing_cols:
    fail(f"Missing columns: {missing_cols}")
ok("All required columns present")

# ---------------- Basic integrity ----------------
if df.isna().any().any():
    fail("Dataset contains NaN values")
ok("No NaN values")

if df["molecule"].duplicated().any():
    dups = df[df["molecule"].duplicated()]["molecule"].tolist()
    fail(f"Duplicate molecule names found: {dups}")
ok("No duplicate molecules")

# ---------------- Physical sanity ----------------
if not (df["num_atoms"] > 0).all():
    fail("Invalid atom count detected")
ok("Atom counts valid")

if not (df["pbe_energy_hartree"] < 0).all():
    warn("Some total energies are non-negative (check unusual cases)")

if not (df["pbe_gap_ev"] > 0).all():
    fail("Non-positive PBE gap detected")
ok("PBE gaps positive")

if not (df["hse_gap_ev"] > 0).all():
    fail("Non-positive HSE gap detected")
ok("HSE gaps positive")

# HSE >= PBE (allow tolerance)
if not (df["hse_gap_ev"] >= df["pbe_gap_ev"] - GAP_TOL).all():
    bad = df[df["hse_gap_ev"] < df["pbe_gap_ev"] - GAP_TOL]
    fail(f"HSE gap significantly smaller than PBE for: {bad['molecule'].tolist()}")
ok("HSE ≥ PBE gaps (within tolerance)")

# Delta-gap consistency
delta_calc = df["hse_gap_ev"] - df["pbe_gap_ev"]
if not np.allclose(delta_calc, df["delta_gap_ev"], atol=1e-6):
    fail("delta_gap_ev inconsistent with HSE − PBE")
ok("delta_gap_ev consistent")

# ---------------- Outlier detection ----------------
if (df["pbe_gap_ev"] > MAX_GAP).any():
    warn("Very large PBE gap detected (>20 eV)")

if (df["hse_gap_ev"] > MAX_GAP).any():
    warn("Very large HSE gap detected (>20 eV)")

# ---------------- ML sanity ----------------
numeric_cols = df.select_dtypes(include=[np.number])

constant_cols = [
    col for col in numeric_cols.columns
    if numeric_cols[col].nunique() == 1
]

if constant_cols:
    warn(f"Constant columns detected (may be useless for ML): {constant_cols}")
else:
    ok("No constant numeric columns")

# ---------------- Summary ----------------
print("\n=== DATASET SUMMARY ===")
print(df.describe())

print("\n[SUCCESS] Dataset passed all critical checks")
