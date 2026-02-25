import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# config

CSV_PATH = "dataset2.csv"
RANDOM_STATE = 42
N_SPLITS = 10
MODEL_OUT = "modeldata.json"

HARTREE_TO_EV = 27.211386

df = pd.read_csv(CSV_PATH)

df["pbe_bandwidth_ev"] = (
    (df["pbe_lumo_plus1"] - df["pbe_homo_minus1"]) * HARTREE_TO_EV
)

df["homo_lumo_mid"] = 0.5 * (
    df["pbe_homo_hartree"] + df["pbe_lumo_hartree"]
)

df["level_spacing_occ"] = (
    df["pbe_homo_hartree"] - df["pbe_homo_minus1"]
)

df["level_spacing_virt"] = (
    df["pbe_lumo_plus1"] - df["pbe_lumo_hartree"]
)

df["dipole_per_atom"] = df["pbe_dipole_debye"] / df["num_atoms"]

df["dipole_anisotropy"] = (
    df[["pbe_mu_x", "pbe_mu_y", "pbe_mu_z"]]
    .abs()
    .max(axis=1)
    / (df["pbe_dipole_debye"] + 1e-6)
)

df["gap_per_atom"] = df["pbe_gap_ev"] / df["num_atoms"]
df["energy_per_atom"] = df["pbe_energy_hartree"] / df["num_atoms"]

feature_cols = [
    "num_atoms",
    "energy_per_atom",

    "pbe_homo_hartree",
    "pbe_lumo_hartree",
    "pbe_gap_ev",

    "pbe_homo_minus1",
    "pbe_lumo_plus1",
    "pbe_bandwidth_ev",
    "level_spacing_occ",
    "level_spacing_virt",

    "pbe_dipole_debye",
    "dipole_per_atom",
    "dipole_anisotropy",

    "homo_lumo_mid",
    "gap_per_atom",
]

missing = set(feature_cols) - set(df.columns)
assert not missing, f"Missing columns: {missing}"

X = df[feature_cols]

# log delta gap

y_delta = df["delta_gap_ev"]
y_log = np.log(y_delta + 1e-6)

# model

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_depth=3,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
)

# 10-fold cv

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

mae_list = []
rmse_list = []
r2_list = []

y_true_all = []
y_pred_all = []
molecules_all = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_log.iloc[train_idx], y_log.iloc[test_idx]
 
    weights = 1 / (df.iloc[train_idx]["hse_gap_ev"] + 0.1)

    model.fit(X_train, y_train, sample_weight=weights)

    delta_pred = np.exp(model.predict(X_test))

    hse_pred = df.iloc[test_idx]["pbe_gap_ev"].values + delta_pred
    hse_true = df.iloc[test_idx]["hse_gap_ev"].values

    y_true_all.extend(hse_true)
    y_pred_all.extend(hse_pred)
    molecules_all.extend(df.iloc[test_idx]["molecule"])

    mae_list.append(mean_absolute_error(hse_true, hse_pred))
    rmse_list.append(np.sqrt(mean_squared_error(hse_true, hse_pred)))
    r2_list.append(r2_score(hse_true, hse_pred))

print("\n10-Fold CV performance (average):")
print(f"  MAE  = {np.mean(mae_list):.3f} ± {np.std(mae_list):.3f} eV")
print(f"  RMSE = {np.mean(rmse_list):.3f} ± {np.std(rmse_list):.3f} eV")
print(f"  R²   = {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}")

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

percent_error = np.abs(y_pred_all - y_true_all) / y_true_all * 100
within_10 = np.mean(percent_error < 10) * 100

print(f"\nMolecules within 10% error: {within_10:.1f}%")

model.get_booster().save_model(MODEL_OUT)

from xgboost import XGBRegressor
import joblib

# final model

final_model = XGBRegressor(
    n_estimators=500,          
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y_delta)

feature_names = list(X.columns)
joblib.dump(final_model, "xgb_delta_hse_final.pkl")
joblib.dump(feature_names, "xgb_feature_names.pkl")

pred_train = final_model.predict(X)

'''
#=====================
# PERCENT ERROR PER MOLECULE WITH LABELS
# =====================
import matplotlib.pyplot as plt
import numpy as np

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)
percent_error = np.abs(y_pred_all - y_true_all) / y_true_all * 100

plt.figure(figsize=(10,5))
plt.scatter(range(len(percent_error)), percent_error, color='blue', edgecolor='k', s=60)
plt.axhline(10, color='red', linestyle='--', lw=2, label='10% error threshold')

# Label molecules exceeding 10% error
for i, mol in enumerate(molecules_all):
    if percent_error[i] > 10:
        #print(mol)
        plt.text(i, percent_error[i]+0.5, mol, rotation=45, fontsize=8)

plt.xlabel("Molecule index (from CV test sets)")
plt.ylabel("Absolute % Error")
plt.title("Absolute % Error per Molecule (10-Fold CV)")
plt.legend()
plt.tight_layout()
plt.show()

# =====================
# PARITY PLOT (HSE GAP)
# =====================

import matplotlib.pyplot as plt

# Reconstruct HSE gaps
hse_true = df.loc[y_true_all.index if hasattr(y_true_all, "index") else df.index[:len(y_true_all)], "pbe_gap_ev"].values + y_true_all
hse_pred = df.loc[:len(y_pred_all)-1, "pbe_gap_ev"].values + y_pred_all

plt.figure(figsize=(6,6))
plt.scatter(hse_true, hse_pred, alpha=0.7, edgecolor="k", s=60)

min_val = min(hse_true.min(), hse_pred.min())
max_val = max(hse_true.max(), hse_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

plt.xlabel("True HSE Gap (eV)")
plt.ylabel("Predicted HSE Gap (eV)")
plt.title("Parity Plot: HSE Band Gap (10-Fold CV)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =====================
# FEATURE IMPORTANCE
# =====================

booster = model.get_booster()
importance = booster.get_score(importance_type="gain")

# Convert to sorted DataFrame
imp_df = (
    pd.DataFrame(
        importance.items(),
        columns=["feature", "gain"]
    )
    .sort_values("gain", ascending=True)
)

plt.figure(figsize=(8,6))
plt.barh(imp_df["feature"], imp_df["gain"])
plt.xlabel("Gain")
plt.title("XGBoost Feature Importance (Gain)")
plt.tight_layout()
plt.show()
'''