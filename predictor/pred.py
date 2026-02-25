import os
import re
import numpy as np
import pandas as pd
import joblib

HARTREE_TO_EV = 27.211386245988

# ===============================
# Load trained ML model
# ===============================
model = joblib.load("xgb_delta_hse_final.pkl")
feature_names = joblib.load("xgb_feature_names.pkl")

# ===============================
# Gaussian parsing helpers
# ===============================

def parse_orbitals(lines):
    occ = []
    virt = []

    for line in lines:
        if "occ. eigenvalues" in line:
            nums = re.findall(r"-?\d+\.\d+", line)
            occ.extend(map(float, nums))

        elif "virt. eigenvalues" in line:
            nums = re.findall(r"-?\d+\.\d+", line)
            virt.extend(map(float, nums))

    if len(occ) < 2 or len(virt) < 2:
        raise ValueError("Insufficient orbital energies found")

    occ = np.array(occ)
    virt = np.array(virt)

    homo = occ.max()
    homo_minus1 = np.sort(occ)[-2]
    lumo = virt.min()
    lumo_plus1 = np.sort(virt)[1]

    return homo, homo_minus1, lumo, lumo_plus1



def parse_energy(lines):
    for line in lines:
        if "SCF Done" in line:
            return float(line.split()[4])
    raise ValueError("SCF energy not found")


def parse_dipole(lines, fname=None):
    """
    Parses the dipole moment from a Gaussian .out file.
    Handles both single-line and multi-line Gaussian outputs.
    
    Returns:
        dipole (float): total dipole magnitude in Debye
        mu (list of floats): [mu_x, mu_y, mu_z]
    """
    mu = None

    for i, line in enumerate(lines):
        # Multi-line format (most common)
        if "Dipole moment (field-independent basis, Debye)" in line:
            # Next line should contain X= Y= Z= Tot=
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", next_line)
                if len(nums) >= 4:
                    mu = [float(nums[0]), float(nums[1]), float(nums[2])]
                    break

        # Alternative single-line formats
        elif "Dipole moment" in line and "Debye" in line:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(nums) >= 3:
                mu = [float(nums[0]), float(nums[1]), float(nums[2])]
                break
        elif "Tot=" in line and "Debye" in line:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(nums) >= 4:
                mu = [float(nums[0]), float(nums[1]), float(nums[2])]
                break

    # If dipole not found, assume zero (symmetric molecules)
    if mu is None:
        if fname:
            print(f"WARNING: Dipole not found in {fname}, setting to zero")
        mu = [0.0, 0.0, 0.0]

    dipole = np.linalg.norm(mu)
    return dipole, mu


def parse_num_atoms(lines):
    for line in lines:
        if "NAtoms=" in line:
            return int(line.split()[1])
    raise ValueError("Number of atoms not found")


def parse_gap_from_orbitals(lines):
    homo, _, lumo, _ = parse_orbitals(lines)
    return (lumo - homo) * HARTREE_TO_EV


# ===============================
# Feature construction
# ===============================

def extract_features_from_out(filepath):
    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    num_atoms = parse_num_atoms(lines)
    energy = parse_energy(lines)
    dipole, mu = parse_dipole(lines, os.path.basename(filepath))
    homo, homo_m1, lumo, lumo_p1 = parse_orbitals(lines)

    pbe_gap_ev = (lumo - homo) * HARTREE_TO_EV

    features = {
        "num_atoms": num_atoms,
        "energy_per_atom": energy / num_atoms,

        "pbe_homo_hartree": homo,
        "pbe_lumo_hartree": lumo,
        "pbe_gap_ev": pbe_gap_ev,

        "pbe_homo_minus1": homo_m1,
        "pbe_lumo_plus1": lumo_p1,

        "pbe_bandwidth_ev": (lumo_p1 - homo_m1) * HARTREE_TO_EV,
        "level_spacing_occ": (homo - homo_m1) * HARTREE_TO_EV,
        "level_spacing_virt": (lumo_p1 - lumo) * HARTREE_TO_EV,

        "pbe_dipole_debye": dipole,
        "dipole_per_atom": dipole / num_atoms,
        "dipole_anisotropy": np.std(mu),

        "homo_lumo_mid": 0.5 * (homo + lumo),
        "gap_per_atom": pbe_gap_ev / num_atoms,
    }

    return features


# ===============================
# Main prediction loop
# ===============================

def predict_folder(pbe_out_dir, hse_out_dir):
    results = []

    for fname in os.listdir(pbe_out_dir):
        if not fname.endswith(".out"):
            continue

        pbe_path = os.path.join(pbe_out_dir, fname)
        hse_path = os.path.join(hse_out_dir, fname)

        try:
            feats = extract_features_from_out(pbe_path)
            X = pd.DataFrame([feats])[feature_names]

            delta_pred = model.predict(X)[0]
            hse_gap_pred = feats["pbe_gap_ev"] + delta_pred

            actual_hse_gap = None
            error_percent = None
            if os.path.exists(hse_path):
                with open(hse_path, "r", errors="ignore") as f:
                    hse_lines = f.readlines()
                actual_hse_gap = parse_gap_from_orbitals(hse_lines)

                # compute error %
                if actual_hse_gap > 1e-6:  # avoid division by zero
                    error_percent = 100.0 * abs(hse_gap_pred - actual_hse_gap) / actual_hse_gap

            results.append({
                "molecule": fname.replace(".out", ""),
                "pbe_gap_ev": round(feats["pbe_gap_ev"], 3),
                "delta_gap_pred": round(delta_pred, 3),
                "hse_gap_pred": round(hse_gap_pred, 3),
                "actual_hse_gap_ev": round(actual_hse_gap, 3) if actual_hse_gap else None,
                "error_percent": round(error_percent, 2) if error_percent else None
            })

        except Exception as e:
            print(f"FAILED: {fname} → {e}")

    return pd.DataFrame(results)



# ===============================
# Run
# ===============================

if __name__ == "__main__":
    pbe_dir = "pbe_log"
    hse_dir = "hse_log"

    df = predict_folder(pbe_dir, hse_dir)
    df.to_csv("final_hse_pred.csv", index=False)

    print("Saved predictions to final_hse_pred.csv")
