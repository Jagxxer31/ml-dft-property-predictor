import os
import re
import pandas as pd

HARTREE_TO_EV = 27.211386

# ---------- Common eigenvalue parser ----------
def extract_orbitals(lines):
    occ = []
    virt = []

    float_pattern = re.compile(r"-?\d+\.\d+")

    for line in lines:
        if "Alpha  occ. eigenvalues" in line:
            occ.extend([float(x) for x in float_pattern.findall(line)])
        elif "Alpha virt. eigenvalues" in line:
            virt.extend([float(x) for x in float_pattern.findall(line)])

    if len(occ) >= 2 and len(virt) >= 2:
        return {
            "homo": occ[-1],
            "homo_minus1": occ[-2],
            "lumo": virt[0],
            "lumo_plus1": virt[1],
            "gap_ev": (virt[0] - occ[-1]) * HARTREE_TO_EV
        }

    return None

from rdkit import Chem

def add_structural_descriptors(df, sdf_dir):
    num_rings = []
    num_hetero = []

    for mol_name in df["molecule"]:
        mol = Chem.MolFromMolFile(os.path.join(sdf_dir, mol_name + ".sdf"))
        num_rings.append(mol.GetRingInfo().NumRings())
        num_hetero.append(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [1,6]))

    df["num_rings"] = num_rings
    df["num_heteroatoms"] = num_hetero

# ---------- PBE parser ----------
def parse_pbe_out(filepath):
    data = {
        "molecule": os.path.basename(filepath).replace(".out", ""),
        "num_atoms": None,
        "pbe_energy_hartree": None,
        "pbe_homo_hartree": None,
        "pbe_homo_minus1": None,
        "pbe_lumo_hartree": None,
        "pbe_lumo_plus1": None,
        "pbe_gap_ev": None,
        "pbe_dipole_debye": None,
        "pbe_mu_x": None,
        "pbe_mu_y": None,
        "pbe_mu_z": None,
    }


    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "SCF Done:" in line:
            m = re.search(r"SCF Done:\s+E\([^)]+\)\s+=\s+(-?\d+\.\d+)", line)
            if m:
                data["pbe_energy_hartree"] = float(m.group(1))

        if "NAtoms=" in line:
            data["num_atoms"] = int(line.split()[1])

        if "Dipole moment (field-independent basis, Debye)" in line:
            for j in range(i, i + 5):
                if "X=" in lines[j]:
                    parts = lines[j].replace("=", " ").split()
                    data["pbe_mu_x"] = float(parts[1])
                    data["pbe_mu_y"] = float(parts[3])
                    data["pbe_mu_z"] = float(parts[5])
                    data["pbe_dipole_debye"] = float(parts[7])
                    break


    orb = extract_orbitals(lines)
    if orb:
        data["pbe_homo_hartree"] = round(orb["homo"], 6)
        data["pbe_homo_minus1"] = round(orb["homo_minus1"], 6)
        data["pbe_lumo_hartree"] = round(orb["lumo"], 6)
        data["pbe_lumo_plus1"] = round(orb["lumo_plus1"], 6)
        data["pbe_gap_ev"] = round(orb["gap_ev"], 6)

    return data


# ---------- HSE parser (gap only) ----------
def parse_hse_out(filepath):
    molecule = os.path.basename(filepath).replace(".out", "")

    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    orb = extract_orbitals(lines)
    if orb is None:
        return None

    return {
        "molecule": molecule,
        "hse_gap_ev": round(orb["gap_ev"], 6)
    }



# ---------- Main workflow ----------
pbe_dir = "pbe_log"
hse_dir = "hse_log"

pbe_records = {}
for file in os.listdir(pbe_dir):
    if file.endswith(".out"):
        rec = parse_pbe_out(os.path.join(pbe_dir, file))
        if rec["pbe_gap_ev"] is not None:
            pbe_records[rec["molecule"]] = rec

hse_records = {}
for file in os.listdir(hse_dir):
    if file.endswith(".out"):
        rec = parse_hse_out(os.path.join(hse_dir, file))
        if rec:
            hse_records[rec["molecule"]] = rec

# ---------- Merge ----------
final_rows = []

for mol, pbe_data in pbe_records.items():
    if mol in hse_records:
        hse_gap = hse_records[mol]["hse_gap_ev"]
        delta_gap = round(hse_gap - pbe_data["pbe_gap_ev"], 6)

        row = {
            **pbe_data,
            "hse_gap_ev": hse_gap,
            "delta_gap_ev": delta_gap
        }
        final_rows.append(row)

df = pd.DataFrame(final_rows)
df.to_csv("dataset2.csv", index=False)

#print(df.head())
print(f"\nFinal dataset size: {len(df)} molecules")

# 1. No NaNs
assert not df.isna().any().any()

# 2. Positive gaps
assert (df["pbe_gap_ev"] > 0).all()
assert (df["hse_gap_ev"] > 0).all()

# 3. HSE > PBE
assert (df["hse_gap_ev"] > df["pbe_gap_ev"]).all()

# 4. Dataset size
print("Final dataset size:", len(df))

