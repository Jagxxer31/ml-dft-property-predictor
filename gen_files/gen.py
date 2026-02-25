import os
import requests
from rdkit import Chem
from rdkit.Chem import AllChem

# ------------------------------
# Step 0: Settings
# ------------------------------
extra_molecules_30 = [
    "ammonia",
    "hydrogen cyanide",
    "hydrogen sulfide",
    "carbon monoxide",
    "carbon dioxide",
    "nitric oxide",
    "isocyanic acid",
    "cyanamide",
    "methanol",
    "ethanethiol",
    "dimethylamine",
    "ethylamine",
    "trimethylamine",
    "chloroethane",
    "fluoroethane",
    "bromoethane",
    "vinyl fluoride",
    "vinyl bromide",
    "acrylonitrile",
    "acrolein",
    "acetaldehyde oxime",
    "formaldehyde oxime",
    "ethyl acetate",
    "methyl acetate",
    "dimethyl carbonate",
    "ethylene oxide",
    "propylene oxide",
    "cyclopropane",
    "aziridine",
    "thiirane"
]



all_molecules = extra_molecules_30
all_molecules=[
    "maleic acid"
]

SDF_DIR = "sdf"
GJF_DIR = "d_pbe_gjf"
os.makedirs(SDF_DIR, exist_ok=True)
os.makedirs(GJF_DIR, exist_ok=True)

# ------------------------------
# Step 1: Fetch SDF from PubChem
# ------------------------------
def fetch_sdf(mol_name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{mol_name}/SDF"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"[FAIL] Could not fetch {mol_name}")
        return None
    path = os.path.join(SDF_DIR, f"{mol_name}.sdf")
    with open(path, "wb") as f:
        f.write(r.content)
    return path

sdf_paths = []
for mol in all_molecules:
    p = fetch_sdf(mol)
    if p:
        sdf_paths.append(p)
        print("Added sdf for "+os.path.splitext(os.path.basename(mol))[0])

# ------------------------------
# Step 2: Convert SDF → Gaussian (.gjf)
# ------------------------------
ROUTE_SECTION = "#PBEPBE/6-31G(d) SCF=Tight OPT=Loose Pop=Minimal"
CHARGE_MULT = "0 1"

for sdf_file in sdf_paths:
    mol = Chem.MolFromMolFile(sdf_file, removeHs=False)
    if mol is None:
        print(f"[FAIL] RDKit could not read {sdf_file}")
        continue

    # 🔴 CRITICAL FIX 1: add hydrogens explicitly
    mol = Chem.AddHs(mol)

    # 🔴 CRITICAL FIX 2: always generate reliable 3D geometry
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    # Sanity check
    if mol.GetNumAtoms() < 3:
        print(f"[SKIP] Suspicious molecule: {sdf_file}")
        continue

    base = os.path.splitext(os.path.basename(sdf_file))[0]
    gjf_path = os.path.join(GJF_DIR, f"{base}.gjf")

    # 🔴 CRITICAL FIX 3: Windows Gaussian-safe encoding
    with open(gjf_path, "w", encoding="mbcs") as f:
        f.write("%nprocshared=2\n")
        f.write(ROUTE_SECTION + "\n\n")
        f.write(base + "\n\n")
        f.write(CHARGE_MULT + "\n")

        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(
                f"{atom.GetSymbol():<2} "
                f"{pos.x:>12.6f} {pos.y:>12.6f} {pos.z:>12.6f}\n"
            )
        f.write("\n")
    print("Finished writing "+ base)

print(f"✅ {len(os.listdir(GJF_DIR))} Gaussian input files ready.")
