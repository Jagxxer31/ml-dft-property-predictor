import os
import re

# =========================
# CONFIGURATION
# =========================
PBE_OUT_DIR = "d_pbe_out"
HSE_GJF_DIR = "d_hse_gjf"

METHOD_LINE = "#HSEH1PBE/6-31G(d) SP SCF=Tight"
MEM = "8GB"
NPROC = 8
CHARGE = 0
MULTIPLICITY = 1

os.makedirs(HSE_GJF_DIR, exist_ok=True)

# Atomic number → element symbol
PERIODIC_TABLE = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
    15: "P", 16: "S", 17: "Cl",
    35: "Br", 53: "I"
}

# =========================
# FUNCTIONS
# =========================
def optimization_converged(text):
    return "Optimization completed." in text


def extract_final_geometry(text):
    lines = text.splitlines()
    geometries = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if "Standard orientation:" in line or "Input orientation:" in line:
            # Move to first dashed line
            i += 1
            while i < len(lines) and "-----" not in lines[i]:
                i += 1

            # Skip dashed + header lines until second dashed line
            dash_count = 0
            while i < len(lines):
                if "-----" in lines[i]:
                    dash_count += 1
                    if dash_count == 2:
                        i += 1
                        break
                i += 1

            # Now read geometry
            geom = []
            while i < len(lines):
                if "-----" in lines[i]:
                    break
                parts = lines[i].split()
                if len(parts) >= 6:
                    try:
                        atomic_number = int(parts[1])
                        x, y, z = map(float, parts[3:6])
                        element = PERIODIC_TABLE.get(atomic_number, f"X{atomic_number}")
                        geom.append((element, x, y, z))
                    except ValueError:
                        pass  # skip lines that can't be parsed
                i += 1

            if geom:
                geometries.append(geom)

        i += 1

    if not geometries:
        return None

    return geometries[-1]



def write_hse_gjf(mol_name, geometry):
    gjf_path = os.path.join(HSE_GJF_DIR, f"{mol_name}.gjf")

    with open(gjf_path, "w") as f:
        f.write(f"%nprocshared=2\n")
        f.write(f"{METHOD_LINE}\n\n")
        f.write(f"{mol_name}\n\n")
        f.write(f"{CHARGE} {MULTIPLICITY}\n")

        for elem, x, y, z in geometry:
            f.write(f"{elem:<2} {x:>12.6f} {y:>12.6f} {z:>12.6f}\n")

        f.write("\n")

# =========================
# MAIN LOOP
# =========================
failed = []

for filename in os.listdir(PBE_OUT_DIR):
    if not filename.endswith(".out"):
        continue

    mol_name = os.path.splitext(filename)[0]
    out_path = os.path.join(PBE_OUT_DIR, filename)

    with open(out_path, "r", errors="ignore") as f:
        text = f.read()

    if not optimization_converged(text):
        print(f"[SKIP] Optimization not converged: {mol_name}")
        failed.append(mol_name)
        continue

    try:
        geometry = extract_final_geometry(text)
        if geometry is None:
            raise RuntimeError("No geometry found")

        write_hse_gjf(mol_name, geometry)
        print(f"[OK] Generated HSE input for {mol_name}")

    except Exception as e:
        print(f"[ERROR] {mol_name}: {e}")
        failed.append(mol_name)

print("\nSummary")
print("--------")
print(f"Total failed/skipped: {len(failed)}")
if failed:
    print("Failed molecules:", failed)
