import os

folder_path = "./pbe_log"

# List all files (ignore subfolders)
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
print("Number of files:", len(files))
#print(files)


