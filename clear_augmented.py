import os
import glob
from tqdm import tqdm


files_list = list(glob.glob("./*/*_augmented_*.jpg")) + list(glob.glob("./*/*_augmented_*.txt"))
for i in tqdm(range(len(files_list)), desc="Deleting augmented files..."):
	f = files_list[i]
	try:
		os.remove(f)
	except OSError:
		print("Can't delete ", f)
