import os
import glob


DEBUG = True


for f in list(glob.glob("./*/*_augmented_*.jpg")) + list(glob.glob("./*/*_augmented_*.txt")):
	try:
		os.remove(f)
		if DEBUG:
			print(f"[Delete] {f}")
	except OSError:
		print("Can't delete ", f)
