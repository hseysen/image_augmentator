import os
import re
from tqdm import tqdm


pattern = r"^[\D]*"
xmax = -float("infinity")
fileword = None
annotation_dir = "./annotations"


for file in os.listdir(f"./{annotation_dir}"):
	regexmatch = re.search(pattern, file)
	word = file[regexmatch.start():regexmatch.end()]

	if fileword is None:
		fileword = word
	else:
		if fileword != word:
			print("Please name your files and annotations systematically.")
			print(f"Could not parse the file '{file}' as it does not have a similar name to '{fileword}'.")
			quit()

	try:
		number = int(file[regexmatch.end():file.find(".")])
	except ValueError:
		print("Please name your files and annotations systematically.")
		print(f"Could not parse the file '{file}' as it does not follow the '[name][number].txt' format.")
		quit()
	else:
		if xmax < number:
			xmax = number


exists = [False for _ in range(xmax + 1)]
for file in os.listdir(f"./{annotation_dir}"):
	exists[int(file[re.search(pattern, file).end():file.find(".")])] = True


for i in tqdm(range(xmax + 1), desc="Creating missing annotations..."):
	if not exists[i]:
		with open(f"./{annotation_dir}/{fileword}{i}.txt", "w") as wf:
			pass


