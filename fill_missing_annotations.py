import os
import re

pattern = r"^[\D]*"
xmin = float("infinity")
xmax = -float("infinity")
fileword = None
annotation_dir = "./annotations"


for file in os.listdir(annotation_dir):
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
		if xmin > number:
			xmin = number
		if xmax < number:
			xmax = number


exists = [False for _ in range(xmax - xmin + 1)]
for file in os.listdir(annotation_dir):
	exists[int(file[re.search(pattern, file).end():file.find(".")])] = True


for i in range(xmax - xmin + 1):
	if not exists[i]:
		with open(f"{annotation_dir}/{fileword}{xmin + i}.txt", "w") as wf:
			pass


