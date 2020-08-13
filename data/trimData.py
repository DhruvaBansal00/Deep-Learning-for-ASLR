import os
import sys
import glob

def trimData(arkFileLoc: str, removeNum: int):
	print(glob.glob(arkFileLoc))
	for file in glob.glob(arkFileLoc):
		currFileData = open(file, "r")
		file_lines = []
		for line in currFileData:
			curr_line_trimmed = line.split()[:-1*removeNum]
			file_lines.append(" ".join(curr_line_trimmed)+"\n")
		
		currFileData = open(file, "w")
		currFileData.writelines(file_lines)

trimData("ark/*.ark", 2)
