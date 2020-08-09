import glob

def writeList(arkLoc: str, listFile: str, listLabel: str):
    files = glob.glob(arkLoc)
    print(files)
    listF = open(listFile, "w")
    allFiles = "\n".join(files)
    listF.write(allFiles)
    fileNames = [" ".join(i.split("/")[-1].split(".")[1].split("_"))+"\n" for i in files]
    listL = open(listLabel, "w")
    listL.writelines(fileNames)


writeList("/home/dhruva/Desktop/CopyCat/Deep-Learning-for-ASLR/data/ark/*", "lists/train.data", "lists/train.en")