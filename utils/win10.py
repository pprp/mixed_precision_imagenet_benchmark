import os 
import shutil 

path1 = r"D:\train"
path2 = r"D:\val"
topath = r"D:\valid"

for i in os.listdir(path1):
    print(i)
    newpath = os.path.join(path2, i)
    print(newpath)
    to_path = os.path.join(topath, i)
    shutil.copytree(newpath, to_path)