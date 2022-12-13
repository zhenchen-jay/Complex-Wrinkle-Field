from multiprocessing.util import is_exiting
import os
import json
import glob
import math
import getpass
import subprocess
from os.path import exists
from CWFCommon import *

def batchImg2Video(CWFDataFolder : str):
    allModelFolders = [os.path.join(CWFDataFolder, o) for o in os.listdir(CWFDataFolder) if os.path.isdir(os.path.join(CWFDataFolder, o))]
    outFolder = "/home/zchen96/Projects/CWFRes/"
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)
    outFolder = outFolder + 'polyimags'
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)
    for modelFolder in allModelFolders:
        modelName = modelFolder.split('/')[-1]
        print(modelName)
        imgsPath = os.path.join(modelFolder, "polyimags/CWFRes")
        args = ["ffmpeg", "-y", "-r", str(5), "-i", imgsPath  + "/output_%d.png", outFolder + "/" + modelName + ".mp4"]
        print(args)
        try:
            cmd = subprocess.check_output(args, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(["run time error"])
            pass
if __name__ == '__main__':
    batchImg2Video("/media/zchen96/Extreme SSD/paperResRerunNewFormula_2000/")