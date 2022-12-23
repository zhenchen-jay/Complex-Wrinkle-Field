from multiprocessing.util import is_exiting
import os
import json
import glob
import math
import getpass
import subprocess
from os.path import exists
from CWFCommon import *

frameList = [10, 25, 100, 200]

def batchCWFUpsampling(exePath : str, CWFDataFolder : str):
    for frame in frameList:
        frameFolder = os.path.join(CWFDataFolder, 'frame' + str(frame))
        allModelFolders = [os.path.join(frameFolder, o) for o in os.listdir(frameFolder) if os.path.isdir(os.path.join(frameFolder, o))]
        for modelFolder in allModelFolders:
            jsonPath = os.path.join(modelFolder, "data.json")
            # if modelFolder.find("pantasma") == -1 and modelFolder.find("face") == -1:
            #     continue
            args = [exePath, "-i", jsonPath]
            print(args)
            try:
                cmd = subprocess.check_output(args, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                print(["run time error"])
                pass

def batchCWF(exePath : str, CWFDataFolder : str):
    for frame in frameList:
        frameFolder = os.path.join(CWFDataFolder, 'frame' + str(frame))
        allModelFolders = [os.path.join(frameFolder, o) for o in os.listdir(frameFolder) if os.path.isdir(os.path.join(frameFolder, o))]
        for modelFolder in allModelFolders:
            jsonPath = os.path.join(modelFolder, "data.json")
            args = [exePath, "-i", jsonPath, "-r"]
            if modelFolder.find('cylinder') != -1:
                continue
            print(args)
            try:
                cmd = subprocess.check_output(args, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                print(["run time error"])
                pass
        
if __name__ == '__main__':
    # batchCWFUpsampling(CWFEXEPath, "/mnt/spinning1/zchen/WrinkleEdition_dataset/paperResRerunNewFormula_1000/")
    batchCWF(CWFEXEPath, '/media/zchen96/Extreme SSD/keyFrameCheck/')