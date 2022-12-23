from multiprocessing.util import is_exiting
import os
import json
import glob
import math
import getpass
import subprocess
from os.path import exists
from OtherApproachesCommon import *

def getMethodRes(exePath: str, jsonPath: str, method: str):
    args = [exePath, "-i", jsonPath, "-m", method]
    print(args)
    try:
        cmd = subprocess.check_output(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print(["run time error"])
        pass

def batchOthers(exePath : str, CWFDataFolder : str):
    allModelFolders = [os.path.join(CWFDataFolder, o) for o in os.listdir(CWFDataFolder) if os.path.isdir(os.path.join(CWFDataFolder, o))]
    for modelFolder in allModelFolders:
        if modelFolder.find('bunny_globalRotation') != -1 or modelFolder.find("pantasma") != -1:
            continue

        jsonPath = os.path.join(modelFolder, "data.json")
        getMethodRes(exePath, jsonPath, "TFW")
        getMethodRes(exePath, jsonPath, "linear")
        getMethodRes(exePath, jsonPath, "knoppel")
        getMethodRes(exePath, jsonPath, "zuenko")

        
if __name__ == '__main__':
    batchOthers('/home/zchen96/Projects/PhaseInterpolation_polyscope/build/bin/OtherApproachesCli_bin', "/media/zchen96/Extreme SSD/otherApproaches/")