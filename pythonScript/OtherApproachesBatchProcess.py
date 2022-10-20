from multiprocessing.util import is_exiting
import os
import json
import glob
import math
import getpass
import subprocess
from os.path import exists
from OtherApproachesCommon import *

def batchOthers(exePath : str, CWFDataFolder : str):
    allModelFolders = [os.path.join(CWFDataFolder, o) for o in os.listdir(CWFDataFolder) if os.path.isdir(os.path.join(CWFDataFolder, o))]
    for modelFolder in allModelFolders:
        jsonPath = os.path.join(modelFolder, "data.json")
        args = [exePath, "-i", jsonPath]
        print(args)
        try:
            cmd = subprocess.check_output(args, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(["run time error"])
            pass
        
if __name__ == '__main__':
    batchOthers(CWFEXEPath, CWFDataFolder)