from png2jpeg import *
from multiprocessing.util import is_exiting
import os
import json
import glob
import math
import getpass
import subprocess

imagFolder = '/mnt/spinning1/zchen/WrinkleEdition_dataset/Kinematic-Wrinkle-Edition-with-Complex-Representation/Figs/imag'

def batchPNGProcess():
    allModelFolders = [os.path.join(imagFolder, o) for o in os.listdir(imagFolder) if os.path.isdir(os.path.join(imagFolder, o))]
    for modelFolder in allModelFolders:
        args = ['python3', '/home/zchen96/Projects/PhaseInterpolation_polyscope/pythonScript/png2jpeg.py', "-i", modelFolder]
        print(args)
        try:
            cmd = subprocess.check_output(args, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(["run time error"])
            pass
        
if __name__ == '__main__':
    batchPNGProcess()