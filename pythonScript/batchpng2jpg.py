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

def batchCWFRenderPNGProcess(CWFDataFolder : str):
    allModelFolders = [os.path.join(CWFDataFolder, o) for o in os.listdir(CWFDataFolder) if os.path.isdir(os.path.join(CWFDataFolder, o))]
    for modelFolder in allModelFolders:
        modelName = modelFolder.split('/')[-1]
        print(modelName)
        imgsPath = os.path.join(modelFolder, "rendered_CWF")

        args = ['python', 'C:/Users/csyzz/PhaseInterpolation_polyscope/pythonScript/png2jpeg.py', "-i", imgsPath]
        print(args)
        try:
            cmd = subprocess.check_output(args, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(["run time error"])
            pass

methodList = ["Knoppel", "linear", "TFW"]
def batchOtherApproachRenderPNGProcess(otherDataFolder : str):
    allModelFolders = [os.path.join(otherDataFolder, o) for o in os.listdir(otherDataFolder) if os.path.isdir(os.path.join(otherDataFolder, o))]
    for modelFolder in allModelFolders:
        modelName = modelFolder.split('/')[-1]
        print(modelName)
        for method in methodList:
            methodFolder = os.path.join(modelFolder, method + "Res")

            imgsPath = os.path.join(methodFolder, "rendered_" + method)

            args = ['python', 'C:/Users/csyzz/PhaseInterpolation_polyscope/pythonScript/png2jpeg.py', "-i", imgsPath]
            print(args)
            try:
                cmd = subprocess.check_output(args, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                print(["run time error"])
                pass

frameList = [0, 10, 25, 50, 100, 200]
def batchKeyframeRenderPNGProcess(keyframeDataFolder : str):
    for frame in frameList:
        frameFolder = os.path.join(keyframeDataFolder, 'frame'+str(frame))
        allModelFolders = [os.path.join(frameFolder, o) for o in os.listdir(frameFolder) if os.path.isdir(os.path.join(frameFolder, o))]
        for modelFolder in allModelFolders:
            modelName = modelFolder.split('/')[-1]
            print(modelName)
            imgsPath = os.path.join(modelFolder, "rendered_CWF")

            args = ['python', 'C:/Users/csyzz/PhaseInterpolation_polyscope/pythonScript/png2jpeg.py', "-i", imgsPath]
            print(args)
            try:
                cmd = subprocess.check_output(args, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                print(["run time error"])
                pass
        
if __name__ == '__main__':
    # batchPNGProcess()
    # batchCWFRenderPNGProcess('E:/CWF_Dataset/paperResRerunNewFormula_final')
    # batchOtherApproachRenderPNGProcess('E:/CWF_Dataset/otherApproaches')
    batchKeyframeRenderPNGProcess('E:/CWF_Dataset/keyFrameCheck')