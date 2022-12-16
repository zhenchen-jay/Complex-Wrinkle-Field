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
        args = ["ffmpeg", "-y", "-r", str(5), "-i", imgsPath  + "/output_%d.png", "-crf", str(10), outFolder + "/" + modelName + ".mp4"]
        print(args)
        try:
            cmd = subprocess.check_output(args, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(["run time error"])
            pass

def ffmpeg2Video(imgsPath : str, outFolder : str, modelName : str, suffix : str):
    prefix = modelName.split('_')[0]
    args = ["ffmpeg", "-y", "-r", str(20), "-i", imgsPath  + '/' + prefix + "_" + suffix + "_%d.png", "-crf", str(10), outFolder + "/" + modelName + "_" + suffix + ".mp4"]
    if not os.path.isfile(imgsPath  + '/' + prefix + "_wrinkledMesh_0.png"):
        args = ["ffmpeg", "-y", "-r", str(20), "-i", imgsPath  + '/' + prefix + "_" + suffix + "_back_%d.png" , "-crf", str(10), outFolder + "/" + modelName + "_" + suffix + ".mp4"]
    print(args)
    try:
        cmd = subprocess.check_output(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print(["run time error"])
        pass

def batchRenderedImg2Video(CWFDataFolder : str):
    allModelFolders = [os.path.join(CWFDataFolder, o) for o in os.listdir(CWFDataFolder) 
    if os.path.isdir(os.path.join(CWFDataFolder, o))]
    
    parentFolder = os.path.split(CWFDataFolder)[0]
    outFolder = parentFolder + "/CWFVideos/"
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)
    outFolder = outFolder + 'rendered'
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)

    for modelFolder in allModelFolders:
        modelName = modelFolder.split('/')[-1]
        print(modelName)
        imgsPath = os.path.join(modelFolder, "rendered_CWF")

        ffmpeg2Video(imgsPath, outFolder, modelName, "wrinkledMesh")
        ffmpeg2Video(imgsPath, outFolder, modelName, "amp")
        ffmpeg2Video(imgsPath, outFolder, modelName, "phi")

if __name__ == '__main__':
    CWFDataFolder = "/media/zchen96/Extreme SSD/CWF_Dataset/paperResRerunNewFormula_2000/"
    # batchImg2Video(CWFDataFolder)
    batchRenderedImg2Video(CWFDataFolder)