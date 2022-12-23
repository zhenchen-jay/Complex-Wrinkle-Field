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
    # outFolder = "/home/zchen96/Projects/CWFRes/"
    outFolder = CWFDataFolder
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)
    outFolder = outFolder + 'polyimags'
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)
    for modelFolder in allModelFolders:
        modelName = os.path.split(modelFolder)[-1]
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
    
    parentFolder = os.path.join(CWFDataFolder, os.pardir) 
    outFolder = os.path.join(parentFolder, "videos")
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)
    outFolder = os.path.join(outFolder, "CWFVideos")
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)

    for modelFolder in allModelFolders:
        modelName = modelFolder.split('/')[-1]
        print(modelName)
        imgsPath = os.path.join(modelFolder, "rendered_CWF")

        ffmpeg2Video(imgsPath, outFolder, modelName, "wrinkledMesh")
        ffmpeg2Video(imgsPath, outFolder, modelName, "amp")
        ffmpeg2Video(imgsPath, outFolder, modelName, "phi")

otherMethodList = ['Knoppel', 'linear', 'TFW'] 
def batchRenderedImg2VideoOtherApproaches(otherFolder : str):
    allModelFolders = [os.path.join(otherFolder, o) for o in os.listdir(otherFolder) if os.path.isdir(os.path.join(otherFolder, o))]
    parentFolder = os.path.join(otherFolder, os.pardir) 
    outFolder = os.path.join(parentFolder, "videos")
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)
    
    for modelFolder in allModelFolders:
        modelName = os.path.split(modelFolder)[-1]
        print(modelName)
        for method in otherMethodList:
            videoFolder = os.path.join(outFolder, method+"Videos")
            if not os.path.isdir(videoFolder):
                os.mkdir(videoFolder)
            methodImagsFolder = os.path.join(os.path.join(modelFolder, method + 'Res'), 'rendered_' + method)
            ffmpeg2Video(methodImagsFolder, videoFolder, modelName, "wrinkledMesh")
            ffmpeg2Video(methodImagsFolder, videoFolder, modelName, "amp")
            ffmpeg2Video(methodImagsFolder, videoFolder, modelName, "phi")

if __name__ == '__main__':
    CWFDataFolder = "/media/zchen96/Extreme SSD/CWF_Dataset/paperResRerunNewFormula_final/"
    otherFolder = "/media/zchen96/Extreme SSD/CWF_Dataset/otherApproaches/"
    # batchImg2Video(CWFDataFolder)
    # batchRenderedImg2Video(CWFDataFolder)
    batchRenderedImg2VideoOtherApproaches(otherFolder)