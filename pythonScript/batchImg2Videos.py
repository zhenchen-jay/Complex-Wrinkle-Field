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


# imgType: wrinkledMesh, amp, phi
def ffmpegPNG2Video(imgsPath : str, outFolder : str, modelName : str, imgType : str, suffix = ""):
    prefix = modelName.split('_')[0]
    args = ["ffmpeg", 
    "-y", 
    "-r", str(20), 
    "-i", imgsPath  + '/' + prefix + "_" + imgType + "_%d.png",
    "-vcodec", "qtrle",
    # "-pix_fmt", "argb", 
    # "-c:v", "libx264", 
    "-crf", str(15), 
    outFolder + "/" + modelName + "_" + imgType + suffix + ".mov"]
    if not os.path.isfile(imgsPath  + '/' + prefix + "_wrinkledMesh_0.png"):
        args = ["ffmpeg", 
        "-y", 
        "-r", str(20), 
        "-i", imgsPath  + '/' + prefix + "_" + imgType + "_back_%d.png",
        "-vcodec", "qtrle", 
        # "-pix_fmt", "argb", 
        # "-c:v", "libx264", 
        "-crf", str(15), 
        outFolder + "/" + modelName + "_" + imgType + suffix + ".mov"]
    print(args)
    try:
        cmd = subprocess.check_output(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print(["run time error"])
        pass

# imgType: wrinkledMesh, amp, phi
def ffmpegJPG2Video(imgsPath : str, outFolder : str, modelName : str, imgType : str, suffix = ""):
    prefix = modelName.split('_')[0]
    args = ["ffmpeg", 
    "-y", 
    "-r", str(20), 
    "-i", imgsPath  + '/' + prefix + "_" + imgType + "_%d.jpg",
    # "-vcodec", "qtrle",
    # "-pix_fmt", "argb", 
    "-c:v", "libx264", 
    "-crf", str(10), 
    outFolder + "/" + modelName + "_" + imgType + suffix + ".mp4"]
    if not os.path.isfile(imgsPath  + '/' + prefix + "_wrinkledMesh_0.jpg"):
        args = ["ffmpeg", 
        "-y", 
        "-r", str(20), 
        "-i", imgsPath  + '/' + prefix + "_" + imgType + "_back_%d.jpg",
        # "-vcodec", "qtrle", 
        # "-pix_fmt", "argb", 
        "-c:v", "libx264", 
        "-crf", str(10), 
        outFolder + "/" + modelName + "_" + imgType + suffix + ".mp4"]
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
        if modelName.find("hand") == -1: 
            continue
        print(modelName)
        imgsPath = os.path.join(modelFolder, "rendered_CWF")

        ffmpegJPG2Video(imgsPath, outFolder, modelName, "wrinkledMesh")
        ffmpegJPG2Video(imgsPath, outFolder, modelName, "amp")
        ffmpegJPG2Video(imgsPath, outFolder, modelName, "phi")

        ffmpegPNG2Video(imgsPath, outFolder, modelName, "wrinkledMesh")
        ffmpegPNG2Video(imgsPath, outFolder, modelName, "amp")
        ffmpegPNG2Video(imgsPath, outFolder, modelName, "phi")

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
            ffmpegJPG2Video(methodImagsFolder, videoFolder, modelName, "wrinkledMesh")
            ffmpegJPG2Video(methodImagsFolder, videoFolder, modelName, "amp")
            ffmpegJPG2Video(methodImagsFolder, videoFolder, modelName, "phi")

keyframeList = [0, 10, 25, 50, 100, 200]
def batchRenderImg2VideoKeyframes(keyframeFolder : str):
    parentFolder = os.path.join(keyframeFolder, os.pardir) 
    outFolder = os.path.join(parentFolder, "videos")
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)

    outFolder = os.path.join(outFolder, "keyframeAnalysis")
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)

    for keyframe in keyframeList:
        workingFolder = os.path.join(keyframeFolder, 'frame'+str(keyframe)) 
        allModelFolders = [os.path.join(workingFolder, o) for o in os.listdir(workingFolder) if os.path.isdir(os.path.join(workingFolder, o))]
        for modelFolder in allModelFolders:
            modelName = os.path.split(modelFolder)[-1]
            print(modelName)
            imgsPath = os.path.join(modelFolder, "rendered_CWF")

            ffmpegJPG2Video(imgsPath, outFolder, modelName, "wrinkledMesh", "_frame_" + str(keyframe))
            ffmpegJPG2Video(imgsPath, outFolder, modelName, "amp", "_frame_" + str(keyframe))
            ffmpegJPG2Video(imgsPath, outFolder, modelName, "phi", "_frame_" + str(keyframe))

if __name__ == '__main__':
    prefix = "E:/"
    CWFDataFolder = prefix + "CWF_Dataset/paperResRerunNewFormula_final/"
    otherFolder = prefix + "CWF_Dataset/otherApproaches/"
    keyframeFolder =  prefix + "CWF_Dataset/keyFrameCheck"
    # batchImg2Video(CWFDataFolder)
    batchRenderedImg2Video(CWFDataFolder)
    # batchRenderedImg2VideoOtherApproaches(otherFolder)
    # batchRenderImg2VideoKeyframes(keyframeFolder)