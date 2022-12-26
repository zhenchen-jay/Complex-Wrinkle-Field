# importing the modules
import os
import shutil


def copyFilesInFolder(source : str, target : str):
   allModelFolders = [os.path.join(source, o) for o in os.listdir(source) if os.path.isdir(os.path.join(source, o))]
   if not os.path.isdir(target):
      os.mkdir(target)
   for modelFolder in allModelFolders:
      if modelFolder.find("Videos") != -1 or modelFolder.find("polyimags") != -1:
         continue
      modelName = os.path.split(modelFolder)[-1]
      print(modelName)
      outFolder = os.path.join(target, modelName)
      if not os.path.isdir(outFolder):
         os.mkdir(outFolder)

      files = [o for o in os.listdir(modelFolder) if os.path.isfile(os.path.join(modelFolder, o))]
      for file in files:
         shutil.copy(os.path.join(modelFolder, file), os.path.join(outFolder, file))

def copyFilesInFolderNew(source : str, target : str):
   allModelFolders = [os.path.join(source, o) for o in os.listdir(source) if os.path.isdir(os.path.join(source, o))]
   if not os.path.isdir(target):
      os.mkdir(target)
   for frameFolder in allModelFolders:
      if frameFolder.find("Videos") != -1 or frameFolder.find("polyimags") != -1:
         continue
      modelName = os.path.split(frameFolder)[-1]
      print(modelName)
      tarframeFolder = os.path.join(target, modelName)
      if not os.path.isdir(tarframeFolder):
         os.mkdir(tarframeFolder)

      modelFolders = [os.path.join(frameFolder, o) for o in os.listdir(frameFolder) if os.path.isdir(os.path.join(frameFolder, o))]

      for modelFolder in modelFolders:   
         modelName = os.path.split(modelFolder)[-1]
         print(modelName)
         outFolder = os.path.join(tarframeFolder, modelName)
         if not os.path.isdir(outFolder):
            os.mkdir(outFolder)

         files = [o for o in os.listdir(modelFolder) if os.path.isfile(os.path.join(modelFolder, o))]
         for file in files:
            shutil.copy(os.path.join(modelFolder, file), os.path.join(outFolder, file))

if __name__ == '__main__':
   # copyFilesInFolder('/media/zchen96/Extreme SSD/CWF_Dataset/paperResRerunNewFormula_final', '/media/zchen96/Extreme SSD/CWF_Dataset/paperResRerunNewFormula_finalVersion')
   copyFilesInFolderNew('/media/zchen96/Extreme SSD/CWF_Dataset/keyFrameCheck', '/media/zchen96/Extreme SSD/CWF_Dataset/keyFrameCheck_finalVersion')