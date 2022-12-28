from PIL import Image
import argparse
import os
import glob

def arguments():
    a = argparse.ArgumentParser(
        description="PNG 2 JPEG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    a.add_argument(
        "-i", "--input-folder", type=str, required=True, help="input folder",
    )

    args = a.parse_args()
    return args

def png2jpeg(folderPath : str):
    allImgs = []
    if os.path.isfile(folderPath):
        allImgs.append(folderPath)
    else:
        allImgs.extend(glob.glob(os.path.join(folderPath, '*.png')))
    print(allImgs)
    for im in allImgs:
        im1 = Image.open(im)
        rgb_im =  Image.new("RGB", im1.size, (255,255,255))
        rgb_im.paste(im1,im1)
        savePath = im.replace('.png', '.jpg')
        rgb_im.save(savePath) 


if __name__ == '__main__':
    args = arguments()
    png2jpeg(args.input_folder)