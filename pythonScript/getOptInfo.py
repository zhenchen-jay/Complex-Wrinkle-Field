import glob
import argparse
import os
import glob

def arguments():
    a = argparse.ArgumentParser(
        description="Get CWF Opt Info",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    a.add_argument(
        "-i", "--input-file", type=str, required=True, help="input file",
    )

    args = a.parse_args()
    return args

def getCWFOptInfo(filePath : str):
    with open(filePath, 'r') as f:
        f = f.readlines()
        timeInfo = f[-1]
        timeInfo = timeInfo.split(',')
        assemblingTime = float(timeInfo[0].split()[-1])
        LLTTime = float(timeInfo[1].split()[-1])
        lineSearchTime = float(timeInfo[2].split()[-1])
        totalTime = assemblingTime + LLTTime + lineSearchTime
        totalIter = int(f[-6].split()[-1])
        print([totalIter, LLTTime, totalTime])



if __name__ == '__main__':
    args = arguments()
    getCWFOptInfo(args.input_file)
