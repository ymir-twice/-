"""
    本文件用于创建GroundTrue. 运行一次就够了. 除非人为把 eval/test_ground_truths 中的文件删去, 否则没必要再运行
    本文件也不配备 .vscode task 配置
"""
import numpy as np
import pandas as pd
import cv2
import shutil
import os

def main():
    if os.path.isdir("eval/test_ground_truths"):
        shutil.rmtree("eval/test_ground_truths")
        os.mkdir("eval/test_ground_truths")

    testmeta = pd.read_csv("stats/testB-meta.csv", dtype=object)
    ids = list(testmeta['id'])
    for id in ids:
        imdata = cv2.imread("data/annotations/testB/" + id + '.png', cv2.IMREAD_GRAYSCALE)
        np.save("eval/test_ground_truths/ground_truth_" + id, imdata)

if __name__ == "__main__":
    main()