import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import natsort
path="training/training"
im_file=os.listdir(path)
im_file=natsort.natsorted(im_file)
#info=pd.read_csv("training/solution.csv")

for f in im_file:
    im=cv2.imread(path+'/'+f)
    print(f)
    #im=cv2.resize(im,(img_dim[1],img_dim[0]))
    cv2.imshow("img",im)
    cv2.waitKey(0) and ord('q')
