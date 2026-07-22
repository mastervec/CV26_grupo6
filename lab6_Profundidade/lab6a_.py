import numpy as np
import cv2
from matplotlib import pyplot as plt

# Reading the left and right images.
 
imgL = cv2.imread("im0.png")
imgR = cv2.imread("im1.png")

imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

dwidth = 720
dheight = 492
dim = (dwidth, dheight)
imgL_g = cv2.resize(imgL_gray, dim, interpolation= cv2.INTER_LINEAR)
imgR_g = cv2.resize(imgR_gray, dim, interpolation= cv2.INTER_LINEAR)

 
# Setting parameters for StereoSGBM algorithm
minDisparity = 0;
numDisparities = 64;
blockSize = 8;
disp12MaxDiff = 1;
uniquenessRatio = 10;
speckleWindowSize = 10;
speckleRange = 8;
 
# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange = speckleRange
    )
 
# Calculating disparith using the StereoSGBM algorithm
disp = stereo.compute(imgL_g, imgR_g)
disp32 = disp.astype(np.float32)
disparity = cv2.normalize(disp32,0,255,cv2.NORM_MINMAX)
 
# Displaying the disparity map
cv2.imshow("disparity",disparity)
cv2.waitKey(0)
