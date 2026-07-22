import numpy as np
import cv2

depth_thresh = 100.0 # Threshold for SAFE distance (in cm)
 
# Mask to segment regions with depth less than threshold
mask = cv2.inRange(depth_map,10,depth_thresh)
 
# Check if a significantly large obstacle is present and filter out smaller noisy regions
if np.sum(mask)/255.0 > 0.01*mask.shape[0]*mask.shape[1]:
 
  # Contour detection
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts = sorted(contours, key=cv2.contourArea, reverse=True)
   
  # Check if detected contour is significantly large (to avoid multiple tiny regions)
  if cv2.contourArea(cnts[0]) > 0.01*mask.shape[0]*mask.shape[1]:
 
    x,y,w,h = cv2.boundingRect(cnts[0])
 
    # finding average depth of region represented by the largest contour
    mask2 = np.zeros_like(mask)
    cv2.drawContours(mask2, cnts, 0, (255), -1)
 
    # Calculating the average depth of the object closer than the safe distance
    depth_mean, _ = cv2.meanStdDev(depth_map, mask=mask2)
     
    # Display warning text
    cv2.putText(output_canvas, "WARNING !", (x+5,y-40), 1, 2, (0,0,255), 2, 2)
    cv2.putText(output_canvas, "Object at", (x+5,y), 1, 2, (100,10,25), 2, 2)
    cv2.putText(output_canvas, "%.2f cm"%depth_mean, (x+5,y+40), 1, 2, (100,10,25), 2, 2)
 
else:
  cv2.putText(output_canvas, "SAFE!", (100,100),1,3,(0,255,0),2,3)
 
cv2.imshow('output_canvas',output_canvas)
