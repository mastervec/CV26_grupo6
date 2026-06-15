import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
ca1 = cv.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
if not ca1.isOpened():
    print("Cannot open camera")
    exit()
    
print('entrando no while')
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    ret1, frame1 = ca1.read()
    # if frame is read correctly ret is True
    print('definições corretas')
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    if not ret1:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    print('pegou imagem')
    # Display the resulting frame
    cv.imshow('frame',cv.hconcat([frame, frame1]))
    
    
    if cv.waitKey(1) == ord('x'):
        cv.imwrite('imagem junta.jpg', cv.hconcat([frame, frame1]))
        break
        
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
ca1.release()
cv.destroyAllWindows()
