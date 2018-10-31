import sys
import argparse
import os
import cv2
#print(cv2.__version__)
if os.getcwd()[-3:] == 'src': #running from src/ directory
    base_path = os.getcwd()
else: #running from project directory
    base_path = os.getcwd() + '/src'
    
sys.path.append(base_path)

def extractImages(pathIn, pathOut, pathForLastImg):
    print("Extraction starts")
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      #print ('Read a new frame: ', success)
      if success:
          # can support at most 1e8 number of photos 
          cv2.imwrite( pathOut + "/%08d.jpg" % count, image)     # save frame as JPEG file
          count += 1
      else:
          cv2.imwrite( pathForLastImg + "/%08d.jpg" % count, image)
    print(count, " images have been generated")  
        
# pathIn = "test.mp4"
# pathOut = "video/"
# extractImages(pathIn, pathOut)

#main()