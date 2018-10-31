import cv2
import argparse
import os
import functools
import sys
import cv2

if os.getcwd()[-3:] == 'src': #running from src/ directory
    base_path = os.getcwd()
else: #running from project directory
    base_path = os.getcwd() + '/src'
    
sys.path.append(base_path)

# Arguments
#dir_path = 'transfered'
#ext = 'jpg' #args['extension']
#output = 'output.mp4' #args['output']


def my_number_giving_function(a, b):
    aa = a[:-4]
    bb = b[:-4]
    return int(aa) > int(bb)

def jpg_to_video(dir_path, output, ext='.jpg'):
    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)


    tmp_image = [img[:-4] for img in images]
    #print(tmp_image)
    tmp_image.sort(key=int) 
    #images = sorted(images, key=functools.cmp_to_key(my_number_giving_function))
    images = [img + ext for img in tmp_image]

    #print(images)

    #Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    #cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for image in images:

        image_path = os.path.join(dir_path, image)
        #print(image)
        frame = cv2.imread(image_path)

        out.write(frame) # Write out frame to video

        #cv2.imshow('video',frame)
#         if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
#             break

    # Release everything if job is finished
    out.release()
#     cv2.destroyAllWindows()

    print("The output video is {}".format(output))