# import packages
import argparse # handle parsing commandline arguments
import cv2 # opencv

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", # short-form for the switch/commandline argument
                "--image", # long-form for the switch/commandline argument
                required=True, # a value must be provided
                help="path to image")
ap.add_argument("-s", # short-form for the switch/commandline argument
                "--save", # long-form for the switch/commandline argument
                required=False, # optional
                help="should the file be saved back as a jpeg?")

args = vars(ap.parse_args())

# load the image and show some basic information on it
image = cv2.imread(args["image"]) # load the image
print("width: %d pixels" % (image.shape[1])) # print out the width of image
print("height: %d pixels" % (image.shape[0])) # print out the length of image
print("channels: %d" % (image.shape[2])) # print out the channel count

# show the image and wait for a keypress
# cv2.imshow("Image", image)
# cv2.waitKey(0)

# OpenCV handles converting filetypes automatically

if args["save"] is not None:
    cv2.imwrite(args["save"], image)
