import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv2
import os


def region_mask_scaled(imagesize, vcut=0.0, hcut=0.0):
    #vcut and hcut are in fractions - 0.01 = 1%
    #vcut = reduce height of mask
    #hcut = reduces breadth of mask base

    # magic values computed from solidWhiteRight.mp4
    # region_mask = [[50, 540], [450, 320], [500, 320], [920, 540]]
    ymax = 540
    xmax = 960
    bottom_left = [50/xmax, ymax/ymax]
    bottom_right = [920/xmax, ymax/ymax]
    top_right = [500/xmax, 320/ymax]
    top_left = [450/xmax, 320/ymax]

    y = imagesize[0]
    x = imagesize[1]
    bottom_left = [ (bottom_left[0] + hcut/2)*x, bottom_left[1]*y]
    bottom_right = [ (bottom_right[0] - hcut/2)*x, bottom_right[1]*y]
    top_right = [top_right[0]*x, (top_right[1]-vcut)*y]
    top_left = [top_left[0]*x, (top_left[1]-vcut)*y]

    v = [bottom_left, bottom_right, top_right, top_left]
    return(v)


class PhotuMask:
    type = None
    mask = None

    def __init__(self, type):
        self.type = type
        return(None)

    def create_color_mask(self, image, param):
        color_threshold = param  # for color-filtering [ , , ]
        mask = (image[:,:,0] < color_threshold[0]) \
            | (image[:,:,1] < color_threshold[1]) \
            | (image[:,:,2] < color_threshold[2])
        self.mask = mask
        return(self)

    def create_region_mask(self, image, param=None):
        if(param is not None):
            region = param  # [[], [], [], []]
        else:
            region = region_mask_scaled(image.shape)
        mask = np.zeros_like(image)

        if len(image.shape) > 2:
            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        vertices = np.array([region], dtype=np.int32) # is this necessary?

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        self.mask = mask #cv2.bitwise_and(image, mask)
        return(self)

    def create_invert_mask(self, image):
        self.mask = ~ self.mask
        return(self)

    def get_mask(self, image, param=None):
        if(self.type=='color'):
            newmask = self.create_color_mask(image, param).mask
        if(self.type=='region'):
            newmask = self.create_region_mask(image, param).mask

        self.mask = newmask
        return(self)


###########################################


class Photu:
    '''A class for image manipulation'''
    #self.path = None
    #self.image = None
    #self.image_last = None
    cascades = "./Photu/haar_cascades/"

    Photu_HAAR_EYE = cascades + "haarcascade_eye.xml"
    Photu_HAAR_EYEGLASSES = cascades + "haarcascade_eye_tree_eyeglasses.xml"
    Photu_HAAR_FRONTALFACE_ALT = cascades + "haarcascade_frontalface_alt.xml"
    Photu_HAAR_FRONTALFACE_ALT2 = cascades + "haarcascade_frontalface_alt2.xml"
    Photu_HAAR_FRONTALFACE = cascades + "haarcascade_frontalface_default.xml"
    Photu_HAAR_SMILE = cascades + "haarcascade_smile.xml"

    def read(self):
        self.image = cv2.imread(self.path)
        self.image_last = np.copy(self.image)
        return(self)

    def save(self, path):
        cv2.imwrite(path, self.image)
        return(self)

    def info(self):
        print('This image is: ',  type(self.image),
              'with dimensions:', self.image.shape)
        print('Dimension: ', self.image.shape)
        print('Current Path: ', os.path.dirname(os.path.realpath(__file__)))
        return(self)

    def __repr__(self):
        return('Dimension: ' + str(self.image.shape))

    def bgr2rgb(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return(img)

    def display(self, title="Default Title"):
        plt.imshow(self.bgr2rgb(self.image))
        plt.show()
        return(self)

    def plot(self):
        plt.imshow(self.bgr2rgb(self.image))
        plt.show()
        return(self)

    def plot2(self):
        res = np.hstack((self.image_last, self.image))
        plt.figure(figsize=(20,10))
        self.plotnp(res)
        return(self)

    def plotnp(self, img):
        plt.imshow(img)
        plt.show()
        return(self)

    def __init__(self, imgobj):
        if(isinstance(imgobj, str)):
            self.path = imgobj
            self.read()
        else:
            self.image = imgobj
            self.image_last = np.copy(self.image)
        return(None)

    def face_detector(self):
        self.image_last = np.copy(self.image)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        detector = cv2.CascadeClassifier(self.Photu_HAAR_FRONTALFACE)
        rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            cv2.rectangle(self.image, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
        return(self)

    def translate(self, x, y):
        M = np.float32([[1, 0, x], [0, 1, y]])
        self.image_last = np.copy(self.image)
        self.image = cv2.warpAffine(self.image, M,
                    (self.image.shape[1], self.image.shape[0]))
        return(self)

    def rotate(self, angle, center=None, scale=1.0):
        self.image_last = np.copy(self.image)
        
        # grab the dimensions of the image
        (h, w) = self.image.shape[:2]

        # if the center is None, initialize it as the center of
        # the image
        if center is None:
            center = (w // 2, h // 2)

        # perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        self.image = cv2.warpAffine(self.image, M, (w, h))

        # return the rotated image
        return(self)

# PhotuMask related stuff
class Photu(Photu):
    mask = None

    def create_invert_mask(self):
        self.mask = ~ self.mask
        return(self)

    def apply_mask(self, color=[0, 0, 0]):
        self.image[self.mask] = color
        return(self)

    def image_and_mask(self, color=[0, 0, 0]):
        '''bitwise and on image and mask
        replace image with this'''
        #self.image = self.image & self.mask
        self.image = cv2.bitwise_and(self.image, self.mask)
        return(self)

    def finalize(self):
        self.image = ~self.image & self.image_last
        return(self)

    def weighted_img(self, α=0.5, β=1.0, λ=0.0):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        self.image = cv2.addWeighted(self.image, α, self.image_last, β, λ)
        return(self)

    def create_mask(self, type, param=None):
        new_mask = PhotuMask(type)
        self.mask = new_mask.get_mask(self.image, param).mask
        return(self)


# Canny Edge Detection
class Photu(Photu):
    def make_gray(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        return(self)

    def gaussian_blur(self, kernel_size):
        self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        return(self)

    def canny(self, low_threshold, high_threshold):
        self.image = cv2.Canny(self.image, low_threshold, high_threshold)
        return(self)

# Hough Transform
class Photu(Photu):
    def hough(self, rho, theta, threshold, min_line_length, max_line_gap, color=[255, 0, 0]):
        line_image = np.copy(self.image)*0  #creating a blank to draw lines on

        # Run Hough on edge detected image
        lines = cv2.HoughLinesP(self.image, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        self.mask = lines

        # Iterate over the output "lines" and draw lines on the blank
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image, (x1,y1), (x2,y2), color, 10)

        # ToDo: Review if this is necessary
        # Create a "color" binary image to combine with line image
        self.image = np.dstack((self.image, self.image, self.image))
        line_image = np.dstack((line_image, line_image, line_image))

        # Draw the lines on the edge image
        self.image = cv2.addWeighted(self.image, 0.8, line_image, 1, 0)
        return(self)

# Doing Lines
def separate_lines(lines):
    """ Takes an array of hough lines and separates them by +/- slope.
        The y-axis is inverted in matplotlib, so the calculated positive slopes will be right
        lane lines and negative slopes will be left lanes. """
    right = []
    left = []
    for x1,y1,x2,y2 in lines[:, 0]:
        m = (float(y2) - y1) / (x2 - x1)
        if m >= 0:
            right.append([x1,y1,x2,y2,m])
        else:
            left.append([x1,y1,x2,y2,m])

    return right, left

def extend_point(x1, y1, x2, y2, length):
    """ Takes line endpoints and extroplates new endpoint by a specfic length"""
    line_len = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    x = x2 + (x2 - x1) / line_len * length
    y = y2 + (y2 - y1) / line_len * length
    return x, y

def reject_outliers(data, cutoff, threshold=0.08):
    """Reduces jitter by rejecting lines based on a hard cutoff range and outlier slope """
    data = np.array(data)
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    m = np.mean(data[:, 4], axis=0)
    return data[(data[:, 4] <= m+threshold) & (data[:, 4] >= m-threshold)]


def merge_lines(lines):
    """Merges all Hough lines by the mean of each endpoint,
       then extends them off across the image"""

    lines = np.array(lines)[:, :4] ## Drop last column (slope)

    x1,y1,x2,y2 = np.mean(lines, axis=0)
    x1e, y1e = extend_point(x1,y1,x2,y2, -1000) # bottom point
    x2e, y2e = extend_point(x1,y1,x2,y2, 1000)  # top point
    line = np.array([[x1e,y1e,x2e,y2e]])

    return np.array([line], dtype=np.int32)


class Photu(Photu):
    def mask_to_lines(self, color = [255, 255, 255]):
        right, left = separate_lines(self.mask)

        right = reject_outliers(right,  cutoff=(0.45, 0.75))
        right = merge_lines(right)
        left = reject_outliers(left, cutoff=(-0.85, -0.6))
        left = merge_lines(left)

        lines = np.concatenate((right, left))
        # Iterate over the output "lines" and draw lines on the blank
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(self.image, (x1,y1), (x2,y2), color, 10)

        #self.image = np.dstack((self.image, self.image, self.image))
        #Draw the lines on the edge image
        #self.image = cv2.addWeighted(self.image, 0.8, lines, 1, 0)
        return(self)
