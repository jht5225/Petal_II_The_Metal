import cv2
import numpy as np
import imageutilities as iu
import sys
from skimage import io
import image_shapes as shapes

def main():
    img2 = cv2.imread(sys.argv[1])
    im_in = shapes.get_vein_shape(img2)
    im_in = shapes.get_vein_shape(im_in)
    io.imshow(im_in)
    io.show()
main()
