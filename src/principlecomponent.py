# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:04:44 2020

@authors: Abbey Felley, Jack Taylor
"""

import cv2
import numpy as np

def pca_to_grey(image):
    x,y,z = image.shape
    mat = image.reshape([x*y,z])

    mean, eigenvectors = cv2.PCACompute(mat, np.mean(mat, axis=0).reshape(1,-1))
    axis = eigenvectors[0,:].reshape([3])

    newmat = np.dot(mat, axis)
    newmat = np.around(newmat).astype(int)

    grey = newmat.reshape([x,y])
    return grey

def create_point_cloud(image):
    post_pca_image = pca_to_grey(image)

    rows, columns = post_pca_image.shape

    points_x = []
    points_y = []

    for x in range(rows):
        for y in range(columns):
            val = 255 - post_pca_image[x,y]
            val = val // 10

            for _ in range(val):
                points_x.append(x)
                points_y.append(y)

    point_cloud = np.array([points_x, points_y]).T
    return point_cloud
