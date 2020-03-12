# Petal_II_The_Metal

## image_alignment.py

This program allows the user to align an image of a single flower petal with the corresponding vein image. Running the file prompts the user to enter as command-line arguments the names of the two image files (original, then vein). It uses openCV to find the critical points and align the two images.

## image_shapes.py

Used to find the shape of the petal within the vein and original images. This makes it easy for image_alignment.py to compare the two and find a homography between them.

## gaussian.py

This program uses a Bayesian Gaussian mixture model to find the center and width of the spots present in the given image. Running the file prompts the user to enter the name of an image file to analyze.

## principlecomponent.py

pca_to_grey uses principal component analysis to convert the given image to greyscale by projecting the data onto the color axis that preserves the most variance.

create_point_cloud converts a greyscale image to a scatterplot where darker colors are represented by more points.

## imageutilities.py

Includes many helpful preprocessing functions that are called by shapetransform.py and gaussian.py. This includes resize, convert to greyscale, and brighten.
