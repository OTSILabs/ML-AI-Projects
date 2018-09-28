Face Morphing:
This is a tool which creates a morphing effect. It takes two facial images as input and returns a video morphing from the first image to the second.

How to run:
python facemorphing.py path_to_first_image path_to_second_image
example: python facemerge.py input.jpg target.jpg
Using the above command you can generate a GIF and it is placed where your code exists.

How it works:
1. Find point-to-point correspondences between the two images using Dlib's Facial Landmark Detection.
2. Find the Delaunay Triangulation for the average of these points.
3. Using these corresponding triangles in both initial and final images, perform Wraping and Alpha Blending and obtain morphed images to be used in creating GIF.

Matching percentage betweeen given input and target image displays on prompt window.

Note:
To run this code we have to install some packages please go through the requirements.txt
