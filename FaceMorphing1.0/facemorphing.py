'''
   It takes two facial images as input and returns a video morphing from the first image to the second.
'''
import datetime
import os
import numpy as np
from PIL import Image
import imageio
import dlib
import cv2
import sys

def create_gif(filenames, duration):
    """
    method to create gif from morphed images
    params:
	    create_gif(images,duration)
		images : two or more created morphed image names to form a gif
		duration : time duration for one image display
    """
    morphed_images = []
    for filename in filenames:
        morphed_images.append(imageio.imread(filename))
    gif = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    imageio.mimsave(gif, morphed_images, duration=duration)
def shape_to_landmarks(shape):
    """
	method to find landmarks in face
	shape_to_landmarks(shape, dtype)
	shape : detected face from dlib.get_frontal_face_detector()
    """
    facial_coordinates = []
    for i in range(0, 68):
        facial_coordinates.append((shape.part(i).x, shape.part(i).y))
    return facial_coordinates
def calculatedelaunaytriangles(rect, points):
    """
    method to calculate delaunay triangles from given face
	calculate delaunay triangles(rect, points)
	rect : face after prediction
	points : face landmarks
	"""
    #create subdiv
    subdiv = cv2.Subdiv2D(rect)
    # Insert points into subdiv
    for point in points:
        subdiv.insert(point)
    trianglelist = subdiv.getTriangleList()
    delaunaytri = []
    for triangle in trianglelist:
        triangle_pts = [(triangle[0], triangle[1]), (triangle[2], triangle[3]), (triangle[4], triangle[5])]
        ind = []
        for j in range(0, 3):
            for k in range(0, len(points)):
                if (abs(triangle_pts[j][0] - points[k][0]) < 1.0 and abs(triangle_pts[j][1] - points[k][1]) < 1.0):
                    ind.append(k)
        if len(ind) == 3:
            delaunaytri.append((ind[0], ind[1], ind[2]))
    return delaunaytri
def applyaffinetransform(source, sourcetriangles, destinationtriangles, size):
    """
	method to applyaffinetransform
	applyaffinetransform(source, sourcetriangles, destinationtriangles, size)
	params:
	    source : image
		sourcetriangles : triangles from source image
		destinationtriangles : triangles from destination image
	"""
    # Given a pair of triangles, find the affine transform.
    warpmat = cv2.getAffineTransform(np.float32(sourcetriangles), np.float32(destinationtriangles))
    # Apply the Affine Transform just found to the source image
    destination = cv2.warpAffine(source, warpmat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return destination
def morphtriangle(img1, img2, img, triangle1, triangle2, triangle, alpha):
    """
	method to morph the images based on triangles
	morphtriangle(img1, img2, img, triangle1, triangle2, t, alpha)
	params:
	    img1 : input image
		img2 : target image
		img : matrix of zeros of input image size
		triangle1 : landmarks of input image of first, second, third triangles
		triangle2 : landmarks of target image of first, second, third triangles
		triangle : combined landmarks of two images for alpha(value) of first, second, third triangles from
		alpha : value for amount of blend
	"""
    # Find bounding rectangle for each triangle
    rectangle1 = cv2.boundingRect(np.float32([triangle1]))
    rectangle2 = cv2.boundingRect(np.float32([triangle2]))
    rectangle = cv2.boundingRect(np.float32([triangle]))
    # Offset points by left top corner of the respective rectangles
    triangle1rect = []
    triangle2rect = []
    trianglerect = []
    for i in range(0, 3):
        trianglerect.append(((triangle[i][0] - rectangle[0]), (triangle[i][1] - rectangle[1])))
        triangle1rect.append(((triangle1[i][0] - rectangle1[0]), (triangle1[i][1] - rectangle1[1])))
        triangle2rect.append(((triangle2[i][0] - rectangle2[0]), (triangle2[i][1] - rectangle2[1])))
    # Get mask by filling triangle
    mask = np.zeros((rectangle[3], rectangle[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(trianglerect), (1.0, 1.0, 1.0), 16, 0)
    # Apply warpImage to small rectangular patches
    img1rect = img1[rectangle1[1]:rectangle1[1] + rectangle1[3], rectangle1[0]:rectangle1[0] + rectangle1[2]]
    img2rect = img2[rectangle2[1]:rectangle2[1] + rectangle2[3], rectangle2[0]:rectangle2[0] + rectangle2[2]]
    size = (rectangle[2], rectangle[3])
    warpimage1 = applyaffinetransform(img1rect, triangle1rect, trianglerect, size)
    warpimage2 = applyaffinetransform(img2rect, triangle2rect, trianglerect, size)
    # Alpha blend rectangular patches
    imgrect = (1.0 - alpha) * warpimage1 + alpha * warpimage2
    # Copy triangular region of the rectangular patch to the output image
    img[rectangle[1]:rectangle[1]+rectangle[3], rectangle[0]:rectangle[0]+rectangle[2]] = img[rectangle[1]:rectangle[1]+rectangle[3], rectangle[0]:rectangle[0]+rectangle[2]] * (1 - mask) + imgrect * mask
def croping1(filename, landmarks):
    """
    This function is used when the face is not in adequate size  
	method for croping1
	croping1(filename, landmarks)
	params:
	    filename : image name
		landmarks : facial landmarks
		setting coordinates to crop
 #landmarks:
	0  -  left ear point
	8  -  chin point
	16 -  right ear point
    27 -  nose bridge point
	"""
    img_shape = cv2.imread(filename)
    width = img_shape.shape[0]
    height = img_shape.shape[1]
    img = Image.open(filename)
    left = landmarks[27][0] - landmarks[0][0]
    top = landmarks[8][1] - landmarks[27][1]
    right = landmarks[16][0] - landmarks[27][0]
    if (landmarks[0][0] - left) - 50 < 0:
        left = 0
    else:
        left = (landmarks[0][0] - left) - 50
    if (landmarks[27][1] - top) - 50 < 0:
        top = 0
    else:
        top = (landmarks[27][1] - top) - 50
    if (landmarks[16][0] + right) + 50 > width:
        right = width
    else:
        right = (landmarks[16][0] + right) + 50
    if (landmarks[8][1] + top) + 100 > height:
        down = height
    else:
        down = (landmarks[8][1] + top) + 100
#saving cropped image
    img_croped = img.crop((left, top, right, down))
    img_croped.save("resized1.jpg")	
def croping2(filename, landmarks):
    """
    This function is used when the face is not in adequate size
	method for croping
	croping(filename, landmarks)
	params:
	    filename : image name
		landmarks : facial landmarks
		setting coordinates to crop
 #landmarks:
	0  -  left ear point
	8  -  chin point
	16 -  right ear point
    27 -  nose bridge point
	"""
    img_shape = cv2.imread(filename)
    width = img_shape.shape[0]
    height = img_shape.shape[1]
    img = Image.open(filename)
    left = landmarks[27][0] - landmarks[0][0]
    top = landmarks[8][1] - landmarks[27][1]
    right = landmarks[16][0] - landmarks[27][0]
    if (landmarks[0][0] - left) - 50 < 0:
        left = 0
    else:
        left = (landmarks[0][0] - left) - 50
    if (landmarks[27][1] - top) - 50 < 0:
        top = 0
    else:
        top = (landmarks[27][1] - top) - 50
    if (landmarks[16][0] + right) + 50 > width:
        right = width
    else:
        right = (landmarks[16][0] + right) + 50
    if (landmarks[8][1] + top) + 100 > height:
        down = height
    else:
        down = (landmarks[8][1] + top) + 100
#saving cropped image
    img_croped = img.crop((left, top, right, down))
    img_croped.save("resized2.jpg")
def matching(img1, img2):
    """
	This method is to find the matching percentage of input image and target image 
	matching(img1, img2)
	params:
	    img1 : source_img
		img2 : target_img
	returns:
	   matched_percentage
	"""
    i1 = Image.open(img1)
    i2 = Image.open(img2)
    pairs = zip(i1.getdata(), i2.getdata())
    if len(i1.getbands()) == 1:
        dif = sum(abs(p1 - p2) for p1, p2 in pairs)
    else:
        dif = sum(abs(c1 - c2) for p1, p2 in pairs for c1, c2 in zip(p1, p2))
    ncomponents = i1.size[0] * i1.size[1] * 3
    diff = (dif / 255.0 * 100) / ncomponents
    return 80 - diff
#image processing function
def process(source_img, target_img):
    """
	method to process the input image and target image
	process(source_img, target_img)
	params:
	    source_img : input image name
		target_img : target image name
	returns:
	    percentage_matched with target image
	"""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape.dat")
# Process 1st images
    img1 = Image.open(source_img)
    imag1 = cv2.imread(source_img)
    rects = detector(imag1, 1)
    if len(rects) > 1:
        raise IndexError
# Assume one face for now
    face_rect = rects[0]
    landmarks1 = shape_to_landmarks(predictor(imag1, face_rect))
    if landmarks1[16][0] - landmarks1[0][1] > (0.4 * imag1.shape[0]) and landmarks1[8][1] - landmarks1[27][1] < (0.35 * imag1.shape[1]):
        croping1(source_img, landmarks1)
        img1 = Image.open("resized1.jpg")
        img1 = img1.resize((500, 500))
        img1.save("resized1.jpg")
        image1 = cv2.imread("resized1.jpg")
    else:
        img1 = img1.resize((500, 500))
        img1.save("resized1.jpg")
        image1 = cv2.imread("resized1.jpg")
    rects = detector(image1, 1)
# Assume one face for now
    face_rect = rects[0]
    landmarks1 = shape_to_landmarks(predictor(image1, face_rect))
    size = image1.shape
    landmarks1.extend([(0, 0), (0, size[0] - 1), (size[1] - 1, 0), (size[1] - 1, size[0] - 1)])
    rect = (0, 0, size[1], size[0])
    triangles = calculatedelaunaytriangles(rect, landmarks1)
    # Process 2nd image
    image2 = cv2.imread(target_img)
    rects = detector(image2, 1)
    img2 = Image.open(target_img)
    if len(rects) > 1:
        raise IndexError
    face_rect = rects[0]
    landmarks2 = shape_to_landmarks(predictor(image2, face_rect))
    if landmarks2[16][0] - landmarks2[0][1] > (0.4 * image2.shape[0]) and landmarks2[8][1] - landmarks2[27][1] < (0.35 * image2.shape[1]):
        croping2(target_img, landmarks2)
        img2 = Image.open("resized2.jpg")
        img2 = img2.resize((500, 500))
        img2.save("resized2.jpg")
        image2 = cv2.imread("resized2.jpg")
    else:
        img2 = img2.resize((500, 500))
        img2.save("resized2.jpg")
        image2 = cv2.imread("resized2.jpg")
    rects = detector(image2, 1)
    face_rect = rects[0]
    landmarks2 = shape_to_landmarks(predictor(image2, face_rect))
    size = image2.shape
    landmarks2.extend([(0, 0), (0, size[0] - 1), (size[1] - 1, 0), (size[1] - 1, size[0] - 1)])
# Morph the images together
# Convert Mat to float data type
    img1 = np.float32(image1)
    img2 = np.float32(image2)
    morphed_images = []
    for i in range(0, 11):
        alpha = float(i)/10
        landmarks = []
    # Compute weighted average point coordinates
        for landmark in range(0, len(landmarks1)):
            x_coordinate = (1 - alpha) * landmarks1[landmark][0] + alpha * landmarks2[landmark][0]
            y_coordinate = (1 - alpha) * landmarks1[landmark][1] + alpha * landmarks2[landmark][1]
            landmarks.append((x_coordinate, y_coordinate))
    # Allocate space for final output
        imgmorph = np.zeros(img1.shape, dtype=img1.dtype)
        for triangle in triangles:
            first_triangle = int(triangle[0])
            second_triangle = int(triangle[1])
            third_triangle = int(triangle[2])
            triangle1 = [landmarks1[first_triangle], landmarks1[second_triangle], landmarks1[third_triangle]]
            triangle2 = [landmarks2[first_triangle], landmarks2[second_triangle], landmarks2[third_triangle]]
            triangle = [landmarks[first_triangle], landmarks[second_triangle], landmarks[third_triangle]]
        # Morph one triangle at a time.
            morphtriangle(img1, img2, imgmorph, triangle1, triangle2, triangle, alpha)
        morphed_images.append(np.uint8(imgmorph))
    images_morphed = []
    for num in range(len(morphed_images)):
        cv2.imwrite("morphed image{}.jpg".format(num), morphed_images[num])
        morphed_image_read = cv2.imread("morphed image{}.jpg".format(num))
        cv2.imwrite("morphed image{}.jpg".format(num), morphed_image_read)
        images_morphed.append("morphed image{}.jpg".format(num))
    percentage = matching(source_img, target_img)
    create_gif(images_morphed, 0.35)
    os.remove('morphed image0.jpg')
    os.remove('morphed image1.jpg')
    os.remove('morphed image2.jpg')
    os.remove('morphed image3.jpg')
    os.remove('morphed image4.jpg')
    os.remove('morphed image5.jpg')
    os.remove('morphed image6.jpg')
    os.remove('morphed image7.jpg')
    os.remove('morphed image8.jpg')
    os.remove('morphed image9.jpg')
    os.remove('morphed image10.jpg')
    os.remove('resized1.jpg')
    os.remove('resized2.jpg')
    return round(percentage)
try:
    #calling the process function
    percentage_matched = process(sys.argv[1], sys.argv[2])
    print("Percentage matched with target image is : ",percentage_matched)
except IndexError:
    print("zero or multiple faces detected please upload another image")
except ValueError:
    print("face not detected please upload another image")
except:
    print("please upload another picture")