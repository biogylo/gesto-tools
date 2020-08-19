## Given a picture and its face landmarks, it should put 4 on the center line.
## then rotate the picture along 4 so that 3 and 1 are horizontal
import cv2
import numpy as np
import config as cfg
import skimage.draw
from imutils import face_utils
import argparse
import imutils
import dlib

detector = dlib.get_frontal_face_detector()
cnndetector = dlib.cnn_face_detection_model_v1(cfg.FACE_DETECTOR_LOCATION)

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def get_blurryness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def get_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_face(img): ## gets the bounding box for rectangles
    ## Returns a dlib rectangle, or false if no face was found
    faces = detector(get_gray(img))

    if len(faces) == 1:
        return faces[0]
    else:
        return len(faces)

def get_landmarks(img,face): #Face is the dlib rectangle, img the original face
    return face_utils.shape_to_np(predictor(get_gray(img),face))

predictor = dlib.shape_predictor(cfg.FACE_SHAPE_PREDICTOR_LOCATION)
IDEAL_IMG = cv2.imread('narrow-ideal.bmp')
IDEAL_FACE = get_face(IDEAL_IMG)
IDEAL_SIZE = IDEAL_IMG.shape
BW_SIZE = IDEAL_SIZE[0:2]


EXPECTED_LOCATIONS = get_landmarks(IDEAL_IMG,IDEAL_FACE)
IMPORTANT_POINTS = np.array((9,37,46)) - 1 #For more info on the points, check landmarks.png

for lm in EXPECTED_LOCATIONS[IMPORTANT_POINTS]:
    cv2.circle(IDEAL_IMG,tuple(lm),1,(0),2)

cv2.imshow('',IDEAL_IMG)
cv2.waitKey(500)
cv2.destroyAllWindows()
print(EXPECTED_LOCATIONS[IMPORTANT_POINTS])


pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))]) # Pad for translation on affine transformation
unpad = lambda x: x[:,:-1]


clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))

def get_fixed_img(img,equalize=True,isolate=True,show_landmarks=False): #Receives an openCV images and fixes the face
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[:,:,0] = clahe.apply(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

    face = get_face(img)

    if isinstance(face,int):
        return None

    landmarks = get_landmarks(img,face)
    rows,cols,ch = img.shape
    M = cv2.getAffineTransform(np.float32(landmarks[IMPORTANT_POINTS,:]),np.float32(EXPECTED_LOCATIONS[IMPORTANT_POINTS,:]))
    fixed_img = cv2.warpAffine(img,M,(rows,cols))
    cropped_img = fixed_img[0:IDEAL_SIZE[0],0:IDEAL_SIZE[1],:] #cropped image

    #ycr_img[:,:,0] = clahe.apply(ycr_img[:,:,0])#cv2.equalizeHist(ycr_img[:,:,0])

    fixed_landmarks = np.dot(pad(landmarks),M.T).astype(np.int32)
    landmarks = np.copy(fixed_landmarks)
    fixed_landmarks[17:27,1] -= 50

    if isolate:
        outline = fixed_landmarks[[*range(17), *range(26,16,-1)]]

        y,x = skimage.draw.polygon(outline[:,1], outline[:,0])

        x = np.clip(x,0,BW_SIZE[1]-1)
        y = np.clip(y,0,BW_SIZE[0]-1)

        isolated_img = np.zeros(IDEAL_SIZE, dtype=np.uint8)
        isolated_img[y,x,:] = cropped_img[y,x,:]
        if show_landmarks:
            for lm in landmarks:
                cv2.circle(isolated_img,tuple(lm),1,(0),2)
        return isolated_img

    else:
        return cropped_img
