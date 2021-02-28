# Necessary imports
import cv2
import dlib
import numpy as np
import os
import imutils
import glob


## set directories
os.chdir('C:\\Users\\User\\Desktop\\Facial-mask-overlay-with-OpenCV-Dlib')
paths = glob.glob('C:\\Users\\User\\Desktop\\Facial-mask-overlay-with-OpenCV-Dlib\\image\\*.jpg')

j = 7811

for path in paths:
    try:
        img= cv2.imread(path)
        img = imutils.resize(img, width = 500)
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # print(path)

        #Initialize color [color_type] = (Blue, Green, Red)
        color_blue = (254,207,110)
        color_white = (255,255,255)
        color_black = (0, 0, 0)

        # Initialize dlib's face detector
        detector = dlib.get_frontal_face_detector()

        """
        Detecting faces in the grayscale image and creating an object - faces to store the list of bounding rectangles coordinates
        The "1" in the second argument indicates that we should upsample the image 1 time.  
        This will make everything bigger and allow us to detect more faces
        """

        faces = detector(gray, 1)

        """
        Detecting facial landmarks using facial landmark predictor dlib.shape_predictor from dlib library
        This shape prediction method requires the file called "shape_predictor_68_face_landmarks.dat" to be downloaded
        Source of file: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        """
        # Path of file
        p = "C:\\Users\\User\\Desktop\\Facial-mask-overlay-with-OpenCV-Dlib\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat"
        # Initialize dlib's shape predictor
        predictor = dlib.shape_predictor(p)

        # Get the shape using the predictor
        # result_img = []
        for face in faces:
            landmarks = predictor(gray, face)

            points = []
            for i in range(1, 16):
                point = [landmarks.part(i).x, landmarks.part(i).y]
                points.append(point)
            # print(points)

            # Coordinates for the additional 5 points for wide, low coverage mask (lower nose points) - in sequence
            mask_e = [((landmarks.part(29).x), (landmarks.part(29).y))]

            fmask_e = points + mask_e

            fmask_e = np.array(fmask_e, dtype=np.int32)

            # mask_type = {1: fmask_a, 2: fmask_c, 3: fmask_e}
            # mask_type[choice2]


            # change parameter [mask_type] and color_type for various combination
            img2 = cv2.polylines(img, [fmask_e], True, (14,14,14), thickness=2, lineType=cv2.LINE_8)

            # Using Python OpenCV – cv2.fillPoly() method to fill mask
            # change parameter [mask_type] and color_type for various combination
            img3 = cv2.fillPoly(img2, [fmask_e], (14,14,14), lineType=cv2.LINE_AA)

        # cv2.imshow("image with mask outline", img2)
        # cv2.imshow("image with mask", img3)
        #Save the output file for testing
        # outputNameofImage = "output/test.jpg"
        # print("Saving output image to", outputNameofImage)
        
        cv2.imwrite('output/with_mask' + str(j) + '.jpg', img3)
        j+=1
    except AttributeError:
        print(path)
        os.remove(path)
        pass

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()