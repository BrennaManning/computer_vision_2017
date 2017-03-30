"""
dlib test

Learning how to used dlib.
Used tutorials: 
http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/
http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
by __author__ = "Paul van Gent, 2016"

and also referenced 
opencv dlib documentation https://www.learnopencv.com/tag/dlib/


"""

import cv2
import dlib
import numpy as np
import math
import csv
import sklearn


class EmotionRecognizerNode(object):
    def __init__(self):
        self.image = None
        self.clahe_image = None
        self.video_capture = cv2.VideoCapture(0) #Webcam object
        self.detector = dlib.get_frontal_face_detector() #Face detector
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
        self.detections = None
        self.xmean = 0
        self.ymean = 0
        self.all_landmarks_vectorized = []
        
    # def get_landmarks(self, image):
    #     detections = self.detector(image, 1)
    #     for k,d in enumerate(detections): #For all detected face instances individually
    #         shape = self.predictor(image, d) #Draw Facial Landmarks with the predictor class
    #         xlist = []
    #         ylist = []
    #         for i in range(1,68): #Store X and Y coordinates in two lists
    #             xlist.append(float(shape.part(i).x))
    #             ylist.append(float(shape.part(i).y))
                
    #         for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
    #             landmarks.append(x)
    #             landmarks.append(y)
    #     if len(detections) > 0:
    #         return landmarks
    #     else: #If no faces are detected, return error message to other function to handle
    #         landmarks = "error"
    #         return landmarks


    def get_landmarks_vectorized(self, image):
        print "landmarks_vectorized 1"
        landmarks_vectorized = []
        print landmarks_vectorized
        detections = self.detector(image, 1)
        for k,d in enumerate(detections): # for each face detected (if more than one)
            predicted_landmarks = self.predictor(image, d) #Draw Facial Landmarks with the predictor class
            xlist = []
            ylist = []
            for i in range(1,68): #Store X and Y coordinates in two lists
                xlist.append(float(predicted_landmarks.part(i).x))
                ylist.append(float(predicted_landmarks.part(i).y))
                
            self.xmean = np.mean(xlist) #Find both coordinates of centre of gravity
            self.ymean = np.mean(ylist)
            xcentral = [(x-self.xmean) for x in xlist] # x distance from point to center
            ycentral = [(y-self.ymean) for y in ylist] # y distance from point to center
            
            if xlist[26] == xlist[29]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
                anglenose = 0
            else:
                anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi) #point 29 is the tip of the nose, point 26 is the top of the nose brigde

            if anglenose < 0: #Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
                anglenose += 90
            else:
                anglenose -= 90

            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorized.append(x) #Add the coordinates relative to the centre of gravity
                landmarks_vectorized.append(y)
                print "landmarks_vectorized 2"
                print landmarks_vectorized
                #Get the euclidean distance between each point and the centre point (the vector length)
                meannp = np.asarray((self.ymean,self.xmean))
                coornp = np.asarray((z,w))
                dist = np.linalg.norm(coornp-meannp)
                landmarks_vectorized.append(dist)

                #Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde has when the face is not perfectly horizontal
                anglerelative = (math.atan((z-self.ymean)/(w-self.xmean))*180/math.pi) - anglenose
                landmarks_vectorized.append(anglerelative)
                
        if len(detections) < 1: 
            landmarks_vectorized = "error" #If no face is detected set the data to value "error" to catch detection errors
        
        self.all_landmarks_vectorized.append(landmarks_vectorized)
        return landmarks_vectorized

    def draw(self, landmarks):
        """ Draws keypoint and center of mass on webcam face image."""
        # features
        for i in range(1, 68):
            cv2.circle(self.frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (0,0,255), thickness=2)
        # center of mass
        cv2.circle(self.frame, (int(self.xmean), int(self.ymean)), 1, (255, 0, 0), thickness = 3)
        
        
    

    def run(self):
        while True:
            #ret, self.frame = self.video_capture.read()
            self.frame = cv2.imread("test.png")
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            self.clahe_image = clahe.apply(gray)
            landmark_vectors = self.get_landmarks_vectorized(self.clahe_image)
            
            self.detections = self.detector(self.clahe_image, 1) # detect faces
            for k,d in enumerate(self.detections):
                predicted_landmarks = self.predictor(self.clahe_image, d)
                self.draw(predicted_landmarks)
            cv2.imshow("image", self.frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                print "================================="
                print "Quitting.. Saving data to .CSV"
                print "================================="
                with open('../all_landmarks_vectorized.csv', 'wb') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    for row in self.all_landmarks_vectorized:
                        wr.writerow(row)
                break

node = EmotionRecognizerNode()
node.run()