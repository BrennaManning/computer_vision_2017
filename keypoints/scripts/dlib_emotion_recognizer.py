"""
dlib emotion recognizer 

Used tutorials: 
http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/
http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
by Paul van Gent, 2016

and also referenced 
opencv dlib documentation https://www.learnopencv.com/tag/dlib/

"""

import cv2
import dlib
import numpy as np
import math
import csv
import sklearn
import os
import glob
from shutil import copyfile

def sortfiles():
    """ This function for sorting through the ck database was taken from 
    the first Paul Vangent tutorial. """
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
    participants = [d for d in os.listdir('source_emotion') if os.path.isdir(os.path.join('source_emotion', d))]

    #participants = glob.glob("source_emotion\\*") #Returns a list of all folders with participant numbers
    
    #
    for x in participants:
        part = "%s" %x[-4:] #store current participant number
        
        session_list = [d for d in os.listdir('source_emotion/'+str(part)) if os.path.isdir(os.path.join('source_emotion/'+str(part), d))]


        for session in session_list: #Store list of sessions for current participant
            
            
            filepath = 'source_emotion/'+str(part)+'/'+str(session)
            file_list = os.listdir(filepath)
            print "FILE LIST"
            print file_list
            for files in file_list:
                print ""
                print files
                current_session = files[20:-30]
                file = open(filepath+'/' +files, 'r')
                
                emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
                
                sourcefile_emotion = glob.glob("source_images\\%s\\%s\\*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion
                sourcefile_neutral = glob.glob("source_images\\%s\\%s\\*" %(part, current_session))[0] #do same for neutral image
                
                dest_neut = "ck-sorted\\neutral\\%s" %sourcefile_neutral[25:] #Generate path to put neutral image
                dest_emot = "ck-sorted\\%s\\%s" %(emotions[emotion], sourcefile_emotion[25:]) #Do same for emotion containing image
                
                copyfile(sourcefile_neutral, dest_neut) #Copy file
                copyfile(sourcefile_emotion, dest_emot) #Copy file


class Face(object):
    """ Object for storing face data."""
    def __init__(self):
        self.x_distances = []
        self.y_distances = []
        self.vector_lengths = []
        self.vector_angles = []
        self.emotion = None


class EmotionRecognizerNode(object):
    """ Finding and saving face keypoint landmark vector data. """
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
        
    def get_landmarks_vectorized(self, image):
        landmarks_vectorized = []
        detections = self.detector(image, 1)
        if len(detections) == 1:
            predicted_landmarks = self.predictor(image, detections[0]) # Get facial from dlib shape predictor
           
            # center point of face (center of gravity of landmark keypoints)
            xlist = [float(predicted_landmarks.part(i).x) for i in range (1,68)]
            ylist = [float(predicted_landmarks.part(i).y) for i in range (1,68)]
            self.xmean = np.mean(xlist) 
            self.ymean = np.mean(ylist)
            xdist = [(x-self.xmean) for x in xlist] # x distance from point to center
            ydist = [(y-self.ymean) for y in ylist] # y distance from point to center
            
            # Alternative to other head pose calculation methods
            # Determine and adjust for angular offset based on bridge of nose!
            # http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/
            # -----------------------------------------------------------------------------------
            
            # The 26th and 29th points in the correspond to the bridge of the nose on a face.
            # 29 = tip of nose; 26 = top of nose bridge.
            if xlist[26] == xlist[29]: # Check to prevent dividing by 0  in calculation if they are the same.
                anglenose = 0 # No
            else:
                anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))) # Nose bridge angle in radians
                anglenose = (anglenose*360)/(2*math.pi)  # Convert nose bridge angle from radians to degrees.

            if anglenose < 0: #Get offset by finding how the nose bridge should be rotated to become perpendicular to the horizontal plane
                nose_rotation = anglenose + 90  
            else:
                nose_rotation = anglenose - 90


            # Landmark Vector Info 
            image_x_distances = [] # X distance from each keypoint to the center
            image_y_distances = [] # Y distance from each keypoint to the center
            image_vector_lengths = [] # Euclidean distance from each keypoint to the center
            image_vector_angles = [] # Corrected for offset using nose bridge angle

            # Iterate through all keypoints
            for x, y, w, z in zip(xdist, ydist, xlist, ylist):

                # To landmarks_vectorized:
                # append x distance from center
                # append y distance from center
                # append vector length
                # append relative angle

                image_x_distances.append(x) #keypoint x distance from center

                image_y_distances.append(y) # keypoint y distance from center

                dist = np.sqrt(x**2 + y**2) # vector length of keypoint to center
                image_vector_lengths.append(dist)

                anglerelative = (math.atan((z-self.ymean)/(w-self.xmean))*180/math.pi) - nose_rotation
                image_vector_angles.append(anglerelative) # angle of keypoint -> center point angle adjusted for nose rotation.
                

        elif len(detections) < 1: 
            print "no faces detected"
        else:
            print "too many faces!!!"

        image_landmark_vectors = [image_x_distances, image_y_distances, image_vector_lengths, image_vector_angles]

        return image_landmark_vectors

    def draw(self, landmarks):
        """ Draws keypoints and center of mass on ace image."""
        # features
        for i in range(1, 68):
            cv2.circle(self.frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (0,0,255), thickness=2)
        # center of mass
        cv2.circle(self.frame, (int(self.xmean), int(self.ymean)), 1, (255, 0, 0), thickness = 3)
        
        
    def save(self, image_data, filepath):
        if not os.path.isfile(filepath):
             with open(filepath, 'wb') as myfile:
                        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                        wr.writerow('', '', '', )
                        for row in self.all_landmarks_vectorized:
                            wr.writerow(row)
        else:
            pass

    def run(self):
        while True:
            #ret, self.frame = self.video_capture.read()
            img_list = ["test.png", "test2.png", "test3.png"]
            for img_path in img_list:
                self.frame = cv2.imread(img_path)
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

#node = EmotionRecognizerNode()
#node.run()
sortfiles()