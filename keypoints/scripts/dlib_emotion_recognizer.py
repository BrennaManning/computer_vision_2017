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
import cv 

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
            
            for files in file_list:
                current_session = files[20:-30]
                file = open(filepath+'/' +files, 'r')
                
                emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.

                session_image_path = "source_images/" + str(part) + "/" + str(session)
                
                dest_image_path = "ck-sorted/" + str(emotions[emotion])
                dest_neutral_image_path = "ck-sorted/neutral"
                
                source_session_emotion = os.listdir(session_image_path)[-1]
                
                source_session_neutral = os.listdir(session_image_path)[0]
                dest_session_neutral = "ck-sorted/neutral/" + str(source_session_neutral)[25:] #file path to sorted neutral image
                
                source_session_emotion_path = session_image_path + "/" + source_session_emotion
                dest_session_emotion_path = dest_image_path + "/" + source_session_emotion

                source_session_neutral_path = session_image_path + "/" + source_session_neutral
                dest_session_neutral_path = dest_neutral_image_path + "/" + source_session_neutral

                print ("source: ", source_session_emotion_path)
                print ("dest: ", dest_session_emotion_path)
                copyfile(source_session_emotion_path, dest_session_emotion_path) #Copy file
                
                print ("source: ", source_session_neutral_path)
                print ("dest: ", dest_session_neutral_path)
                copyfile(source_session_neutral_path, dest_session_neutral_path) #Copy file

def create_mini_images():   
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    for emotion in emotions:
        source_path = "ck-sorted/"+emotion
        write_path = "mini_images/"+emotion
        for image in os.listdir(source_path):
            original = cv2.imread(source_path + "/" + image)
            try:
                original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            except:
                pass
            casc = "haarcascade_frontalface_default.xml"
            cropped = crop_face(original, casc)
            if cropped is not None:
                small = cv2.resize(cropped, (24,24))
                cv2.imwrite(write_path + "/" + image, small)


def crop_face(image, cascPath):
    """
    http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html

    """
    faceCascade = cv2.CascadeClassifier(cascPath)
    faces = faceCascade.detectMultiScale(image)
    if len(faces) == 1:
        for (x, y, w, h) in faces:
            crop = image[y:y+h, x:x+w]
            return crop
    else:
        print "no face"
        return None
   

class Face(object):
    """ Object for storing face data."""
    def __init__(self):
        self.image = None
        self.clahe_image = None
        self.detections = None
        self.predicted_landmarks = None
        self.xcenter = 0
        self.ycenter = 0
        self.vector_lengths = []
        self.vector_angles = []
        self.emotion = None
        self.vectors = []

def get_landmark_vectors(face, image, predictor, detector):
    # Landmark Vector Info  
    vector_lengths = [] # Euclidean distance from each keypoint to the center
    vector_angles = [] # Corrected for offset using nose bridge angle
    detections = detector(image, 1)
    face.detections = detections # get faces
    # If there is one face
    if len(detections) == 1:
        face.predicted_landmarks = predictor(image, face.detections[0]) # Get facial from dlib shape predictor
        # center point of face (center of gravity of landmark keypoints)
        xlist = [float(face.predicted_landmarks.part(i).x) for i in range (1,68)]
        ylist = [float(face.predicted_landmarks.part(i).y) for i in range (1,68)]
        face.xcenter = np.mean(xlist) 
        face.ycenter = np.mean(ylist)
        xdist = [(x-face.xcenter) for x in xlist] # x distance from point to center
        ydist = [(y-face.ycenter) for y in ylist] # y distance from point to center
        
        # Alternative to other head pose calculation methods
        # Determine and adjust for angular offset based on bridge of nose!
        # http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/
        # -----------------------------------------------------------------------------------
        
        # The 26th and 29th points in the correspond to the bridge of the nose on a face.
        # 29 = tip of nose; 26 = top of nose bridge.
        nose_rotation = 0
        if xlist[26] == xlist[29]: # Check to prevent dividing by 0  in calculation if they are the same.
            anglenose = 0 # No
        else:
            anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))) # Nose bridge angle in radians
            anglenose = (anglenose*360)/(2*math.pi)  # Convert nose bridge angle from radians to degrees.

        if anglenose < 0: #Get offset by finding how the nose bridge should be rotated to become perpendicular to the horizontal plane
            nose_rotation = anglenose + 90  
        else:
            nose_rotation = anglenose - 90

        # Iterate through all keypoints
        for x, y, w, z in zip(xdist, ydist, xlist, ylist):

            dist = np.sqrt(x**2 + y**2) # vector length of keypoint to center
            vector_lengths.append(dist)

            anglerelative = (math.atan((z-np.mean(ylist))/(w-np.mean(xlist))*180/math.pi) - nose_rotation)
            vector_angles.append(anglerelative) # angle of keypoint -> center point angle adjusted for nose rotation.
        
        face.vector_lengths.append(vector_lengths)
        face.vector_angles.append(vector_angles)
        face.vectors.append((vector_lengths, vector_angles))

    elif len(detections) < 1: 
        print "no faces detected"
    else:
        print "too many faces!!!"

    return face

def draw(frame, face):
    """ Draws keypoints and center of mass on ace image."""
    landmarks = face.predicted_landmarks
    # features
    for i in range(1, 68):
        cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (0,0,255), thickness=2)
    # center of mass
    cv2.circle(frame, (int(face.xcenter), int(face.ycenter)), 1, (255, 0, 0), thickness = 3)
    
    return frame


def save(image_data, filepath):
    if not os.path.isfile(filepath):
         with open(filepath, 'wb') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    wr.writerow('', '', '', )
                    for row in image_data:
                        wr.writerow(row)
    else:
        pass

def run():
    detector = dlib.get_frontal_face_detector() # Face detector
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # Landmark identifier

    while True:
        #video_capture = cv2.VideoCapture(0) # webcam object
        #ret, self.frame = self.video_capture.read()
        img_list = ["test.png"]
        for img_path in img_list:
            print "IMAGE"

            face = Face()
            frame = cv2.imread(img_path)
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = gray.copy()
                
            except:
                pass
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_image = clahe.apply(frame)
            detections = detector(frame, 1)
            face = get_landmark_vectors(face=face, image=clahe_image, predictor=predictor, detector=detector)
            
            frame = draw(frame, face)
            
            cv2.imshow("image", frame)
            cv2.waitKey(0)
            print(face.vectors)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                print "================================="
                print "Quitting.. Saving data to .CSV"
                print "================================="
                with open('../all_landmarks_vectorized.csv', 'wb') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    for row in face.vectors:
                        wr.writerow(row)
                break

#node = EmotionRecognizerNode()
#node.run()
run()