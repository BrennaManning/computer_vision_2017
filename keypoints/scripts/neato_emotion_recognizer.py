#!/usr/bin/env python

"""  Neato recognizes emotions and reacts. """

from geometry_msgs.msg import Vector3, Twist, Point
from std_msgs.msg import Header, ColorRGBA
import rospy
import math
import sys
import cv2
import dlib
import numpy as np
import sklearn
import os
import random
from sklearn.svm import SVC
import time


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

class EmotionNode(object):
    """
    Class for a emotion detection node
    """
    def __init__(self):
        """
        Initialize an EmotionNode object.
        Create publishers to publish /cmd_vel Twists.
        """
        rospy.init_node('emotion_detect')
        self.r = rospy.Rate(5)
        self.publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.closest_corner = None
        self.emotion = None

    def stop(self):
        """
        Publish a /cmd_vel Twist corresponding to not moving
        """
        self.publisher.publish(Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0)))

    def read_emotion():
        """
        Function to check if there is a face in view and if so
        what emotion the face is displaying.
        """

        pass


    

    def run(self):
        print "connected to neato"
        emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
        

        # GETTING CLF 
        clf = learning_SVM(1, emotions)

        # MAKING PREDICTION ON WEBCAM IMAGE
        video_capture = cv2.VideoCapture(0)
        last_time = 0
        rospy.on_shutdown(self.stop)
        while not rospy.is_shutdown():
            ret, frame = video_capture.read()
            curr_time = time.time()
            if curr_time - last_time > 3.5:
                face = get_landmark_vectors(frame)
                emotion_prediction = make_SVM_prediction(clf, face, emotions)
                self.emotion = emotion_prediction
                frame = draw(frame, face)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, emotion_prediction,(10,400), font, 4,(255,255,255),2)
                cv2.imshow("landmarks", frame)
                last_time = time.time()
                if cv2.waitKey(10) & 0xFF == ord('q'):
                        print "================================="
                        print "Quitting.."
                        print "================================="
                        break
            if self.emotion == "happy":
              # Spin when happy
                print "HAPPY!"
                twist = Twist(linear=Vector3(0.0,0.0,0.0), angular=Vector3(0.0,0.0,1.0))
                self.publisher.publish(twist)
            else:
                print "current emotion: ", self.emotion
                twist = Twist(linear=Vector3(0.0,0.0,0.0), angular=Vector3(0.0,0.0,0))

            self.r.sleep()


def process_image(image):
    # Make grayscale if not already grayscale
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = gray.copy()
    
    except:
        pass
    
    # Histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(image)
    return clahe_image

def get_landmark_vectors(image):
    face = Face()
    image = process_image(image)
    # Landmark Vector Info  
    vector_lengths = [] # Euclidean distance from each keypoint to the center
    vector_angles = [] # Corrected for offset using nose bridge angle
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # Landmark identifier
    detector = dlib.get_frontal_face_detector()
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
        for i in range(len(vector_lengths)):
            face.vectors.append(vector_lengths[i])
            face.vectors.append(vector_angles[i])

    elif len(detections) < 1: 
        print "no faces detected"
    else:
        print "too many faces!!!"

    return face

def draw(frame, face):
    """ Draws keypoints and center of mass on ace image."""
    if face.predicted_landmarks:
        landmarks = face.predicted_landmarks
        # features
        for i in range(1, 68):
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (0,0,255), thickness=2)
        # center of mass
        cv2.circle(frame, (int(face.xcenter), int(face.ycenter)), 1, (255, 0, 0), thickness = 3)
    
    return frame


def get_train_test_split(file_list):
    random.shuffle(file_list)
    training = file_list[:int(len(file_list)*0.8)]
    testing = file_list[int(len(file_list)*0.8):]
    return training, testing

def make_datasets(emotions, foldername="ck-sorted/"):
    print "Current Directory"
    print os.getcwd()
    print "making datasets"
    training_data = []
    training_labels = []
    testing_data = []
    testing_labels = []
    error_count = 0

    for emotion in emotions:
        path = foldername+emotion
        image_files = os.listdir(path)
        training, testing = get_train_test_split(image_files)

        for img in training:
            print emotion, "training ", img 
            img_path = foldername+emotion+"/"+img
            frame = cv2.imread(img_path)
            
            # Get vector data
            try:
                face = get_landmark_vectors(image=frame)
                training_data.append(face.vectors)
                training_labels.append(emotions.index(emotion))
            except RuntimeError as e:
                print "Error getting landmark vectors for image: ", img_path
                print e
                error_count = error_count +1

        for img in testing:
            print emotion, "testing", img
            img_path = foldername+emotion+"/"+img
            frame = cv2.imread(img_path)
            
            # Get vector data
            try:
                face = get_landmark_vectors(image=frame)
                testing_data.append(face.vectors)
                testing_labels.append(emotions.index(emotion))
            
            except RuntimeError as e:
                print "Error getting landmark vectors for image: ", img_path
                print e
                error_count = error_count +1
            
    print "TRAINING DATA SHAPE: ", np.asarray(training_data).shape
    print 'TESTING DATA SHAPE: ', np.asarray(testing_data).shape
    print "----------------------"
    print "TRAINING NP "
    print np.asarray(training_data)
    print "----------------------"
    return training_data, training_labels, testing_data, testing_labels



def learning_SVM(iterations, emotions):
    """ Based on http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/ """ 
    accur_lin = []
    clf = SVC(kernel='linear', probability=True, tol=1e-3) #Set the classifier as a support vector machines with polynomial kernel
    for i in range(iterations):
        training_data, training_labels, testing_data, testing_labels = make_datasets(emotions)
       
        np_training = np.array(training_data) # convert to numpy array for classifier
        np_training_labels = np.array(training_labels)
        #print "training SVM linear: ", i #train SVM
        clf.fit(np_training, training_labels)

        print("getting accuracies: ", i) #Use score() function to get accuracy
        np_test = np.array(testing_data)
        accuracy = clf.score(np_test, testing_labels)
        print "linear accuracy: ", accuracy
        accur_lin.append(accuracy) #Store accuracy in a list

    score = np.mean(accur_lin)
    print "SVM SCORE = ", score
    return clf

def make_SVM_prediction(clf, face, emotions):
    """ """ 
    
    if len(face.vectors) > 0:
        vectors = np.array(face.vectors).reshape(1, -1)
        prediction_num = np.asscalar(clf.predict(vectors))
        try:
            face.emotion = emotions[prediction_num]
        except Exception as e:
            face.emotion = "ERROR in face.emotion = emotions.index(prediction_num)"
            print e
        

        print face.emotion
        return face.emotion

            
if __name__ == "__main__":
    emotion_node = EmotionNode()
    emotion_node.run()
