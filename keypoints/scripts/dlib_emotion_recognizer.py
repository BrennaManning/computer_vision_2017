"""
dlib emotion recognizer 

Primarily used tutorials: 
http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/
http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
by Paul van Gent, 2016
These were very comphrehensible and useful.
I used some of the same functions, but it was always clear what was happening.
From the comments it looks like another about deep learning will be published soon.

and also referenced 
opencv dlib documentation https://www.learnopencv.com/tag/dlib/
https://github.com/its-izhar/Emotion-Recognition-Using-SVMs
https://www.youtube.com/watch?v=KMHDl13ynbg
https://pdfs.semanticscholar.org/4dff/129a6f988d78c457ece463b774c3d81ac5c7.pdf
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

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
import random
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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

def sortfiles():
    """ This function for sorting through the ck database to be easier to use was taken from 
    the first Paul Vangent tutorial. This function is called once after the dataset is first downloaded."""

    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
    participants = [d for d in os.listdir('source_emotion') if os.path.isdir(os.path.join('source_emotion', d))]

    #participants = glob.glob("source_emotion\\*") #Returns a list of all folders with participant numbers
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

def process_image(image):
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


def save(image_data, filepath):
    if not os.path.isfile(filepath):
         with open(filepath, 'wb') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    wr.writerow('', '', '', )
                    for row in image_data:
                        wr.writerow(row)
    else:
        pass


def get_train_test_split(file_list):
    random.shuffle(file_list)
    training = file_list[:int(len(file_list)*0.8)]
    testing = file_list[int(len(file_list)*0.8):]
    return training, testing

def make_datasets(emotions, foldername="ck-sorted/"):
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
        #probabilities = clf.predict_proba(vectors)
        #print "PROBABILITIES"
        #print probabilities

        print "GUESS"

        print face.emotion


def learning_logistic_regression(iteraitons, emotions):
    pass

def learning_k_neighbors():
    KNeighborsClassifier(3)

def run():
    
    #while True:
    #video_capture = cv2.VideoCapture(0) # webcam object
    #ret, self.frame = self.video_capture.read()
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    emotions_subset = ["happy", "sadness"]
   
    video_capture = cv2.VideoCapture(0)
    last_time = 0
    while True:
        curr_time = time.time()
        ret, frame = video_capture.read()

        if curr_time - last_time > 3:
            print "curr_time - last_time = ", curr_time-last_time
            
            cv2.imshow("webcam", frame)
            face = get_landmark_vectors(frame)
            frame = draw(frame, face)
            cv2.imshow("webcam", frame)
            last_time = time.time()
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print "================================="
                print "Quitting.."
                print "================================="
                cv2.waitKey(1)
                break

    clf = learning_SVM(1, emotions)
    print "==========================================="
    print ""
    print "CLF COMPLETE"
    print ""
    happy = cv2.imread("happytest.png")
    sad = cv2.imread("sadtest.png")
    print "--------------------"
    print "HAPPY PREDICTION:  "
    happyface = get_landmark_vectors(happy)
    make_SVM_prediction(clf, happyface, emotions)
    print "-------------------"
    print "SAD PREDICTION:    "
    sadface = get_landmark_vectors(sad)
    make_SVM_prediction(clf, sadface, emotions)

    print "==========================================="

    print "WEBCAM TEST"

    last_time = 0
    while True:
        ret, frame = video_capture.read()
        curr_time = time.time()
        if curr_time - last_time > 3.5:
            cv2.imshow("webcam", frame)
            face = get_landmark_vectors(frame)
            make_SVM_prediction(clf, face, emotions)
            frame = draw(frame, face)
            cv2.imshow("landmarks", frame)
            last_time = time.time()
            if cv2.waitKey(10) & 0xFF == ord('q'):
                    print "================================="
                    print "Quitting.."
                    print "================================="
                    break
run()