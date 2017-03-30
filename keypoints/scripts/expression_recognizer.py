#!/usr/bin/env python

""" This is a script takes an image from the webcam and controls the neato. """

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import copy
from geometry_msgs.msg import Twist, Vector3
import time
import os
import dlib


class ExpressionNode(object):
	def __init__(self):
		""" Initialize expression reader. """
		rospy.init_node('expression_recognizer')
		self.cv_image = None
		self.bridge = CvBridge()
		rospy.Subscriber("/camera/image_raw", Image, self.process_image)
		self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		self.twist = Twist()
		print "init!"


	

	def process_image(self, msg):
		""" Process image messages from ROS. """ 
		self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
		print "processing"



	def detect_face(self):
		pass

	def read_expression(self):
		pass

	def set_twist(self):
		"""  """
		pass

	def run(self):
		r = rospy.Rate(1)
		while not rospy.is_shutdown():
			print "hi"
			r.sleep()


if __name__ == '__main__':
    image_path = "catkin_ws/src/computer_vision_emotion/images/face.jpg"
    node = ExpressionNode()
    node.run()