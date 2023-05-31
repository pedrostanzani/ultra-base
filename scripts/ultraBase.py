#!/usr/bin/env python3

import cv2
import cv2.aruco as aruco
import rospy
import numpy as np

from sensor_msgs.msg import LaserScan, CompressedImage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError

from mobilenet import net, CLASSES, CONFIDANCE, COLORS, detect

from std_msgs.msg import Float64

""" 
Rode cada linha em um terminal diferente
	rosrun modulo4 [nome_do_script].py
"""

class Utils:
	@classmethod
	def center_of_contour(cls, contour):
		M = cv2.moments(contour)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		return int(cX), int(cY)
	
	@classmethod
	def crosshair(cls, img, point, size=15, color=(255, 255, 0)):
		x,y = point
		cv2.line(img,(x - size,y),(x + size,y),color,5)
		cv2.line(img,(x,y - size),(x, y + size),color,5)
		

class Control():
	def __init__(self):
		# Constant class attributes
		self.rate = rospy.Rate(250) # 250 Hz
		self.bridge = CvBridge()
		self.twist = Twist()
		self.odometry, self.roll, self.pitch, self.yaw = None, None, None, None
		self.laser_data = None
		"""
			To access odometry data variables:
				- self.odometry.pose.pose.position.x
				- self.odometry.pose.pose.position.y
				- self.odometry.pose.pose.position.z
		"""

		# Mutable class attributes
		self.kp = 500 
		self.image_shape = (0,   0) # Size in pixels: declared in (h, w)
		self.destination = (-1, -1) # Camera coords for the destination, which will be followed by the robot (x, y)

		# State management
		self.robot_state = 'goto'
		self.state_machine = {
			'initial': self.initial_state,
			'turn': self.turn_state,
			'goto': self.goto_state,
			'stop': self.stop_state
		}

		# Subscribers
		self.image_subscriber    = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
		self.odometry_subscriber = rospy.Subscriber('/odom', Odometry, self.odometry_callback)
		self.laser_subscriber    = rospy.Subscriber('/scan',LaserScan, self.laser_callback)

		# Publishers
		self.speed_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=3)
		# Para habilitar controle da garra: roslaunch mybot_description mybot_control2.launch
		self.braco = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
		self.pinca = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)
		'''
		self.braco.publish(-1.0)       # para baixo
		self.braco.publish(1.5)        # para cima
		self.braco.publish(0.0)        # para frente

		self.pinca.publish(0.0)        # fechado
		self.pinca.publish(-1.0)       # fechado
		'''

	# Getters
	def get_odometry_position(self, round_to=3):
		if self.odometry:
			return round(self.odometry.pose.pose.position.x, round_to), round(self.odometry.pose.pose.position.y, round_to), round(self.odometry.pose.pose.position.z, round_to)
		
	def get_front_laser_scan(self):
		if self.laser_data is not None:
			return list(self.laser_data[:5]) + list(self.laser_data[-5:])

	# Callback helpers
	def color_segmentation(self, cv_image):
		hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

		# Creating a mask
		lower_bound = np.array([25, 190, 190], dtype=np.uint8)
		upper_bound = np.array([40, 255, 255], dtype=np.uint8)
		mask = cv2.inRange(hsv, lower_bound, upper_bound)

		# Noise removal
		kernel = np.ones((5,5),np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

		# Ientification of mask contours and sorting by contour area
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contours = sorted(contours, key=cv2.contourArea)[::-1]

		if len(contours) > 0:
			self.destination = Utils.center_of_contour(contours[0])
			Utils.crosshair(cv_image, self.destination)
		else:
			self.destination = -1, -1

		cv2.imshow('window', cv_image)
		cv2.waitKey(1)

	def generate_Aruco(self, cv_image):
		centros = []
		distancia_aruco = []
		grayColor = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
		dicionarioAruco = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)

	# Subscriber callbacks
	def image_callback(self, msg: CompressedImage):
		try:
			cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)

		height, width, _ = cv_image.shape
		self.image_shape = (height, width)

		self.color_segmentation(cv_image)
		self.generate_Aruco(cv_image)

		self.resultsMobilenet = detect(net, cv_image, CONFIDANCE, COLORS, CLASSES)[1]

	def odometry_callback(self, data: Odometry):
		self.odometry = data
		# self.x = data.pose.pose.position.x
		# self.y = data.pose.pose.position.y
		# self.z = data.pose.pose.position.z
		
		orientation_list = [
			data.pose.pose.orientation.x,
			data.pose.pose.orientation.y,
			data.pose.pose.orientation.z,
			data.pose.pose.orientation.w
		]

		# Convert yaw from [-pi, pi] to [0, 2pi]
		self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_list)
		self.yaw = self.yaw % (2 * np.pi)

	def laser_callback(self, msg: LaserScan):
		self.laser_data = np.array(msg.ranges).round(decimals=2) # Converte para np.array e arredonda para 2 casas decimais
		self.laser_data[self.laser_data == 0] = np.inf
	
	def get_angular_error(self):
		kp = 0.05

		#Turtlebot goto (cX, cY)
		cX = 1
		cY = -2

		x = cX - self.get_odometry_position()[0]
		y = cY - self.get_odometry_position()[1]
		theta = np.arctan2(y,x)

		self.distance = np.sqrt(x**2+y**2)
		self.err = np.rad2deg(theta - self.yaw)
		self.err = (self.err+180) % 360 - 180

		self.twist.angular.z = self.err * kp

		print(self.err)

	# State functions
	def initial_state(self):
		if self.destination[0] != -1:
			err = self.image_shape[1]/2 - self.destination[0]
			self.twist.angular.z = err/self.kp
			self.twist.linear.x = 0.2
		if self.get_odometry_position()[1] >= -1.961:
			self.robot_state = "turn"
		

	def turn_state(self):
		self.twist.linear.x = 0
		self.twist.angular.z = 0.1
	
	def goto_state(self):
		self.get_angular_error()
		self.twist.linear.x = 0.2
		
		if self.distance < 0.1:
			self.robot_state = "stop"

	def stop_state(self):
		self.twist.linear.x = 0
		self.twist.angular.z = 0
		
		
	
	def control(self) -> None:
		'''
		This function is called at least at {self.rate} Hz.
		'''
		#print(self.resultsMobilenet)
		print(f"Current robot state: {self.robot_state}")
		#print(f"Odometry position (x, y, z): {self.get_odometry_position()}")
		#print(f"Laser reading: {self.get_front_laser_scan()}")
		self.state_machine[self.robot_state]()
		
		self.speed_publisher.publish(self.twist)
		self.rate.sleep() # Sleeps the remaining time to keep the rate

def main():
	rospy.init_node('Controler')
	control = Control()
	rospy.sleep(1) # Espera 1 segundo para que os publishers e subscribers sejam criados

	while not rospy.is_shutdown():
		control.control()

if __name__== "__main__":
	main()
