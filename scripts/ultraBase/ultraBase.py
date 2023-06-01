#!/usr/bin/env python3
"""
ultraBase.py é o script definitivo de ROS.

Ele implementa:
    --> Subscribers básicos
        --> Image subscriber
        --> Laser subscriber
        --> Odometry subscriber

    --> Go to waypoint
    --> Aruco markers
    --> MobileNet detection
    --> Garra
    
    --> Follow line simples (erro angular e KP)

Ele NÃO implementa:
	--> Girar x graus (is_within_angular_threshold())
	--> Máscaras dos creepers
    --> Regressão linear
    --> Publicação de tópicos personalizados em arquivos diferentes
"""

import os
import cv2
import cv2.aruco as aruco  # type: ignore
import rospy
import numpy as np

from mobileNet import net, CLASSES, CONFIDANCE, COLORS, detect

from sensor_msgs.msg import LaserScan, CompressedImage
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError

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

	@classmethod
	def rectangle(cls, img, start_point, end_point):		
		color = (255, 0, 0)
		thickness = 2
		return cv2.rectangle(img, start_point, end_point, color, thickness)


class Control():
	def __init__(self):
		# Constant class attributes
		self.rate = rospy.Rate(250) # 250 Hz
		self.bridge = CvBridge()
		self.twist = Twist()

		current_script_directory = os.path.dirname(os.path.abspath(__file__))
		self.camera_matrix       = np.loadtxt(current_script_directory + '/aruco/cameraMatrix_realsense.txt',     delimiter=',')  # used in Aruco marker detection
		self.camera_distortion   = np.loadtxt(current_script_directory + '/aruco/cameraDistortion_realsense.txt', delimiter=',')  # used in Aruco marker detection

		# Mutable class attributes
		self.kp = 500               # Constant of proportionality 
		self.image_shape = (0,   0) # Size in pixels: declared in (h, w)
		self.destination = (-1, -1) # Camera coords for the destination, which will be followed by the robot (x, y)
		self.odometry, self.roll, self.pitch, self.yaw = None, None, None, None
		"""
			To access odometry data variables:
				- self.odometry.pose.pose.position.x
				- self.odometry.pose.pose.position.y
				- self.odometry.pose.pose.position.z

			Preferrably, you may also use the getter method:
				x, y, z = self.get_odometry_position()
		"""
		self.laser_data = None
		self.aruco_data = (None, None, None)   # Aruco data is stored as: (IDs; center coordinates; distance from the robot)
		self.waypoint   = Point(x=0, y=0, z=0) # Map coordinates, used in go_to_waypoint()
		self.mobilenet_results = None

		# State management
		self.robot_state = 'follow_line'
		self.state_machine = {
			'initial': self.initial_state,
			'follow_line': self.follow_line_state,
			'center_waypoint': self.center_waypoint_state,
			'go_to_waypoint': self.go_to_waypoint_state,
		}

		# Subscribers
		self.image_subscriber    = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
		self.odometry_subscriber = rospy.Subscriber('/odom', Odometry, self.odometry_callback)
		self.laser_subscriber    = rospy.Subscriber('/scan',LaserScan, self.laser_callback)

		# Publishers
		self.speed_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=3)
		# Para habilitar controle da garra: roslaunch mybot_description mybot_control2.launch
		self.arm   = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
		self.clamp = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)
		"""
			self.arm.publish(-1.0)       # para baixo
			self.arm.publish(1.5)        # para cima
			self.arm.publish(0.0)        # para frente

			self.clamp.publish(0.0)      # fechado
			self.clamp.publish(-1.0)     # aberto
		"""

	# Getters
	def get_Aruco_ids(self):
		ids, _, _ = self.aruco_data
		return ids

	def get_odometry_position(self):
		if self.odometry:
			return self.odometry.pose.pose.position.x, self.odometry.pose.pose.position.y, self.odometry.pose.pose.position.z
		
	def get_front_laser_scan(self):
		if self.laser_data is not None:
			return list(self.laser_data[:5]) + list(self.laser_data[-5:])
		
	def get_back_laser_scan(self):
		if self.laser_data is not None:
			return list(self.laser_data[175:185])

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
		cornersList, ids, _ = aruco.detectMarkers(grayColor, dicionarioAruco)

		if ids is not None:
			for i in range(len(ids)):
				if ids[i]>99:
					ret = aruco.estimatePoseSingleMarkers(cornersList[i], 19, self.camera_matrix, self.camera_distortion)
					rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
					distancia_aruco.append(np.linalg.norm(tvec))
				else: 
					ret = aruco.estimatePoseSingleMarkers(cornersList[i], 6, self.camera_matrix, self.camera_distortion)
					rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
					distancia_aruco.append(np.linalg.norm(tvec))

			for corners in cornersList:
				for corner in corners:
					centros.append(np.mean(corner, axis=0))

		return ids, centros, distancia_aruco

	# Subscriber callbacks
	def image_callback(self, msg: CompressedImage):
		try:
			cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)

		height, width, _ = cv_image.shape
		self.image_shape = (height, width)

		self.aruco_data = self.generate_Aruco(cv_image)
		self.mobilenet_results = detect(net, cv_image, CONFIDANCE, COLORS, CLASSES)[1]
		self.color_segmentation(cv_image)

	def odometry_callback(self, data: Odometry):
		self.odometry = data
		"""
			To access odometry data variables:
				- self.odometry.pose.pose.position.x
				- self.odometry.pose.pose.position.y
				- self.odometry.pose.pose.position.z

			Preferrably, you may also use the getter method:
				x, y, z = self.get_odometry_position()
		"""
		
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

	# State helpers
	def get_angular_error(self):
		kp = 0.05

		x, y, _ = self.get_odometry_position()
		x = self.waypoint.x - x
		y = self.waypoint.y - y
		theta = np.arctan2(y , x)

		distance = np.sqrt(x**2 + y**2)
		err = np.rad2deg(theta - self.yaw)
		err = (err + 180) % 360 - 180

		self.twist.angular.z = err * kp

		return err, distance


	# State functions
	def initial_state(self):
		self.twist.angular.z = 0
		self.twist.linear.x  = 0
		pass

	def follow_line_state(self):
		height, width = self.image_shape
		x, y          = self.destination
		
		if x != -1:
			error = width / 2 - x
			self.twist.angular.z = error / self.kp
			self.twist.linear.x  = 0.1

		else:
			self.twist.angular.z = 0
			self.twist.linear.x = 0

	def center_waypoint_state(self):
		# Part of the goTo() strategy
		err, _ = self.get_angular_error()
		if abs(err) < 5:
			print("Waypoint centered.")
			self.robot_state = 'go_to_waypoint'

	def go_to_waypoint_state(self):
		# Part of the goTo() strategy
		_, distance = self.get_angular_error()

		if distance > 0.1:
			self.twist.linear.x = np.min([distance, 0.1])
		else:
			print('Waypoint reached')
			self.robot_state = 'initial'

	def control(self) -> None:
		'''
		This function is called at least at {self.rate} Hz.
		'''
		print(f"Current robot state: {self.robot_state}")
		# print(f"Odometry position (x, y, z): {self.get_odometry_position()}")
		# print(f"Laser reading (front): {self.get_front_laser_scan()}")
		# print(f"Laser reading (back): {self.get_back_laser_scan()}")
		# print(f"Aruco IDs: {self.get_Aruco_ids()}")
		# print(f"MobileNet detections: {self.mobilenet_results}")

		self.arm.publish(0.0)
		self.clamp.publish(-1.0)

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
