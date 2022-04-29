import cv2
import rospy

import numpy as np

from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image

from detector import StopSignDetector


class SignDetector:
    def __init__(self):
        rospy.logerr("starting init")
        self.detector = StopSignDetector()
        rospy.logerr("model init'd")
        self.publisher = rospy.Publisher("ml", Float32MultiArray, queue_size=1)
        self.bbox_pub = rospy.Publisher('stop_sign_bbox', Float32MultiArray, queue_size=1)

        self.debug_pub = rospy.Publisher('stop_sign_debug', Image, queue_size=1)
        self.subscriber = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.callback)

        rospy.logerr("finished init")

    def callback(self, img_msg):
        np_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        bgr_img = np_img[:,:,:-1]

        # Reshape for velodyne
        # rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)[::-1, ::-1]
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


        # Unscaled
        if True:
            _, bb = self.detector.predict(rgb_img)
            print(bb)

            ml_msg = Float32MultiArray()
            ml_msg.data = bb
            self.publisher.publish(ml_msg)

        # Scaled
        # if False:
          #   rgb_img = img
          #   scale = 640./max(img.shape[0], img.shape[1])
          #   img_scaled = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

        return


if __name__=="__main__":
    rospy.init_node("stop_sign_detector")
    detect = SignDetector()
    rospy.spin()
