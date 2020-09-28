from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import os
import cv2
import rospy
import yaml

IMAGE_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../../../test_images/simulator/'
MODEL_PATH = '/frozen_inference_graph.pb'
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
SAVE_IMAGES = False

class TLClassifier(object):
    def __init__(self):
        self.model_graph = None
        self.session = None
        self.image_counter = 0
        self.classes = {1: TrafficLight.RED,
                        2: TrafficLight.YELLOW,
                        3: TrafficLight.GREEN,
                        4: TrafficLight.UNKNOWN}

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        model_path = os.path.dirname(os.path.realpath(__file__)) + MODEL_PATH
        
        # load frozen tensorflow model
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.model_graph = tf.Graph()
        with tf.Session(graph=self.model_graph, config=config) as sess:
            self.session = sess
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
        self.image_tensor = self.model_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.model_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.model_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.model_graph.get_tensor_by_name('detection_classes:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        image_np = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        (boxes, scores, classes) = self.session.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes],
            feed_dict={self.image_tensor: np.expand_dims(image_np, axis=0)})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        boxes = np.squeeze(boxes)
        
        min_score_thresh = 0.5

        for i, box in enumerate(boxes):
            if scores[i] > min_score_thresh:
                light_class = self.classes[classes[i]]
                self.save_image(image_np, light_class)
                return light_class
            else:
                self.save_image(image_np, TrafficLight.UNKNOWN)   

        return TrafficLight.UNKNOWN
    
    def save_image(self, image, light_class):
        if SAVE_IMAGES:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(IMAGE_PATH, "img_%d_%04i.jpg" % ( light_class, self.image_counter)), bgr_image)
            self.image_counter += 1