# System integration project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://github.com/udacity/CarND-Capstone).

For this project, I have written ROS nodes to implement core functionality of the autonomous vehicle system, including traffic light detection, control, and waypoint following.
Finally, the code needs to be tested in a simulator.

The project is submitted as an **Individual Submission**.

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/AlessandroPerinetti/capstone-project.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

## Implementation

### System Architecture
The following is a system architecture diagram showing the ROS nodes and topics used in the project.

![](media/final-project-ros-graph.png)

The ROS nodes and topics shown in the diagram are described briefly in the next sections.

### Waypoint updater

The purpose of this node is to update the target velocity property of each waypoint based on traffic light data. This node will subscribe to the ``/base_waypoints``, ``/current_pose`` and ``/traffic_waypoint`` topics, and publish a list of waypoints ahead of the car with target velocities to the ``/final_waypoints`` topic.

The number of waypoints and the publishing rate are choosen in order to obtain good accuracy yet keeping the simulation able to run in the provided workspace and on my local machine.


### Drive-by-wire node

The drive-by-wire (dbw) node (in particular the packages ``dbw_node.py``, ``twist_controller.py``, along with a pid and lowpass filter) is responsible for the throttle, brake, and steering control. 

The dbw_node subscribes to the ``/current_velocity`` topic and the ``/twist_cmd`` topic to receive target linear and angular velocities. Additionally, this node will subscribe to ``/vehicle/dbw_enabled``, which indicates if the car is under dbw or driver control. This node will publish throttle, brake, and steering commands to the ``/vehicle/throttle_cmd``, ``/vehicle/brake_cmd``, and ``/vehicle/steering_cmd`` topics.

The throttle value is managed by a PI controller tuned such that the car is able to reach the target velocity.

On the other hand, the steering angle is obtained from the package ``yaw__controller.py`` and is defined in a way that the car follows the waypoints.

### Traffic Light Detection

This package contains the traffic light detection node ``tl_detector.py``. This node takes in data from the ``/image_color``, ``/current_pose``, and ``/base_waypoints`` topics and publishes the locations to stop for red traffic lights to the ``/traffic_waypoint`` topic.

The ``/current_pose`` topic provides the vehicle's current position, and ``/base_waypoints`` provides a complete list of waypoints the car will be following.

This node is responsible for detecting upcoming traffic lights and classify their states (red, yellow, green or unknown).

I used the pre-trained [SSD](https://github.com/tensorflow/models/tree/master/research/object_detection) model to detect and classify the traffic lights in the images provided by the camera. 

The model was adapted to this contest by training it on the images taken from the simulator.

## Results

During the project developement, I had some trouble concerning the computational power needed by the simulation.
I was able to solve the problem and run the simulation smoothly by tuning some critical parameters.

The car is able to complete the full track following the given waypoints and making stops in case of red traffic lights.

The resuls are visibile in the [video](media/capstone-video.mp4) and in these short previews.


<p float="left">
  <img src="media/waypoint_follower.gif" width="350" />
  <img src="media/traffic-light-stop.gif" width="350" />
</p>


