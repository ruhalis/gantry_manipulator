# ROS2 + Moveit2 for RoArm-M2-S

## Quick start
### Requirements: 
- ROS2
- Any computer arm64/x64/x86

### Install dependencies:

```
sudo apt install ros-humble-moveit
sudo apt install ros-humble-moveit* ros-humble-ros2-control ros-humble-ros2-controllers
```
```
colcon build
```
 
### Commands
```
sudo chmod 666 /dev/ttyUSB0
```
```
ros2 run roarm_driver roarm_driver
```

```
ros2 run roarm_moveit_cmd movepointcmd --ros-args --params-file install/roarm_moveit_cmd/share/roarm_moveit_cmd/config/calibration_offsets.yaml 
```
```
ros2 launch roarm_moveit_cmd command_control.launch.py
```
```
ros2 run roarm_moveit_cmd getposecmd
```
Calibrate RoArm
```
ros2 run roarm_moveit_cmd calibrate_roarm.py --targets-config src/roarm_main/roarm_moveit_cmd/config/calibration_targets.yaml --loops 4 --output-dir
```


### Gantry + arm coordinated control
python3 gantry_arm_controller.py --gantry-port /dev/ttyUSB1



### 1.3 Subsequent Compilations and Usage
Every time you make changes to a package's code, you need to recompile it before using it.

Regular compilation:

    cd ~/roarm_ws_em0
    . build_common.sh

Contents of build_common.sh (automatically run by the script; no manual execution required):

    cd ~/roarm_ws_em0
    colcon build
    colcon build --packages-select roarm_web_app launch_api ros2web_app ros2web_widgets ros2web ros2web_example_py --symlink-install 
    source install/setup.bash 

Compiling only roarm-web-app:

    cd ~/roarm_ws_em0
    . build_roarm_web_app.sh

Contents of build_roarm_web_app.sh (automatically run by the script; no manual execution required):

    cd ~/roarm_ws_em0
    colcon build --packages-select roarm_web_app launch_api ros2web_app ros2web_widgets ros2web ros2web_example_py --symlink-install

## 2 roarm_ws_em0 Package Overview
roarm_ws_em0 is a workspace containing multiple ROS2 packages, each serving a specific purpose in the operation and control of robotic arms. Below is an overview of each package and its main functionalities:

1. roarm_main
Main Functionalities:

Acts as the central package for managing and coordinating the overall operations of the robotic arm system.

2. roarm_driver
Driver for Real Robot:

Responsible for interfacing with and controlling the physical robotic arm hardware.

3. roarm_description
Robotic Arm Model:

Contains the URDF (Unified Robot Description Format) files and other model descriptions necessary for simulating and visualizing the robotic arm.

4. roarm_moveit
Kinematic Configuration:

Provides configurations for MoveIt, a motion planning framework, including setup files and parameters required for the kinematic control of the robotic arm.

5. roarm_moveit_ikfast_plugins
IKFast Kinematics Solver:

Implements the IKFast kinematics solver, which is used for efficient and fast inverse kinematics calculations.

6. roarm_moveit_cmd
Control Commands:

Includes scripts and nodes for sending control commands to the robotic arm, allowing for movement and task execution.

7. roarm_web_app
Web-based Control:

Provides a web application interface for controlling the robotic arm remotely via a web browser.

8. moveit_servo
Joystick Control:

Enables control of the robotic arm using a joystick or gamepad, allowing for intuitive manual operation.

9. roarm_else (Dependency for roarm_web_app)
Additional Dependencies:

Contains various dependencies and supplementary functionalities required by the roarm_web_app package.

10. launch_api
Launch Management:

Provides API functionalities for managing and launching various nodes and packages within the ROS2 ecosystem.

11. ros2web_app
Web Application for ROS2:

Facilitates the development and deployment of web applications that interact with ROS2 nodes and services.

12. ros2web-ros2/ros2web
ROS2 Web Interface:

Implements the core web interface functionalities that allow for seamless integration and communication between web applications and ROS2.

13. ros2web-ros2/ros2web_interfaces
ROS2 Web Interfaces:

Defines the necessary interfaces and messages used for communication between the web applications and ROS2 nodes.


## 3 Controlling a Physical Robotic Arm with Driver Nodes
### 3.1 Connect the Robotic Arm and Identify the Serial Port
Before connecting the robotic arm via USB, check the current serial devices on your Ubuntu system:

    ls /dev/tty*

Then, connect the robotic arm. Be sure to connect to the Type-C port in the middle of the PCB board (the edge Type-C port is for radar connection and does not communicate with the ESP32). Click on `Devices → USB → select the device with "CP210x"` in its name from the Oracle VM VirtualBox menu. If prompted that the device cannot be mounted, you may need to shut down the virtual machine.

Ensure your computer (Windows) can detect this USB device.

Next, in the VM settings, locate the USB device section, check Enable USB Controller, select `USB 3.0 (xHCI)` Controller, and add a USB filter for the device with "CP210x" in its name. Click OK.

Run the VM again, and from the top menu, click Devices → USB → ensure there is a checkmark next to the device with "CP2102N USB".

Check the serial devices again:

    ls /dev/tty*

You should now see a new device like `/dev/ttyUSB0` at the end of the list. If not, disconnect and reconnect the robotic arm.

### 3.2 Change the Serial Port Device
If the detected serial port device is `/dev/ttyUSB0`, you can skip this section and proceed to **3.3** Running the Robotic Arm Driver Node.

If the serial port device is not `/dev/ttyUSB0`, you need to update the serial port device name in the Python script `~/roarm_ws_em0/src/roarm_main/roarm_driver/roarm_driver/roarm_driver.py` by changing line 15:

    serial_port = "/dev/ttyUSB0"

to your actual serial port device name.
![image](images/roarm_driver.py.png)
Then, recompile the ROS2 packages in the terminal:

    cd ~/roarm_ws_em0/
    colcon build
    source install/setup.bash

### 3.3 Running the Robotic Arm Driver Node
According to the ROS2 official documentation, it is not recommended to run ROS2 nodes in the same terminal where you compile the packages. Open a new terminal window using `Ctrl + Alt + T`.

Grant serial port permissions and run the ROS2 robotic arm driver node:

Grant read and write permissions to the serial device using the following command (replace `/dev/ttyUSB0` with your actual device path):

    sudo chmod 666 /dev/ttyUSB0

Run the driver node:

    ros2 run roarm_driver roarm_driver

### 3.4 Viewing the Model Joints
Open a new terminal window with `Ctrl + Alt + T`.

Run Rviz2 to display the robotic arm model and the joint control panel:

Rviz2 is a visualization tool in ROS2 that can display and debug robot models, sensor data, path planning, and more. Using Rviz2, you can visually monitor the robot's movements and current position. Additionally, Rviz2 provides tools such as 3D views, timelines, and parameter adjustments to better understand the robot's behavior.

Note: When you run the following command, the robot's URDF in Rviz2 will publish the joint angles to the driver node. The driver node will convert these angles into JSON control commands for the robotic arm, causing it to move (the arm will be vertical). Ensure there are no fragile items around the robotic arm and keep children away.

    ros2 launch roarm_description display.launch.py

[![](https://res.cloudinary.com/marcomontalbano/image/upload/v1723287900/video_to_markdown/images/youtube--uUkncafp-y4-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/uUkncafp-y4 "")

If you do not see the joint control panel, click the gear icon on the left to bring it to the forefront.

You can control the movement of each joint by dragging the sliders in the control panel. This is the simplest and most basic method to control the robotic arm in ROS2.

You can control the LED on the gripper of the robot arm through the following command,data is an integer, the range is [0-255].

    ros2 topic pub /led_ctrl std_msgs/msg/Float32 {"data: 0"} -1

### 3.5 Manipulating the View in Rviz2
In Rviz2, you can adjust the view of the robot model using the mouse:

Left-click and drag to move the view horizontally.
Right-click and drag to change the viewing direction and angle.
Scroll the mouse wheel to zoom in or out.
Press and hold the middle mouse button (wheel) and drag to move vertically.
These operations allow you to view the robot model from any angle and distance.


## 4 Interacting with the Robotic Arm End Effector Using MoveIt2
### 4.1 Introduction to MoveIt2
MoveIt2 is an open-source software for robotic motion planning, manipulation, and control. It provides a simple user interface to help developers handle complex motion planning problems.

MoveIt2 supports various algorithms and strategies, including motion planning, execution, monitoring, kinematics, and collision detection. Its powerful features make it widely used in industrial, research, and educational fields.

MoveIt2 operates within the ROS2 (Robot Operating System 2) environment and integrates seamlessly with other ROS2 tools and libraries, significantly improving the efficiency and convenience of robot development.

In this tutorial, we will use MoveIt2 to control the robotic arm's movements. By dragging the end effector of the robotic arm, MoveIt2 can automatically calculate the motion path and control the actual movement of the robotic arm through the driver nodes.

### 4.2 Running the MoveIt2 Demo
In the terminal window where Rviz2 is currently running, press `Ctrl + C` to close the Rviz2 session.

Run the following command to execute the robotic arm MoveIt2 demo. This demo includes inverse kinematics solving, allowing you to interact with the robotic arm by dragging the end effector:

    ros2 launch roarm_moveit interact.launch.py 

When controlling the gripper, it should be selected as in the picture
![image](images/rviz_gripper.png)

Note: After executing this command, the robotic arm will move, with the forearm extending forward and parallel to the ground.

[![](https://res.cloudinary.com/marcomontalbano/image/upload/v1723532378/video_to_markdown/images/youtube--BzLm0tmo1lw-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/BzLm0tmo1lw "")

If the robotic arm is not displayed in Rviz2, click on the `Fixed Frame` option in the Displays window after launching Rviz2. Beside `map`, a triangle icon will appear; click on this icon, select `base_link`, and press Enter to confirm.

Next, click on `Add` in the bottom left corner, then select `MotionPlanning` and click `OK`. This will display the robotic arm model. You can refer to the **3.5 Manipulating the View in Rviz2** section to change the view angle.

Now, you can change the posture of the robotic arm by dragging the sphere or XYZ axes at the end effector. This dragging will not immediately move the physical robotic arm. To execute the planned movements on the real robotic arm, you need to click the `Plan & Execute` button in the Planning tab on the right side of the screen.




## 8 Command Control (Invoking ROS2 Services , Actions)
In this chapter, you will learn how to control the robotic arm using command-based control by invoking ROS2 services. Close all terminal windows from the previous chapters, except for the one running roarm_driver.

Run the launch file for command control, which relies on MoveIt2 for motion planning:

    ros2 launch roarm_moveit_cmd command_control.launch.py

Note: After executing this command, the robotic arm will move, with the forearm extending forward and parallel to the horizontal plane.

If the robotic arm is not displayed in Rviz2, follow these steps:

In Rviz2, click on the Displays window.

In the Fixed Frame field, click map, which will show a triangle icon.
Click the triangle icon and select base_link, then press Enter to confirm.
Click Add in the bottom left corner, select MotionPlanning, and click OK.
You should now see the robotic arm's model. Refer to 3.5 Rviz2 View Operations to adjust the view.

You can change the robotic arm's posture by dragging the end-effector's drag ball or XYZ axis in Rviz2. However, these changes will not synchronize with the physical robotic arm until you click Plan & Execute in the Planning tab on the right.

### 8.1 Get Current Position
Open a new terminal to start the node that retrieves the current position:

    ros2 run roarm_moveit_cmd getposecmd

Open another terminal to call the service that gets the current position:

    ros2 service call /get_pose_cmd roarm_moveit/srv/GetPoseCmd

[![](https://res.cloudinary.com/marcomontalbano/image/upload/v1715919239/video_to_markdown/images/youtube--5g5KuA3q5oQ-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/5g5KuA3q5oQ "")

### 8.2 Move the End-Effector to a Specified Position
Start the node for motion control (which receives the target position):

    ros2 run roarm_moveit_cmd movepointcmd

Call the service to control the end-effector's position:

    ros2 service call /move_point_cmd roarm_moveit/srv/MovePointCmd "{x: 0.2, y: 0, z: 0}"

[![](https://res.cloudinary.com/marcomontalbano/image/upload/v1715919259/video_to_markdown/images/youtube--gmcXta85clc-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/gmcXta85clc "")

Here, x, y, and z are the coordinates of the target point in meters.

By calling this service, you can control the robotic arm to move to the target position.

### 8.3 Control the gripper to the specified radian position
Start the gripper motion control node (receive the target radian position):

    ros2 run roarm_moveit_cmd setgrippercmd

Publish data to a topic and use actions to control the radian position of the gripper:

    ros2 topic pub /gripper_cmd std_msgs/msg/Float32  "{data: 0.0}" -1

[![Control the Gripper to a Specified Position](https://res.cloudinary.com/marcomontalbano/image/upload/v1723287232/video_to_markdown/images/youtube--uRUIMIWvNpw-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/uRUIMIWvNpw "Control the Gripper to a Specified Position")

Here, the data is the coordinates of the target point of the gripper, in radians.

By invoking this service, you can control the position of the robotic arm gripper to move the target radian.

### 8.4 Draw a Circle at a Fixed Height
In Rviz2, click Add, add RobotModel, and in the RobotModel tab, find Description Topic to view the trajectory of the end-effector hand_tcp.

![image](images/rviz_hand_tcp.png)

Start the node to draw a circle:


[![](https://res.cloudinary.com/marcomontalbano/image/upload/v1723287696/video_to_markdown/images/youtube--ZcsHhUDnOKw-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/ZcsHhUDnOKw "")

The x, y, and z parameters specify the center of the circle, and radius specifies the radius of the circle in meters.

By calling this service, you can control the robotic arm to draw a circle at the desired position.