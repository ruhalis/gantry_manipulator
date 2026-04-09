sudo chmod 666 /dev/ttyUSB0

ros2 run roarm_driver roarm_driver

ros2 run roarm_moveit_cmd movepointcmd --ros-args --params-file install/roarm_moveit_cmd/share/roarm_moveit_cmd/config/calibration_offsets.yaml 

ros2 launch roarm_moveit_cmd command_control.launch.py

ros2 run roarm_moveit_cmd getposecmd


  ros2 run roarm_moveit_cmd calibrate_roarm.py \
    --targets-config
  src/roarm_main/roarm_moveit_cmd/config/calibration_targets.yaml \
    --loops 4 \
    --output-dir .
# Gantry + arm coordinated control (requires ROS2 stack running)
python3 gantry_arm_controller.py --gantry-port /dev/ttyUSB1


  

  sudo chmod 666 /dev/pts/3

  sudo socat PTY,link=/dev/ttyArduino,rawer TCP:192.168.64.1:54321 &

  ls -la /dev/ttyArduino