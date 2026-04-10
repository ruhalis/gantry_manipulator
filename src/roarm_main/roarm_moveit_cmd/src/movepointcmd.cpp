#include <memory>
#include <math.h>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "roarm_moveit/srv/move_point_cmd.hpp"
#include "roarm_moveit_cmd/ik.h"

class MovePointCmdNode : public rclcpp::Node
{
public:
  MovePointCmdNode()
  : Node("move_point_cmd_node")
  {
    // Declare calibration parameters once at startup
    this->declare_parameter("calibration.enabled", false);
    this->declare_parameter("calibration.offsets.x", 0.0);
    this->declare_parameter("calibration.offsets.y", 0.0);
    this->declare_parameter("calibration.offsets.z", 0.0);

    // MoveGroupInterface must be created after the node exists and is known to the executor.
    // We defer creation to init() which is called after the node is added to the executor.
  }

  // Call this after the node has been added to an executor (so it can receive messages).
  void init()
  {
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "hand");

    server_ = this->create_service<roarm_moveit::srv::MovePointCmd>(
      "move_point_cmd",
      std::bind(&MovePointCmdNode::handle_service, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "MovePointCmd service is ready.");
  }

private:
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  rclcpp::Service<roarm_moveit::srv::MovePointCmd>::SharedPtr server_;

  void handle_service(
    const std::shared_ptr<roarm_moveit::srv::MovePointCmd::Request> request,
    std::shared_ptr<roarm_moveit::srv::MovePointCmd::Response> response)
  {
    // Read calibration parameters from the parameter server
    bool calib_enabled = this->get_parameter("calibration.enabled").as_bool();
    double calib_x = this->get_parameter("calibration.offsets.x").as_double();
    double calib_y = this->get_parameter("calibration.offsets.y").as_double();
    double calib_z = this->get_parameter("calibration.offsets.z").as_double();

    double corrected_x = request->x;
    double corrected_y = request->y;
    double corrected_z = request->z;

    if (calib_enabled) {
      corrected_x = request->x - calib_x;
      corrected_y = request->y - calib_y;
      corrected_z = request->z - calib_z;

      RCLCPP_INFO(this->get_logger(),
        "Calibration enabled: offsets [%.5f, %.5f, %.5f]", calib_x, calib_y, calib_z);
      RCLCPP_INFO(this->get_logger(),
        "Original command: [%.4f, %.4f, %.4f]", request->x, request->y, request->z);
      RCLCPP_INFO(this->get_logger(),
        "Corrected target: [%.4f, %.4f, %.4f]", corrected_x, corrected_y, corrected_z);
    }

    double target_x =  corrected_x;
    double target_y = -1.0 * corrected_y;
    double target_z =  corrected_z;

    cartesian_to_polar(1000.0 * target_x, 1000.0 * target_y, &base_r, &BASE_point_RAD);
    simpleLinkageIkRad(l2, l3, base_r, 1000.0 * target_z);

    RCLCPP_INFO(this->get_logger(),
      "BASE_point_RAD: %f, SHOULDER_point_RAD: %f, ELBOW_point_RAD: %f",
      BASE_point_RAD, -SHOULDER_point_RAD, ELBOW_point_RAD);
    RCLCPP_INFO(this->get_logger(),
      "x: %f, y: %f, z: %f", request->x, request->y, request->z);

    std::vector<double> target = {BASE_point_RAD, -SHOULDER_point_RAD, ELBOW_point_RAD};

    move_group_->setGoalJointTolerance(0.002);
    move_group_->setNumPlanningAttempts(10);
    move_group_->setPlanningTime(5.0);

    // Use the actual current state, not a stale default
    move_group_->setStartStateToCurrentState();
    move_group_->setJointValueTarget(target);

    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool success =
      (move_group_->plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
      move_group_->execute(my_plan);
      response->success = true;
      response->message = "MovePointCmd executed successfully";
      RCLCPP_INFO(this->get_logger(), "MovePointCmd service executed successfully");
    } else {
      response->success = false;
      response->message = "Planning failed!";
      RCLCPP_ERROR(this->get_logger(), "Planning failed!");
    }
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  // Use MultiThreadedExecutor so MoveGroupInterface callbacks (joint states, etc.)
  // are processed while the service handler is executing.
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<MovePointCmdNode>();
  executor.add_node(node);

  // init() after the node is in the executor so MoveGroupInterface can receive messages
  node->init();

  executor.spin();
  rclcpp::shutdown();
  return 0;
}
