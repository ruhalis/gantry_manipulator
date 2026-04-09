#!/usr/bin/env python3
"""
Gantry + RoArm-M2-S Coordinated Controller

Controls a 2-axis XY gantry (Arduino stepper motors) and a Waveshare
RoArm-M2-S manipulator to reach any target position in the combined workspace.

Strategy:
  1. Given target (x, y) in mm, compute how much the gantry must move so the
     arm's base is within ARM_REACH_RADIUS of the target.
  2. Move gantry to that position (relative mm commands over serial).
  3. Compute the local offset from the arm's base to the target.
  4. Command the RoArm via ROS2 /move_point_cmd service (MoveIt2).

The gantry Arduino expects: "dx_mm, dy_mm\n" (relative moves).
The arm is controlled via ROS2 /move_point_cmd service (requires ROS2 stack running).
"""

import math
import time
import serial
import argparse
import sys

import rclpy
from rclpy.node import Node
from roarm_moveit.srv import MovePointCmd

# ============================================================
#  Arm geometry (from ik.h)
# ============================================================
L2A = 236.82
L2B = 30.0
L3A = 280.15
L3B = 1.73
L2 = math.sqrt(L2A**2 + L2B**2)
L3 = math.sqrt(L3A**2 + L3B**2)
ARM_MAX_REACH_MM = L2 + L3  # ~519mm theoretical max

ARM_REACH_RADIUS_MM = 200.0  # default comfortable reach radius in mm
ARM_DEFAULT_Z_MM = -15.0     # default end-effector height in mm


def gantry_to_arm_coords(local_x: float, local_y: float) -> tuple:
    """
    Convert gantry-frame local coordinates to RoArm coordinates.
    The RoArm's coordinate frame is rotated 90 degrees clockwise
    relative to the gantry XY frame.

    Gantry (local_x, local_y) -> Arm (local_y, -local_x)
    """
    arm_x = local_y
    arm_y = -local_x
    return arm_x, arm_y


class GantryController:
    """
    Talks to the Arduino running motor_control.c++ over serial.
    Tracks absolute position in software (starts at 0,0 after homing).
    Sends relative move commands in mm.
    """

    def __init__(self, port: str, baud: int = 115200, timeout: float = 2.0):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)  # wait for Arduino reset
        self._drain_startup()
        self.pos_x = 0.0  # mm, current tracked position
        self.pos_y = 0.0

    def _drain_startup(self):
        """Read and discard the Arduino startup banner."""
        while self.ser.in_waiting:
            self.ser.readline()

    def move_relative(self, dx_mm: float, dy_mm: float, wait: bool = True):
        """
        Send a relative move command to the Arduino.
        dx_mm: motor 1 distance in mm (X axis)
        dy_mm: motor 2 distance in mm (Y axis)
        """
        cmd = f"{dx_mm:.2f}, {dy_mm:.2f}\n"
        self.ser.write(cmd.encode('utf-8'))
        self.pos_x += dx_mm
        self.pos_y += dy_mm

        if wait:
            self._wait_for_move(dx_mm, dy_mm)

        return self.pos_x, self.pos_y

    def move_absolute(self, target_x_mm: float, target_y_mm: float, wait: bool = True):
        """Move to an absolute position by computing the relative delta."""
        dx = target_x_mm - self.pos_x
        dy = target_y_mm - self.pos_y
        return self.move_relative(dx, dy, wait=wait)

    def _wait_for_move(self, dx_mm: float, dy_mm: float):
        """
        Estimate move time from distance and stepper parameters.
        motor_control.c++ uses maxSpeed=4000 steps/s, accel=2000 steps/s^2,
        3200 steps/rev, 125 mm/rev => 25.6 steps/mm
        Max speed: 4000/25.6 = 156.25 mm/s
        """
        steps_per_mm = 3200.0 / 125.0
        max_speed_mm_s = 4000.0 / steps_per_mm
        accel_mm_s2 = 2000.0 / steps_per_mm

        dist = max(abs(dx_mm), abs(dy_mm))
        if dist < 0.01:
            return

        # trapezoidal profile estimate
        t_accel = max_speed_mm_s / accel_mm_s2
        d_accel = 0.5 * accel_mm_s2 * t_accel**2

        if dist < 2 * d_accel:
            # triangular profile
            t_total = 2.0 * math.sqrt(dist / accel_mm_s2)
        else:
            d_cruise = dist - 2 * d_accel
            t_cruise = d_cruise / max_speed_mm_s
            t_total = 2 * t_accel + t_cruise

        time.sleep(t_total + 0.3)  # small buffer

    def home(self):
        """Reset tracked position to 0,0 (assumes physical homing was done)."""
        self.pos_x = 0.0
        self.pos_y = 0.0

    def get_position(self):
        return self.pos_x, self.pos_y

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


class RoArmController:
    """
    Controls Waveshare RoArm-M2-S via ROS2 /move_point_cmd service (MoveIt2).
    Requires the ROS2 stack to be running (command_control.launch.py or similar).
    """

    def __init__(self, node: Node):
        self._node = node
        self._move_client = node.create_client(MovePointCmd, '/move_point_cmd')

    def wait_for_service(self, timeout_sec: float = 10.0) -> bool:
        """Wait for /move_point_cmd service to become available."""
        print("  [ROS2] Waiting for /move_point_cmd service...")
        if not self._move_client.wait_for_service(timeout_sec=timeout_sec):
            print("  [ROS2] Service /move_point_cmd not available!")
            return False
        print("  [ROS2] Service ready.")
        return True

    def init_position(self):
        """Move arm to default home position via ROS2."""
        print("  [RoArm] Moving to init position...")
        self._move_ros2(200.0, 0.0, 100.0)

    def move_to_xyz(self, x_mm: float, y_mm: float, z_mm: float) -> bool:
        """
        Move end-effector to Cartesian coordinates (all in mm).
        x: forward (+) / backward (-)
        y: left (+) / right (-)
        z: up (+) / down (-)
        """
        return self._move_ros2(x_mm, y_mm, z_mm)

    def _move_ros2(self, x_mm: float, y_mm: float, z_mm: float) -> bool:
        """Move via ROS2 /move_point_cmd service (MoveIt2 planning)."""
        x_m = x_mm / 1000.0
        y_m = y_mm / 1000.0
        z_m = z_mm / 1000.0

        request = MovePointCmd.Request()
        request.x = x_m
        request.y = y_m
        request.z = z_m

        print(f"  [ROS2] /move_point_cmd x={x_m:.4f} y={y_m:.4f} z={z_m:.4f}")
        future = self._move_client.call_async(request)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=30.0)

        if future.result() is not None:
            response = future.result()
            print(f"  [ROS2] success={response.success}")
            return response.success
        else:
            print("  [ROS2] Service call timed out or failed")
            return False

    def close(self):
        pass


def compute_gantry_target(target_x_mm: float, target_y_mm: float,
                          reach_radius_mm: float) -> tuple:
    """
    Given a target point and the arm's reach radius, compute where the
    gantry should position the arm's base.

    Strategy: move the gantry so the arm base is exactly (target - offset)
    where offset is along the direction from origin to target, with magnitude
    such that the remaining distance equals the reach radius.

    If the target is already within reach_radius from the current gantry
    position concept (i.e., the total distance is <= reach_radius), the
    gantry stays at 0,0 and the arm does all the work.
    """
    dist = math.sqrt(target_x_mm**2 + target_y_mm**2)

    if dist <= reach_radius_mm:
        # target is within arm reach from origin, no gantry move needed
        gantry_x = 0.0
        gantry_y = 0.0
    else:
        # move gantry so the arm only needs to reach 'reach_radius_mm'
        # gantry moves along the line from origin to target
        overshoot = dist - reach_radius_mm
        angle = math.atan2(target_y_mm, target_x_mm)
        gantry_x = overshoot * math.cos(angle)
        gantry_y = overshoot * math.sin(angle)

    arm_local_x = target_x_mm - gantry_x
    arm_local_y = target_y_mm - gantry_y

    return gantry_x, gantry_y, arm_local_x, arm_local_y


def goto_position(gantry: GantryController, arm: RoArmController,
                  target_x_mm: float, target_y_mm: float,
                  arm_z_mm: float = ARM_DEFAULT_Z_MM,
                  reach_radius_mm: float = ARM_REACH_RADIUS_MM):
    """
    Coordinated move: gantry positions the arm base, then arm reaches the target.
    All units in mm.
    """
    print(f"\n{'='*60}")
    print(f"TARGET: ({target_x_mm:.1f}, {target_y_mm:.1f}) mm")
    print(f"{'='*60}")

    gantry_x, gantry_y, arm_local_x, arm_local_y = compute_gantry_target(
        target_x_mm, target_y_mm, reach_radius_mm
    )

    print(f"  Gantry move to: ({gantry_x:.1f}, {gantry_y:.1f}) mm")
    print(f"  Arm local target: ({arm_local_x:.1f}, {arm_local_y:.1f}) mm")

    # Step 1: move gantry
    cur_gx, cur_gy = gantry.get_position()
    dx = gantry_x - cur_gx
    dy = gantry_y - cur_gy

    if abs(dx) > 0.1 or abs(dy) > 0.1:
        print(f"  Moving gantry by ({dx:.1f}, {dy:.1f}) mm ...")
        gantry.move_relative(dx, dy, wait=True)
        print(f"  Gantry at: ({gantry.pos_x:.1f}, {gantry.pos_y:.1f}) mm")
    else:
        print(f"  Gantry already in position")

    # Step 2: rotate local coords from gantry frame to arm frame, then move
    arm_x, arm_y = gantry_to_arm_coords(arm_local_x, arm_local_y)
    print(f"  Arm local (gantry frame): ({arm_local_x:.1f}, {arm_local_y:.1f}) mm")
    print(f"  Arm command (arm frame):  ({arm_x:.1f}, {arm_y:.1f}, {arm_z_mm:.1f}) mm")
    success = arm.move_to_xyz(arm_x, arm_y, arm_z_mm)

    if success:
        print(f"  DONE - end-effector at global ({target_x_mm:.1f}, {target_y_mm:.1f}) mm")
    else:
        print(f"  WARNING - arm move may have failed")

    return success


def go_home(gantry: GantryController, arm: RoArmController,
            arm_z_mm: float = ARM_DEFAULT_Z_MM):
    """Return both gantry and arm to home (0, 0)."""
    print(f"\n{'='*60}")
    print("HOMING: returning to (0, 0)")
    print(f"{'='*60}")

    # First send arm to init position
    print("  Sending arm to init position...")
    arm.init_position()

    # Then move gantry to 0,0
    gx, gy = gantry.get_position()
    if abs(gx) > 0.1 or abs(gy) > 0.1:
        print(f"  Moving gantry from ({gx:.1f}, {gy:.1f}) to (0, 0) ...")
        gantry.move_absolute(0.0, 0.0, wait=True)
        print(f"  Gantry at home")
    else:
        print(f"  Gantry already at home")


def interactive_mode(gantry: GantryController, arm: RoArmController,
                     reach_radius_mm: float, arm_z_mm: float):
    """Interactive command loop."""
    print(f"\n{'='*60}")
    print("INTERACTIVE GANTRY + ARM CONTROLLER")
    print(f"{'='*60}")
    print(f"  Arm reach radius: {reach_radius_mm:.0f} mm")
    print(f"  Arm Z height:     {arm_z_mm:.0f} mm")
    print(f"  Gantry position:  (0, 0) mm  [homed]")
    print()
    print("Commands:")
    print("  x, y          - go to position (mm), e.g.: 200, 200")
    print("  gantry x, y   - move gantry only (mm)")
    print("  arm x, y      - move arm only (mm, local coords)")
    print("  init           - send arm to init/home position")
    print("  home           - return everything to (0, 0)")
    print("  pos            - show current positions")
    print("  radius N       - set arm reach radius (mm)")
    print("  z N            - set arm Z height (mm)")
    print("  quit / exit    - exit")
    print()

    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts = raw.lower().split()

        if parts[0] in ("quit", "exit", "q"):
            break

        elif parts[0] == "home":
            go_home(gantry, arm, arm_z_mm)

        elif parts[0] == "init":
            arm.init_position()

        elif parts[0] == "pos":
            gx, gy = gantry.get_position()
            print(f"  Gantry: ({gx:.1f}, {gy:.1f}) mm")
            print(f"  Reach radius: {reach_radius_mm:.0f} mm")
            print(f"  Arm Z: {arm_z_mm:.0f} mm")

        elif parts[0] == "radius" and len(parts) >= 2:
            try:
                reach_radius_mm = float(parts[1])
                print(f"  Reach radius set to {reach_radius_mm:.0f} mm")
            except ValueError:
                print("  Invalid number")

        elif parts[0] == "z" and len(parts) >= 2:
            try:
                arm_z_mm = float(parts[1])
                print(f"  Arm Z set to {arm_z_mm:.0f} mm")
            except ValueError:
                print("  Invalid number")

        elif parts[0] == "gantry":
            try:
                coords = raw[len("gantry"):].replace(",", " ").split()
                gx_mm = float(coords[0])
                gy_mm = float(coords[1])
                print(f"  Moving gantry to ({gx_mm:.1f}, {gy_mm:.1f}) mm ...")
                gantry.move_absolute(gx_mm, gy_mm, wait=True)
                print(f"  Gantry at: ({gantry.pos_x:.1f}, {gantry.pos_y:.1f}) mm")
            except (ValueError, IndexError):
                print("  Usage: gantry x, y")

        elif parts[0] == "arm":
            try:
                coords = raw[len("arm"):].replace(",", " ").split()
                ax_mm = float(coords[0])
                ay_mm = float(coords[1])
                arm_x, arm_y = gantry_to_arm_coords(ax_mm, ay_mm)
                print(f"  Gantry frame: ({ax_mm:.1f}, {ay_mm:.1f}) -> Arm frame: ({arm_x:.1f}, {arm_y:.1f}, {arm_z_mm:.1f}) mm")
                arm.move_to_xyz(arm_x, arm_y, arm_z_mm)
            except (ValueError, IndexError):
                print("  Usage: arm x, y  (mm, local to arm base)")

        else:
            # Default: parse as "x, y" target coordinates
            try:
                coords = raw.replace(",", " ").split()
                tx_mm = float(coords[0])
                ty_mm = float(coords[1])
                goto_position(gantry, arm, tx_mm, ty_mm, arm_z_mm, reach_radius_mm)
            except (ValueError, IndexError):
                print("  Unknown command. Enter 'x, y' coordinates or type 'help'.")


def main():
    parser = argparse.ArgumentParser(
        description="Coordinated gantry + RoArm controller (ROS2 mode)"
    )
    parser.add_argument(
        "--gantry-port", type=str, required=True,
        help="Serial port for the Arduino gantry (e.g., /dev/ttyUSB1 or COM3)"
    )
    parser.add_argument(
        "--baud", type=int, default=115200,
        help="Baud rate for gantry serial connection (default: 115200)"
    )
    parser.add_argument(
        "--reach-radius", type=float, default=ARM_REACH_RADIUS_MM,
        help=f"Arm reach radius in mm (default: {ARM_REACH_RADIUS_MM})"
    )
    parser.add_argument(
        "--arm-z", type=float, default=ARM_DEFAULT_Z_MM,
        help=f"Default arm Z height in mm (default: {ARM_DEFAULT_Z_MM})"
    )
    parser.add_argument(
        "--goto", nargs=2, type=float, metavar=("X", "Y"),
        help="Go to position (mm) then exit. E.g.: --goto 200 200"
    )

    args = parser.parse_args()

    print("Connecting to gantry Arduino...")
    gantry = GantryController(args.gantry_port, args.baud)
    print(f"  Connected on {args.gantry_port}")

    rclpy.init()
    ros2_node = Node('gantry_arm_controller')
    print("  ROS2 node initialized")

    arm = RoArmController(node=ros2_node)
    print("  RoArm via ROS2 service calls")

    if not arm.wait_for_service(timeout_sec=10.0):
        print("ERROR: /move_point_cmd not available. Is movepointcmd running?")
        gantry.close()
        ros2_node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    try:
        if args.goto:
            goto_position(
                gantry, arm,
                args.goto[0], args.goto[1],
                args.arm_z, args.reach_radius
            )
        else:
            interactive_mode(gantry, arm, args.reach_radius, args.arm_z)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Closing connections...")
        gantry.close()
        arm.close()
        ros2_node.destroy_node()
        rclpy.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
