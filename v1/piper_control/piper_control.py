"""Simple control wrapper for piper_sdk."""

import dataclasses
import time
from enum import IntEnum
from typing import Callable, Literal, Sequence, TypeGuard

import numpy as np
import piper_sdk

# Global constants
DEG_TO_RAD = 0.017444
RAD_TO_DEG = 1 / DEG_TO_RAD
GRIPPER_ANGLE_MAX = 0.07  # 70mm
GRIPPER_EFFORT_MAX = 2.0  # 2 Nm


class EmergencyStop(IntEnum):
  INVALID = 0x00
  STOP = 0x01
  RESUME = 0x02


def validate_emergency_stop(
    state: EmergencyStop,
) -> TypeGuard[Literal[0, 1, 2]]:
  return state in {0, 1, 2}


# https://github.com/agilexrobotics/piper_sdk/blob/4eddfcf817cd87de9acee316a72cf5b988025378/piper_msgs/msg_v2/feedback/arm_status.py#L108
class ControlMode(IntEnum):
  STANDBY = 0x00
  CAN_COMMAND = 0x01
  TEACH_MODE = 0x02
  ETHERNET = 0x03
  WIFI = 0x04
  REMOTE = 0x05
  LINKAGE_TEACHING = 0x06
  OFFLINE_TRAJECTORY = 0x07


def validate_control_mode(
    mode: ControlMode,
) -> TypeGuard[Literal[0, 1, 3, 4, 7]]:
  """
  Validate the control mode is one of the allowed values from Piper SDK.

  This function is mainly here for type checking and making linters happy.

  Args:
      mode (ControlMode): The control mode to validate.

  Returns:
      bool: True if the control mode is valid, False otherwise.
  """
  return mode in {0, 1, 3, 4, 7}


class MoveMode(IntEnum):
  POSITION = 0x00
  JOINT = 0x01
  LINEAR = 0x02
  CIRCULAR = 0x03
  MIT = 0x04


def validate_move_mode(mode: MoveMode) -> TypeGuard[Literal[0, 1, 2, 3, 4]]:
  """
  Validate the move mode is one of the allowed values from Piper SDK.

  This function is mainly here for type checking and making linters happy.

  Args:
      mode (MoveMode): The move mode to validate.

  Returns:
      bool: True if the move mode is valid, False otherwise.
  """
  return mode in {0, 1, 2, 3, 4}


class ArmController(IntEnum):
  POSITION_VELOCITY = 0x00
  MIT = 0xAD
  INVALID = 0xFF


def validate_arm_controller(
    controller: ArmController,
) -> TypeGuard[Literal[0, 173, 255]]:
  """
  Validate the arm controller is one of the allowed values from Piper SDK.

  This function is mainly here for type checking and making linters happy.

  Args:
      controller (ArmController): The arm controller to validate.

  Returns:
      bool: True if the arm controller is valid, False otherwise.
  """
  return controller in {0, 173, 255}


# https://github.com/agilexrobotics/piper_sdk/blob/4eddfcf817cd87de9acee316a72cf5b988025378/piper_msgs/msg_v2/feedback/arm_status.py#L117
class ArmStatus(IntEnum):
  NORMAL = 0x00
  EMERGENCY_STOP = 0x01
  NO_SOLUTION = 0x02
  SINGULARITY = 0x03
  TARGET_ANGLE_EXCEEDS_LIMIT = 0x04
  JOINT_COMMUNICATION_EXCEPTION = 0x05
  JOINT_BRAKE_NOT_RELEASED = 0x06
  COLLISION = 0x07
  OVERSPEED_DURING_TEACHING = 0x08
  JOINT_STATUS_ABNORMAL = 0x09
  OTHER_EXCEPTION = 0x0A
  TEACHING_RECORD = 0x0B
  TEACHING_EXECUTION = 0x0C
  TEACHING_PAUSE = 0x0D
  MAIN_CONTROLLER_NTC_OVER_TEMPERATURE = 0x0E
  RELEASE_RESISTOR_NTC_OVER_TEMPERATURE = 0x0F


# https://github.com/agilexrobotics/piper_sdk/blob/4eddfcf817cd87de9acee316a72cf5b988025378/piper_msgs/msg_v2/feedback/arm_status.py#L140
class TeachStatus(IntEnum):
  OFF = 0x00
  START_RECORD = 0x01
  END_RECORD = 0x02
  EXECUTE = 0x03
  PAUSE = 0x04
  CONTINUE = 0x05
  TERMINATE = 0x06
  MOVE_TO_START = 0x07


# https://github.com/agilexrobotics/piper_sdk/blob/4eddfcf817cd87de9acee316a72cf5b988025378/piper_msgs/msg_v2/feedback/arm_status.py#L149
class MotionStatus(IntEnum):
  REACHED_TARGET = 0x00
  NOT_YET_REACHED_TARGET = 0x01


class GripperCode(IntEnum):
  DISABLE = 0x00
  ENABLE = 0x01
  DISABLE_AND_CLEAR_ERROR = 0x02
  ENABLE_AND_CLEAR_ERROR = 0x03


class ArmInstallationPos(IntEnum):
  UPRIGHT = 0x01  # Horizontal upright

  # Side mount left. In this orientation, the rear cable is facing backward and
  # the green LED is above the cable socket.
  LEFT = 0x02

  # Side mount right. In this orientation, the rear cable is facing backward and
  # the green LED is below the cable socket.
  RIGHT = 0x03


JOINT_LIMITS_RAD = {
    "min": [-2.6179, 0.0, -2.967, -1.745, -1.22, -2.09439],
    "max": [2.6179, 3.14, 0.0, 1.745, 1.22, 2.09439],
}

# Torque limits as absolute magnitude per joint. Units are whatever units
# of torque the Piper MIT control accepts (Nm?). These limits make sense
# only in the case of an upright robot orientation.
MIT_TORQUE_LIMITS = [0.5, 3.0, 2.0, 2.0, 2.0, 0.5]

REST_POSITION = [0.0, 0.05, 0.0, 0.02, 0.5, 0.0]
DOWN_POSITION = [0.0, 1.6, -0.85, 0.0, 0.75, 0.0]

# For some reason in MIT mode, some of the joints behave in a "reverse" manner,
# whether for direct torque commands or for setting a desired position reference.
# this mapping tells us which joints are 'flipped'
_MIT_JOINT_FLIP = [True, True, False, True, False, True]


@dataclasses.dataclass
class MitJointMoveCommand:
  """Command bundle for moving a single joint using MIT mode."""

  joint_idx: int  # 0-indexed joint id, must be in range 0 - 5.
  target_pos: float  # joint position in radians.
  p_gain: float = (
      10.0  # position gain for the PD controller. Must be in range 0 - 100.
  )
  d_gain: float = (
      0.8  # derivative gain for the PD controller. Must be in range 0 - 5.
  )


@dataclasses.dataclass
class MitTorqueCommand:
  """Command bundle for applying a torque on a single joint using MIT mode."""

  joint_idx: int  # 0-indexed joint id, must be in range 0 - 5.
  torque: float  # joint torque in (presumably) Nm.


class PiperControl:
  """
  A wrapper around the Piper robot SDK to provide high-level control and state
  functionality.
  """

  def __init__(
      self,
      can_port: str = "can0",
  ) -> None:
    """
    Initializes the PiperControl with a specified CAN port.

    Args:
      can_port (str): The CAN interface port name (e.g., "can0").
    """
    self.can_port = can_port

    self.piper = piper_sdk.C_PiperInterface_V2(can_name=can_port)
    self.piper.ConnectPort()

  def set_installation_pos(
      self, installation_pos: ArmInstallationPos = ArmInstallationPos.UPRIGHT
  ) -> None:
    """Sets the robot installation pose, call this right after connecting."""
    self.piper.MotionCtrl_2(0x01, 0x01, 0, 0, 0, installation_pos.value)

  def get_arm_status(self) -> piper_sdk.C_PiperInterface_V2.ArmStatus:
    """
    Gets the current control mode of the robot.

    Returns:
      ArmStatus: The current arm status.
    """
    return self.piper.GetArmStatus()

  @property
  def control_mode(self) -> ControlMode:
    return ControlMode(self.get_arm_status().arm_status.ctrl_mode)

  @property
  def motion_status(self) -> MotionStatus:
    return MotionStatus(self.get_arm_status().arm_status.motion_status)

  def get_gripper_status(
      self,
  ) -> piper_sdk.C_PiperInterface_V2.ArmGripper:
    """
    Gets the current gripper status of the robot.

    Returns:
      ArmGripperStatus: The current gripper status.
    """
    return self.piper.GetArmGripperMsgs()

  def show_status(self, arm_status=None) -> None:
    """
    Prints a human friendly status of the arm and gripper.

    Args:
      arm_status: Optional arm status to display. If None, the current status
        will be fetched from the robot.
    """
    error_names = {False: "OK", True: "ERROR"}

    arm_status = self.get_arm_status()
    print("Arm Status:")
    print(f"ctrl_mode: {ControlMode(arm_status.arm_status.ctrl_mode).name}")
    print(f"arm_status: {ArmStatus(arm_status.arm_status.arm_status).name}")
    print(f"mode_feed: {MoveMode(arm_status.arm_status.mode_feed).name}")
    print(f"teach_mode: {TeachStatus(arm_status.arm_status.teach_status).name}")
    print(
        "motion_status:"
        f" {MotionStatus(arm_status.arm_status.motion_status).name}"
    )
    print(f"trajectory_num: {arm_status.arm_status.trajectory_num}")
    print(f"err_code: {arm_status.arm_status.err_code}")

    # https://github.com/agilexrobotics/piper_sdk/blob/4eddfcf817cd87de9acee316a72cf5b988025378/piper_msgs/msg_v2/feedback/arm_status.py#L228
    # Decoding err_code manually here because afaict the setter is not used to
    # update the err_status
    err_code = arm_status.arm_status.err_code
    limit_errors = [bool(err_code & (1 << b)) for b in range(6)]
    comms_errors = [bool(err_code & (1 << (b + 9))) for b in range(6)]
    motor_errors = self.get_motor_errors()
    for i in range(6):
      limit = limit_errors[i]
      comms = comms_errors[i]
      motor = motor_errors[i]
      print(
          f"  Joint {i+1}:"
          f" limit={error_names[limit]}"
          f" comms={error_names[comms]}"
          f" motor={error_names[motor]}"
      )

    gripper_status = self.get_gripper_status()
    print("\nGripper Status:")
    foc_status = gripper_status.gripper_state.foc_status
    print(f"  voltage_too_low     : {error_names[foc_status.voltage_too_low]}")
    print(
        f"  motor_overheating   : {error_names[foc_status.motor_overheating]}"
    )
    print(
        f"  driver_overcurrent  : {error_names[foc_status.driver_overcurrent]}"
    )
    print(
        f"  driver_overheating  : {error_names[foc_status.driver_overheating]}"
    )
    print(f"  sensor_status       : {error_names[foc_status.sensor_status]}")
    print(
        f"  driver_error_status : {error_names[foc_status.driver_error_status]}"
    )
    print(f"  driver_enable_status: {foc_status.driver_enable_status}")
    print(f"  homing_status       : {error_names[foc_status.homing_status]}")

  def is_gripper_enabled(self) -> bool:
    """
    Checks if the gripper is enabled.
    """
    gripper_msgs = self.piper.GetArmGripperMsgs()
    return gripper_msgs.gripper_state.foc_status.driver_enable_status

  def is_arm_enabled(self) -> bool:
    """
    Checks if the arm is enabled.
    """
    arm_msgs = self.piper.GetArmLowSpdInfoMsgs()
    return (
        arm_msgs.motor_1.foc_status.driver_enable_status
        and arm_msgs.motor_2.foc_status.driver_enable_status
        and arm_msgs.motor_3.foc_status.driver_enable_status
        and arm_msgs.motor_4.foc_status.driver_enable_status
        and arm_msgs.motor_5.foc_status.driver_enable_status
        and arm_msgs.motor_6.foc_status.driver_enable_status
    )

  def is_enabled(self) -> bool:
    """
    Checks if the robot arm and gripper are enabled.

    Returns:
      bool: True if the arm and gripper are enabled, False otherwise.
    """
    return self.is_arm_enabled() and self.is_gripper_enabled()

  def disable_arm(
      self,
      *,
      timeout_seconds: float = 10.0,
  ):
    """
    Disables the arm.

    WARNING: This depowers the arm and it will drop if not supported.
    """
    timeout_trigger = _create_timeout(timeout_seconds)
    while True:
      self.set_emergency_stop(EmergencyStop.RESUME)
      time.sleep(1.0)
      if (
          self.get_arm_status().arm_status.ctrl_mode == ControlMode.STANDBY
          and self.get_arm_status().arm_status.arm_status == 0x00  # 0 = normal
          and self.get_arm_status().arm_status.teach_status == 0x00  # 0 = off
      ):
        break
      timeout_trigger()

  def enable_arm(
      self,
      arm_controller: ArmController = ArmController.POSITION_VELOCITY,
      move_mode: MoveMode = MoveMode.POSITION,
      *,speed,
      timeout_seconds: float = 10.0,
  ):
    """
    Enables the arm.

    Args:
      arm_controller (ArmController): The arm controller to use.
      move_mode (MoveMode): The move mode to use.
    """
    timeout_trigger = _create_timeout(timeout_seconds)
    while True:
      self.piper.EnableArm(7)
      time.sleep(1.0)
      if self.is_arm_enabled():
        break
      timeout_trigger()
    # enable motion
    # Don't loop on this because sometimes (eg when coming out of teach mode) it
    # fails to set the control mode. If that happens you need to call
    # disable_arm() and then call this function again. This is not done
    # automatically because disable_arm() will cause the arm to lose power and
    # crash down if it is not supported, so it's pretty rude to do that
    # unexpectedly.
    self.piper.MotionCtrl_2(
        ControlMode.CAN_COMMAND,
        move_mode.value,
        speed,  # move speed
        arm_controller.value,
    )
    time.sleep(1.0)

  def reset_arm(
      self,
      arm_controller: ArmController = ArmController.POSITION_VELOCITY,
      move_mode: MoveMode = MoveMode.POSITION,
      *,
      timeout_seconds: float = 10.0,
  ):
    """
    Resets the arm.

    WARNING: This depowers the arm and it will drop if not supported.

    Args:
      arm_controller (ArmController): The arm controller to use when enabling.
      move_mode (MoveMode): The move mode to use when enabling.
    """
    timeout_trigger = _create_timeout(timeout_seconds)
    while True:
      self.disable_arm(timeout_seconds=timeout_seconds)
      self.enable_arm(
          arm_controller=arm_controller,
          move_mode=move_mode,
          speed = 100,
          timeout_seconds=timeout_seconds,
      )
      # Sometimes (eg when coming out of teach mode) the first round of resets
      # don't work, keep going until they do.
      if self.get_arm_status().arm_status.ctrl_mode == ControlMode.CAN_COMMAND:
        break
      timeout_trigger()

  def disable_gripper(self, *, timeout_seconds: float = 10.0):
    """
    Disables the gripper.
    """
    timeout_trigger = _create_timeout(timeout_seconds)
    while True:
      self.piper.GripperCtrl(0, 0, GripperCode.DISABLE_AND_CLEAR_ERROR, 0)
      time.sleep(1.0)
      if not self.is_gripper_enabled():
        break
      timeout_trigger()

  def enable_gripper(self, *, timeout_seconds: float = 10.0):
    """
    Enables the gripper.
    """
    timeout_trigger = _create_timeout(timeout_seconds)
    while True:
      self.piper.GripperCtrl(0, 1000, GripperCode.ENABLE, 0)
      time.sleep(1.0)
      if self.is_gripper_enabled():
        break
      else:
        self.disable_gripper(timeout_seconds=timeout_seconds)
      timeout_trigger()

  def reset_gripper(self, *, timeout_seconds: float = 10.0):
    """
    Resets the gripper.
    """
    timeout_trigger = _create_timeout(timeout_seconds)
    while True:
      self.disable_gripper(timeout_seconds=timeout_seconds)
      self.enable_gripper(timeout_seconds=timeout_seconds)
      if self.is_gripper_enabled():
        break
      timeout_trigger()

  def disable(self, *, timeout_seconds: float = 10.0) -> None:
    """
    Disables the robot arm and gripper.

    WARNING: This depowers the arm and it will drop if not supported.
    """
    self.disable_arm(timeout_seconds=timeout_seconds)
    self.disable_gripper(timeout_seconds=timeout_seconds)

  def enable(
      self,
      arm_controller: ArmController = ArmController.POSITION_VELOCITY,
      move_mode: MoveMode = MoveMode.POSITION,
      *,
      timeout_seconds: float = 10.0,
  ) -> None:
    """
    Enables the robot arm and gripper.
    """
    self.enable_arm(
        arm_controller=arm_controller,
        move_mode=move_mode,
        speed = 100,
        timeout_seconds=timeout_seconds,
    )
    self.enable_gripper(timeout_seconds=timeout_seconds)

  def reset(
      self,
      arm_controller: ArmController = ArmController.POSITION_VELOCITY,
      move_mode: MoveMode = MoveMode.POSITION,
      *,
      timeout_seconds: float = 10.0,
  ) -> None:
    """
    Resets the robot arm and gripper.

    WARNING: This depowers the arm and it will drop if not supported.

    """
    self.reset_arm(
        arm_controller=arm_controller,
        move_mode=move_mode,
        timeout_seconds=timeout_seconds,
    )
    self.reset_gripper(timeout_seconds=timeout_seconds)

  def _standby(
      self,
      move_mode: MoveMode = MoveMode.JOINT,
      arm_controller: ArmController = ArmController.POSITION_VELOCITY,
  ) -> None:
    """
    Puts the robot into standby mode.
    """
    if not validate_move_mode(move_mode):
      raise ValueError(f"Invalid move mode: {move_mode}")
    if not validate_arm_controller(arm_controller):
      raise ValueError(f"Invalid arm controller: {arm_controller}")
    self.piper.MotionCtrl_2(
        ControlMode.STANDBY,
        move_mode,
        0,
        arm_controller,
    )

  def _enable_motion(
      self,
      speed: int = 100,
      move_mode: MoveMode = MoveMode.JOINT,
      ctrl_mode: ControlMode = ControlMode.CAN_COMMAND,
      arm_controller: ArmController = ArmController.POSITION_VELOCITY,
  ) -> None:
    """
    Enables motion control for the arm and gripper.

    Args:
      speed (int): Speed setting for the motion control.
      move_mode (MoveMode): Move mode to use (e.g., POSITION, JOINT).
      ctrl_mode (ControlMode): Control mode to use (e.g., CAN_COMMAND).
      arm_controller (ArmController): MIT mode to use (e.g., POSITION_VELOCITY).
    """
    if not validate_move_mode(move_mode):
      raise ValueError(f"Invalid move mode: {move_mode}")
    if not validate_arm_controller(arm_controller):
      raise ValueError(f"Invalid arm controller: {arm_controller}")
    if not validate_control_mode(ctrl_mode):
      raise ValueError(f"Invalid control mode: {ctrl_mode}")
    self.piper.MotionCtrl_2(
        ctrl_mode,
        move_mode,
        speed,
        arm_controller,
    )

  def get_joint_positions(self) -> list[float]:
    """
    Returns the current joint positions as a sequence of floats (radians).

    Returns:
      Sequence[float]: Joint positions in radians.
    """
    return [
        self.piper.GetArmJointMsgs().joint_state.joint_1 / 1e3 * DEG_TO_RAD,
        self.piper.GetArmJointMsgs().joint_state.joint_2 / 1e3 * DEG_TO_RAD,
        self.piper.GetArmJointMsgs().joint_state.joint_3 / 1e3 * DEG_TO_RAD,
        self.piper.GetArmJointMsgs().joint_state.joint_4 / 1e3 * DEG_TO_RAD,
        self.piper.GetArmJointMsgs().joint_state.joint_5 / 1e3 * DEG_TO_RAD,
        self.piper.GetArmJointMsgs().joint_state.joint_6 / 1e3 * DEG_TO_RAD,
    ]

  def get_joint_velocities(self) -> list[float]:
    """
    Returns the current joint velocities as a sequence of floats.

    Returns:
      Sequence[float]: Joint velocities in radians per second.
    """
    raw_speeds = [
        self.piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed,
        self.piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed,
        self.piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed,
        self.piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed,
        self.piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed,
        self.piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed,
    ]

    # API reports speeds in milli-degrees. Convert raw speeds to degrees first,
    # then to radians.
    return [speed / 1e3 * DEG_TO_RAD for speed in raw_speeds]

  def get_joint_efforts(self) -> list[float]:
    """
    Returns the current joint efforts as a sequence of floats.

    Returns:
      Sequence[float]: Joint efforts in Nm.
    """
    return [
        self.piper.GetArmHighSpdInfoMsgs().motor_1.effort / 1e3,
        self.piper.GetArmHighSpdInfoMsgs().motor_2.effort / 1e3,
        self.piper.GetArmHighSpdInfoMsgs().motor_3.effort / 1e3,
        self.piper.GetArmHighSpdInfoMsgs().motor_4.effort / 1e3,
        self.piper.GetArmHighSpdInfoMsgs().motor_5.effort / 1e3,
        self.piper.GetArmHighSpdInfoMsgs().motor_6.effort / 1e3,
    ]

  def get_gripper_state(self) -> tuple[float, float]:
    """
    Returns the current gripper state as a tuple of angle and effort.

    Returns:
      tuple[float, float]: (gripper_angle, gripper_effort)
    """
    raw_angle = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle
    raw_effort = self.piper.GetArmGripperMsgs().gripper_state.grippers_effort

    angle = raw_angle / 1e6
    effort = raw_effort / 1e3

    return angle, effort

  def set_joint_positions(
      self,
      positions: Sequence[float],
      *,
      wait_for_completion: bool = False,
  ) -> None:
    """
    Sets the joint positions using JointCtrl.

    Args:
      positions (Sequence[float]): A list of joint angles in radians.
    """
    self._enable_motion(
        arm_controller=ArmController.POSITION_VELOCITY, move_mode=MoveMode.JOINT
    )
    joint_angles = []
    for i, pos in enumerate(positions):
      min_rad, max_rad = JOINT_LIMITS_RAD["min"][i], JOINT_LIMITS_RAD["max"][i]
      clipped_pos = min(max(pos, min_rad), max_rad)
      pos_deg = clipped_pos * RAD_TO_DEG
      joint_angle = round(pos_deg * 1e3)  # Convert to millidegrees
      joint_angles.append(joint_angle)
    self.piper.JointCtrl(*joint_angles)  # pylint: disable=no-value-for-parameter

    if wait_for_completion:
      while True:
        time.sleep(0.05)
        if self.motion_status == MotionStatus.REACHED_TARGET:
          break

  def set_joint_zero_positions(self, joints: Sequence[int]) -> None:
    """
    Re-zeros the specified joints at their current positions.

    Args:
      joints (Sequence[int]): The indices of the joints to zero (zero-indexed).
    """

    def _validate_zero_indexed_joint(
        joint: int,
    ) -> TypeGuard[Literal[0, 1, 2, 3, 4, 5]]:
      return 0 <= joint <= 5

    if not all(_validate_zero_indexed_joint(j) for j in joints):
      raise ValueError(f"Invalid joint indices: {joints}")

    for joint in joints:
      self.piper.JointConfig(
          joint_num=joint + 1,  # type: ignore
          set_zero=0xAE,
          acc_param_is_effective=0,
          max_joint_acc=0,
          clear_err=0,
      )

  def set_joint_mit_ctrl(
      self,
      positions: Sequence[float],
      velocities: Sequence[float],
      kps: Sequence[float],
      kds: Sequence[float],
      efforts: Sequence[float],
      motor_idxs: Sequence[int] = range(6),
  ) -> None:
    """
    Sets the MIT control mode for multiple joints.

    Args:
      positions (Sequence[float]): Desired positions in radians.
      velocities (Sequence[float]): Desired velocities in radians per second.
      kps (Sequence[float]): Proportional gains.
      kds (Sequence[float]): Derivative gains.
      efforts (Sequence[float]): Target torques in Nm.
      motor_idxs (Sequence[int]): Motor indices to control (default: range(6)).
    """
    self._enable_motion(arm_controller=ArmController.MIT)
    for idx in motor_idxs:
      pos_deg = positions[idx] * RAD_TO_DEG
      vel_ref = velocities[idx]
      kp = kps[idx]
      kd = kds[idx]
      t_ref = efforts[idx]
      self.piper.JointMitCtrl(idx + 1, pos_deg, vel_ref, kp, kd, t_ref)

  def set_cartesian_position(self, pose: Sequence[float]) -> None:
    """
    Sets the Cartesian position and orientation of the robot end-effector.

    Args:
        pose: [x, y, z, roll, pitch, yaw] in meters and radians.
    """
    self._enable_motion(
        move_mode=MoveMode.POSITION,
        arm_controller=ArmController.POSITION_VELOCITY,
    )
    x_mm = round(pose[0] * 1e6)
    y_mm = round(pose[1] * 1e6)
    z_mm = round(pose[2] * 1e6)
    roll_deg = round(pose[3] * RAD_TO_DEG * 1e3)
    pitch_deg = round(pose[4] * RAD_TO_DEG * 1e3)
    yaw_deg = round(pose[5] * RAD_TO_DEG * 1e3)
    self.piper.EndPoseCtrl(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg)

  def get_end_effector_pose(self) -> Sequence[float]:
    """
    Returns the current end-effector pose as a sequence of floats.

    Returns:
      Sequence[float]: (x, y, z, roll, pitch, yaw) in meters and radians.
    """
    pose = self.piper.GetArmEndPoseMsgs()
    x = pose.end_pose.X_axis * 1e-6  # Convert from mm to m
    y = pose.end_pose.Y_axis * 1e-6
    z = pose.end_pose.Z_axis * 1e-6
    roll = pose.end_pose.RX_axis * 1e-3 * DEG_TO_RAD
    pitch = pose.end_pose.RY_axis * 1e-3 * DEG_TO_RAD
    yaw = pose.end_pose.RZ_axis * 1e-3 * DEG_TO_RAD
    return [x, y, z, roll, pitch, yaw]

  def set_gripper_ctrl(
      self,
      position: float | None = None,
      effort: float | None = None,
  ) -> None:
    """
    Controls the gripper by setting its position and effort.

    Args:
      position (float | None): The desired position of the gripper in meters.
        Must be between 0.0 and GRIPPER_ANGLE_MAX. If None, the position is not
        updated.
      effort (float | None): The desired effort (force) for the gripper.
        Must be between 0.0 and GRIPPER_EFFORT_MAX. If None, the effort is not
        updated.

    Returns:
      None
    """
    position_int = effort_int = 0
    if position is not None:
      #rint("recieved position:", position)
      position = min(position, GRIPPER_ANGLE_MAX)
      #rint("position before rounding:", position)
      position_int = round(position * 1e6)
      #rint("position_int after rounding:", position_int)
    if effort is not None:
      effort = min(effort, GRIPPER_EFFORT_MAX)
      effort_int = round(effort * 1e3)

    print(f"sending {position_int=} {effort_int=}")
    self.piper.GripperCtrl(position_int, effort_int, 0x01, 0)

  def set_emergency_stop(self, state: EmergencyStop):
    if not validate_emergency_stop(state):
      raise ValueError(f"Invalid emergency stop state {state}.")

    self.piper.MotionCtrl_1(state, 0, 0)

  def get_motor_errors(self) -> list[bool]:
    arm_msgs = self.piper.GetArmLowSpdInfoMsgs()
    return [
        # motor_xx attributes are 1 indexed
        getattr(arm_msgs, f"motor_{i + 1}").foc_status.driver_error_status
        for i in range(6)
    ]

  def relax_joints_mit(
      self, joint_indices: Sequence[int] | None = None
  ) -> None:
    """Relaxes specific joints, using MIT mode. The joints are zero-indexed."""
    if joint_indices is None:
      joint_indices = range(6)

    for ji in joint_indices:
      if ji < 0 or ji > 5:
        raise ValueError(f"Joint index out of range (0 - 5): {joint_indices}")

      # Piper API is 1-indexed for joints so add 1.
      self.piper.JointMitCtrl(ji + 1, 0.0, 0.0, 0.0, 0.0, 0.0)

  def move_to_joint_pos_mit(
      self,
      commands: Sequence[MitJointMoveCommand],
      apply_sign_flip: bool = False,
  ) -> None:
    """Move specific joints using MIT mode.

    Args:
      commands: Per-joint move command, specifying the joint index, the target
        position in radians, and the pd gains.
      apply_sign_flip: We found that the MIT mode position command is interpeted
        as a negative for certain joints. To compensate for this, some of the
        joints have their command auto-flipped.
    """

    for cmd in commands:
      if cmd.joint_idx < 0 or cmd.joint_idx > 5:
        raise ValueError(f"Joint index out of range (0 - 5): {commands}")

      if cmd.p_gain < 0.0 or cmd.p_gain > 100.0:
        raise ValueError(f"P-gain out of range (0 - 100): {commands}")

      if cmd.d_gain < 0.0 or cmd.d_gain > 5.0:
        raise ValueError(f"D-gain out of range (0 - 5): {commands}")

      target_pos = float(
          np.clip(
              cmd.target_pos,
              JOINT_LIMITS_RAD["min"][cmd.joint_idx],
              JOINT_LIMITS_RAD["max"][cmd.joint_idx],
          )
      )

      # Flip the target position if the flip flag is set.
      if apply_sign_flip:
        target_pos = (
            -target_pos if _MIT_JOINT_FLIP[cmd.joint_idx] else target_pos
        )

      # Piper API is 1-indexed for joints so add 1.
      self.piper.JointMitCtrl(
          cmd.joint_idx + 1,  # joint index
          target_pos,  # pos ref
          0.0,  # vel ref
          cmd.p_gain,  # p-gain
          cmd.d_gain,  # d-gain
          0.0,  # raw motor torque
      )

  def apply_joint_torque_mit(
      self,
      commands: Sequence[MitTorqueCommand],
      apply_sign_flip: bool = False,
      apply_torque_limits: bool = True,
  ) -> None:
    """Apply torque to specific joints using MIT mode.

    Args:
      commands: Per-joint move command, specifying the joint index and the
        desired torque.
      apply_sign_flip: We found that the MIT mode command is interpeted as a
        negative for certain joints. To compensate for this, some of the joints
        have their command auto-flipped.
      apply_torque_limits: Whether to clip the torque magnitude per joint to a
        'safe' amount. Note: The clip magnitudes are chosen to make sense for a
        upright robot configuration.
    """

    for cmd in commands:
      # Flip the target position if the flip flag is set.
      torque = cmd.torque
      if apply_sign_flip:
        torque = -torque if _MIT_JOINT_FLIP[cmd.joint_idx] else torque

      if apply_torque_limits:
        torque = np.clip(
            torque,
            -MIT_TORQUE_LIMITS[cmd.joint_idx],
            MIT_TORQUE_LIMITS[cmd.joint_idx],
        )

      # Piper API is 1-indexed for joints so add 1.
      self.piper.JointMitCtrl(
          cmd.joint_idx + 1,  # joint index
          0.0,
          0.0,
          0.0,
          0.0,
          torque,
      )

  def gently_move_to_target(
      self,
      target_joints: Sequence[float],
      timeout_seconds: float = 3.0,
  ) -> None:
    move_steps = int(timeout_seconds / 0.005)
    p_gains = np.geomspace(0.5, 5.0, move_steps)

    for i in range(move_steps):
      joint_positions = self.get_joint_positions()
      if np.all((np.array(joint_positions) - np.array(target_joints)) < 0.05):
        print("Finished early")
        break

      mit_set_commands = []
      for j in range(len(target_joints)):
        mit_set_commands.append(
            MitJointMoveCommand(
                joint_idx=j,
                target_pos=target_joints[j],
                p_gain=p_gains[i],
            )
        )
      self.move_to_joint_pos_mit(mit_set_commands, apply_sign_flip=True)
      time.sleep(0.005)


def _create_timeout(seconds: float) -> Callable[[], None]:
  """Creates a timeout trigger that raises a TimeoutError after a given time."""
  start_time = time.time()

  def timeout_trigger() -> None:
    if time.time() - start_time > seconds:
      raise TimeoutError("Timeout")

  return timeout_trigger
