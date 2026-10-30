# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Example script demonstrating device streaming with typed callbacks.

This script shows how to:
1. Stream data from an Aria device via USB-NCM, Wi-Fi STA, or Wi-Fi SAP interface
2. Use typed callbacks for different sensor data types
3. Handle all sensor types including cameras, IMU, audio, and ML outputs (eye gaze, hand tracking, VIO)
4. Optionally record streaming data to VRS format

For raw message callback example, see device_raw_streaming.py

Usage:
    # Stream over USB-NCM (default)
    python device_streaming.py

    # Stream over Wi-Fi (when USB-NCM is not available)
    python device_streaming.py --interface wifi_sta

    # Record to VRS file
    python device_streaming.py --record-to-vrs /path/to/output.vrs
"""

import argparse
import time

import cv2
import time
from pathlib import Path

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver
from projectaria_tools.core.mps import EyeGaze, hand_tracking, OpenLoopTrajectoryPose
from projectaria_tools.core.sensor_data import (
    AudioData,
    AudioDataRecord,
    FrontendOutput,
    ImageData,
    ImageDataRecord,
    MotionData,
)

device_client = sdk_gen2.DeviceClient()


STREAMING_INTERFACE_MAP = {
    "usb": sdk_gen2.StreamingInterface.USB_NCM,
    "wifi_sta": sdk_gen2.StreamingInterface.WIFI_STA,
    "wifi_sap": sdk_gen2.StreamingInterface.WIFI_SAP,
}

OUTPUT_DIR = Path("stream")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def device_streaming(
    profile_name: str = "profile9",
    custom_profile_path: str = "",
    interface: str = "usb",
):
    # Set up the device client config to specify the device to be connected to e.g. device serial number.
    # If nothing is specified, the first device in the list of connected devices will be connected to
    config = sdk_gen2.DeviceClientConfig()
    device_client.set_client_config(config)
    try:
        device = device_client.connect()
    except Exception as e:
        print(f"Failed to connect to device: {e}")
        return None

    # Custom profile path takes priority over profile name.
    streaming_config = sdk_gen2.HttpStreamingConfig()
    if custom_profile_path:
        streaming_config.profile_json = custom_profile_path
    else:
        streaming_config.profile_name = profile_name
    streaming_config.streaming_interface = STREAMING_INTERFACE_MAP[interface]
    device.set_streaming_config(streaming_config)

    device.start_streaming()
    return device


def image_callback(image_data, image_record):
    # Convert the incoming Aria image to a NumPy array
    frame = image_data.to_numpy_array()

    # Timestamp in nanoseconds (host arrival time)
    timestamp = time.time_ns()

    # Construct filename
    filename = OUTPUT_DIR / f"rgb_{timestamp}.jpg"

    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Save image
    cv2.imwrite(
        str(filename),
        frame_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    print(f"Saved: {filename}")


def audio_callback(
    audio_data: AudioData, audio_record: AudioDataRecord, num_channels: int
):
    print(
        f"Received audio data with {len(audio_data.data)} samples and {len(audio_record.capture_timestamps_ns)} timestamps and num channels {num_channels}"
    )


def imu_callback(imu_data: MotionData, sensor_label: str):
    print(
        f"Received {sensor_label} accel data {imu_data.accel_msec2} and gyro {imu_data.gyro_radsec}"
    )


def eyegaze_callback(eyegaze_data: EyeGaze):
    print(
        f"Received EyeGaze data at timestamp {eyegaze_data.tracking_timestamp.total_seconds()} sec "
        f"with yaw={eyegaze_data.yaw:.3f} rad, pitch={eyegaze_data.pitch:.3f} rad, "
        f"depth={eyegaze_data.depth:.3f} m"
    )


def handtracking_callback(handtracking_data: hand_tracking.HandTrackingResult):
    print(
        f"Received HandTracking data at timestamp {handtracking_data.tracking_timestamp.total_seconds()} sec"
    )

    if handtracking_data.left_hand is not None:
        left_hand = handtracking_data.left_hand
        print(f"  Left hand confidence: {left_hand.confidence:.3f}")
        print(f"  Left wrist position: {left_hand.get_wrist_position_device()}")
        print(f"  Left palm position: {left_hand.get_palm_position_device()}")
        if left_hand.wrist_and_palm_normal_device is not None:
            normals = left_hand.wrist_and_palm_normal_device
            print(f"  Left wrist normal: {normals.wrist_normal_device}")
            print(f"  Left palm normal: {normals.palm_normal_device}")
    else:
        print("  Left hand: No data")

    if handtracking_data.right_hand is not None:
        right_hand = handtracking_data.right_hand
        print(f"  Right hand confidence: {right_hand.confidence:.3f}")
        print(f"  Right wrist position: {right_hand.get_wrist_position_device()}")
        print(f"  Right palm position: {right_hand.get_palm_position_device()}")
        if right_hand.wrist_and_palm_normal_device is not None:
            normals = right_hand.wrist_and_palm_normal_device
            print(f"  Right wrist normal: {normals.wrist_normal_device}")
            print(f"  Right palm normal: {normals.palm_normal_device}")
    else:
        print("  Right hand: No data")


def vio_callback(vio_data: FrontendOutput):
    print(
        f"Received VIO data at timestamp {vio_data.capture_timestamp_ns} with transform_odometry_bodyimu: {vio_data.transform_odometry_bodyimu.rotation().log()} and {vio_data.transform_odometry_bodyimu.translation()} ns"
    )
    calib = vio_data.online_calib
    print(
        f"  online_calib: {calib.num_cameras()} cameras, {len(calib.imu_model_parameters)} IMUs"
    )
    for i, cam in enumerate(calib.cam_parameters):
        print(f"    cam[{i}]: type={cam.type}, intrinsics={cam.intrinsics}")
    for i, imu in enumerate(calib.imu_model_parameters):
        print(
            f"    imu[{i}]: accel_bias={imu.accel_bias_m_sec2}, gyro_bias={imu.gyro_bias_rad_sec}"
        )


def setup_streaming_receiver(device, record_to_vrs):
    # 0.0.0.0 means listen on all network interfaces.
    config = sdk_gen2.HttpServerConfig()
    config.address = "0.0.0.0"
    config.port = 6768

    stream_receiver = receiver.StreamReceiver(enable_image_decoding=True)

    stream_receiver.set_server_config(config)
    if record_to_vrs != "":
        stream_receiver.record_to_vrs(record_to_vrs)

    print("Registering typed callbacks...")
    stream_receiver.register_slam_callback(image_callback)
    stream_receiver.register_rgb_callback(image_callback)
    stream_receiver.register_audio_callback(audio_callback)
    stream_receiver.register_eye_gaze_callback(eyegaze_callback)
    stream_receiver.register_hand_pose_callback(handtracking_callback)
    stream_receiver.register_vio_callback(vio_callback)

    print("Starting streaming server...")
    stream_receiver.start_server()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")

    device.stop_streaming()
    time.sleep(2)

    stream_receiver.stop_server()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record-to-vrs",
        dest="record_to_vrs",
        type=str,
        default="",
        required=False,
        help="Output directory to save the received streaming into VRS",
    )
    parser.add_argument(
        "--profile-name",
        dest="profile_name",
        type=str,
        default="profile9",
        required=False,
        help="Name of the streaming profile to use (default: profile9)",
    )
    parser.add_argument(
        "--custom-profile-path",
        dest="custom_profile_path",
        type=str,
        default="",
        required=False,
        help="Path to a custom profile file (takes priority over --profile-name)",
    )
    parser.add_argument(
        "--interface",
        dest="interface",
        type=str,
        choices=list(STREAMING_INTERFACE_MAP.keys()),
        default="usb",
        required=False,
        help="Streaming interface (default: usb). Use wifi_sta or wifi_sap when USB-NCM is not available.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = device_streaming(
        profile_name=args.profile_name,
        custom_profile_path=args.custom_profile_path,
        interface=args.interface,
    )
    if device is None:
        exit(0)

    setup_streaming_receiver(device, args.record_to_vrs)
