import tensorflow as tf
import cv2
import time
import argparse

from posenet.posenet_factory import load_model
from posenet.utils import draw_skel_and_kp

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet50')  # mobilenet resnet50
parser.add_argument('--stride', type=int, default=16)  # 8, 16, 32 (max 16 for mobilenet)
parser.add_argument('--quant_bytes', type=int, default=4)  # 4 = float
parser.add_argument('--multiplier', type=float, default=1.0)  # only for mobilenet
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--input_file', type=str, help="Give the  video file location")
parser.add_argument('--output_file', type=str, help="Give the  video file location")
args = parser.parse_args()


def main():

    print('Tensorflow version: %s' % tf.__version__)
    assert tf.__version__.startswith('2.'), "Tensorflow version 2.x must be used!"

    model = args.model  # mobilenet resnet50
    stride = args.stride  # 8, 16, 32 (max 16 for mobilenet, min 16 for resnet50)
    quant_bytes = args.quant_bytes  # float
    multiplier = args.multiplier  # only for mobilenet

    posenet = load_model(model, stride, quant_bytes, multiplier)

    # for inspiration, see: https://www.programcreek.com/python/example/72134/cv2.VideoWriter
    if args.input_file is not None:
        cap = cv2.VideoCapture(args.input_file)
    else:
        raise IOError("video file not found")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(args.output_file, fourcc, fps, (width, height))

    max_pose_detections = 20

    # Scaling the input image reduces the quality of the pose detections!
    # The speed gain is about the square of the scale factor.
    posenet_input_height = 540  # scale factor for the posenet input
    posenet_input_scale = 1.0  # posenet_input_height / height  # 1.0
    posenet_input_width = int(width * posenet_input_scale)
    print("posenet_input_scale: %3.4f" % (posenet_input_scale))


    start = time.time()
    frame_count = 0

    ret, frame = cap.read()

    while ret:
        if posenet_input_scale == 1.0:
            frame_rescaled = frame  # no scaling
        else:
            frame_rescaled = \
                cv2.resize(frame, (posenet_input_width, posenet_input_height), interpolation=cv2.INTER_LINEAR)

        pose_scores, keypoint_scores, keypoint_coords = posenet.estimate_multiple_poses(frame_rescaled, max_pose_detections)

        keypoint_coords_upscaled = keypoint_coords / posenet_input_scale
        overlay_frame = draw_skel_and_kp(
            frame, pose_scores, keypoint_scores, keypoint_coords_upscaled,
            min_pose_score=0.15, min_part_score=0.1)

        frame_count += 1
        # This is uncompressed video. cv2 has no way to write compressed videos, so we'll have to use ffmpeg to
        # compress it afterwards! See:
        # https://stackoverflow.com/questions/25998799/specify-compression-quality-in-python-for-opencv-video-object
        video_writer.write(overlay_frame)
        ret, frame = cap.read()

    print('Average FPS: ', frame_count / (time.time() - start))

    video_writer.release()
    cap.release()

if __name__ == "__main__":
    main()
