import tensorflow as tf
import cv2
import time
import argparse
import os
from posenet.posenet_factory import load_model


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet50')  # mobilenet resnet50
parser.add_argument('--stride', type=int, default=16)  # 8, 16, 32 (max 16 for mobilenet)
parser.add_argument('--quant_bytes', type=int, default=4)  # 4 = float
parser.add_argument('--multiplier', type=float, default=1.0)  # only for mobilenet
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--num_images', type=int, default=1000)
args = parser.parse_args()


def main():

    print('Tensorflow version: %s' % tf.__version__)
    assert tf.__version__.startswith('2.'), "Tensorflow version 2.x must be used!"

    model = args.model  # mobilenet resnet50
    stride = args.stride  # 8, 16, 32 (max 16 for mobilenet)
    quant_bytes = args.quant_bytes  # float
    multiplier = args.multiplier  # only for mobilenet

    posenet = load_model(model, stride, quant_bytes, multiplier)

    num_images = args.num_images
    filenames = [
        f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    if len(filenames) > num_images:
        filenames = filenames[:num_images]

    images = {f: cv2.imread(f) for f in filenames}

    start = time.time()
    for i in range(num_images):
        image = images[filenames[i % len(filenames)]]
        posenet.estimate_multiple_poses(image)

    print('Average FPS:', num_images / (time.time() - start))


if __name__ == "__main__":
    main()
