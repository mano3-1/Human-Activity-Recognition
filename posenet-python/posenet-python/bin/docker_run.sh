#!/usr/bin/env bash

WORK=$(pwd)

if [ -z "$POSENET_PYTHON_DEVICE" ]; then
  echo "set the environment variable POSENET_PYTHON_DEVICE to CPU or GPU, or enter your choice below:"
  read -p "Enter your device (CPU or GPU): "  device
  if [ "$device" = "GPU" ]; then
    source exportGPU.sh
  elif [ "$device" = "CPU" ]; then
    source exportCPU.sh
  else
    echo "Device configuration failed..."
    exit 1
  fi
fi

echo "device is: $POSENET_PYTHON_DEVICE"

if [ "$POSENET_PYTHON_DEVICE" = "GPU" ]; then
  image="posenet-python-gpu"
  gpu_opts="--gpus all"
else
  image="posenet-python-cpu"
  gpu_opts=""
fi

docker run $gpu_opts -it --rm -v $WORK:/work "$image" "$@"
