#!/usr/bin/env bash

./bin/docker_run.sh python benchmark.py --model mobilenet --stride 16 --image_dir ./images --num_images 1000
