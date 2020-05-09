#!/usr/bin/env bash

./bin/docker_run.sh python image_demo.py --model resnet50 --stride 16 --image_dir ./images --output_dir ./output
