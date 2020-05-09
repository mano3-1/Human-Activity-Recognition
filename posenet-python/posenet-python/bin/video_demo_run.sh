#!/usr/bin/env bash

#./bin/docker_run.sh python video_demo.py --model resnet50 --stride 16 --input_file "Pexels Videos 3552510.mp4" --output_file "Pexels Videos 3552510-with_pose.mp4"
./bin/docker_run.sh python video_demo.py --model resnet50 --stride 16 --input_file "exki.mp4" --output_file "exki_with_pose.mp4"
./bin/docker_run.sh python video_demo.py --model resnet50 --stride 16 --input_file "night-bridge.mp4" --output_file "night-bridge_with_pose.mp4"
./bin/docker_run.sh python video_demo.py --model resnet50 --stride 16 --input_file "night-colorful.mp4" --output_file "night-colorful_with_pose.mp4"
./bin/docker_run.sh python video_demo.py --model resnet50 --stride 16 --input_file "night-street.mp4" --output_file "night-street_with_pose.mp4"
./bin/docker_run.sh python video_demo.py --model resnet50 --stride 16 --input_file "pedestrians.mp4" --output_file "pedestrians_with_pose.mp4"
./bin/docker_run.sh python video_demo.py --model resnet50 --stride 16 --input_file "sidewalk.mp4" --output_file "sidewalk_with_pose.mp4"
