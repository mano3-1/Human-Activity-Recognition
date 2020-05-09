#!/usr/bin/env bash

# run this from the top-level folder of the project

WORK=$(dirname $(pwd))

docker run --gpus all -it -v $WORK:/work posenet-python tf_upgrade_v2 \
  --intree posenet-python/ \
  --outtree posenet-python_v2/ \
  --reportfile posenet-python/report.txt
