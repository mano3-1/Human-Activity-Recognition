#!/usr/bin/env bash

FOLDER=$1

# e.g.: $> ./inspect_saved_model.sh _tf_models/posenet/mobilenet_v1_100/stride16
./bin/docker_run.sh saved_model_cli show --dir "$FOLDER" --all
