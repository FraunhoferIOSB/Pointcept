#!/usr/bin/env bash

docker run -it \
    --shm-size=50g \
    --net=host \
    --gpus device=0 \
    -v /data/goose/:/workspace/Pointcept/data/goose/ \
    -v $HOME/git/Pointcept/:/workspace/Pointcept/ \
    docker.io/pointcept/pointcept:pytorch2.0.1-cuda11.7-cudnn8-devel \
    bash

