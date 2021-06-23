#!/bin/bash

xhost +local:root

docker run -it --rm \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --network=host \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --env 'DISPLAY' \
    --env="QT_X11_NO_MITSHM=1" \
    -v $(pwd):/data \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    udacity

xhost -local:root
