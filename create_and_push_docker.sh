#!/bin/bash

# usage: 
# For Cython compiled Engine: provide "cython" as a command line argument after the script file name to invoke docker build for Cython
#   eg: "sh create_and_push_docker.sh cython"
# For Python Engine: run the script without any command line argument after the file name. 
#   eg: "sh create_and_push_docker.sh"

if [[ ${1,,} == "cython" ]]; then 
    echo "Building docker with Cython compiled Python Engine"
    docker build -t eu.gcr.io/fluid-door-230710/upstride:py-`cat dist_binary_setup.py | grep version|cut -f2 -d'"'`-tf2.2.0-gpu-cython-compiled -f dist_binary.dockerfile .
    docker push eu.gcr.io/fluid-door-230710/upstride:py-`cat dist_binary_setup.py | grep version|cut -f2 -d'"'`-tf2.2.0-gpu-cython-compiled 
else
    echo "Building docker with Python Engine"
    docker build -t eu.gcr.io/fluid-door-230710/upstride:py-`cat setup.py|grep version|cut -f2 -d'"'`-tf2.2.0-gpu .
    docker push eu.gcr.io/fluid-door-230710/upstride:py-`cat setup.py|grep version|cut -f2 -d'"'`-tf2.2.0-gpu
fi
