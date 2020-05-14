#!/bin/bash
docker build -t eu.gcr.io/fluid-door-230710/upstride:py-`cat setup.py|grep version|cut -f2 -d'"'`-tf2.2.0-gpu .
docker push eu.gcr.io/fluid-door-230710/upstride:py-`cat setup.py|grep version|cut -f2 -d'"'`-tf2.2.0-gpu
