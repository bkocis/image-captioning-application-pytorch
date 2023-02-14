#!/usr/bin/env bash

docker build --tag=image-captioning .

docker run -p 8081:8081 image-captioning
