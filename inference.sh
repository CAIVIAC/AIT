#!bin/bash

while getopts d:c:g:v: option
do
    case "${option}"
        in
        d) DEVICEID=${OPTARG};;
        c) CHECKEPOCH=${OPTARG};;
        g) GROUP=${OPTARG};;
        v) VERSION=${OPTARG};;
    esac
done

for ((c=$CHECKEPOCH; c<=20; c++))
do
    CUDA_VISIBLE_DEVICES=$DEVICEID python test_net_trans.py --dataset coco --net res50\
        --s 1 --checkepoch $c --p 6655 --cuda --g $GROUP --a 4 --version $VERSION
done
