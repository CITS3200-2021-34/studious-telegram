#!/bin/bash

if [ $# -eq 0 ]
    then
        docker build --file ./docker/qna.dockerfile -t qna .
        docker run -it qna
        exit 0
fi

if [ "$1" = "build" ]
    then
        docker build --file ./docker/qna.dockerfile -t qna .
        exit 0
fi

if [ "$1" = "start" ]
    then
        docker run -it qna
        exit 0
fi

if [ "$1" = "test" ]
    then
        echo "No tests currently defined"
        exit 0
fi

echo "Usage:"
echo $0 "           Builds docker image and starts the container"
echo $0 "build      Builds docker image"
echo $0 "start      Starts the container"
echo $0 "test       Runs tests and exits"