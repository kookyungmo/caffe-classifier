# caffe-classifier

## Requirements

- [caffe](https://github.com/BVLC/caffe)

## Installation

    PYTHONPATH=/Your Caffe Path/python

    pip install -r requirements.txt
    
## Demo

Classification

    python classifier.py "*.caffemodel" \
                         "deploy.prototxt" \
                         --gpu \
                         --labels "labels.txt" \
                         --mean "mean.binaryproto"

API Server

    python api_server.py "*.caffemodel" \
                         "deploy.prototxt" \
                         --gpu \
                         --labels "labels.txt" \
                         --mean "mean.binaryproto" \
                         --port 8080 \
                         --download_folder "/tmp/test"
