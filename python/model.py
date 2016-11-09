#!/usr/bin/env python

import os

os.environ["GLOG_minloglevel"] = "2"
import caffe
from caffe.proto import caffe_pb2

from google.protobuf import text_format
import numpy as np
import PIL.Image
import scipy.misc


class Model(object):
    def __init__(self,
                 caffemodel_file, deploy_file, mean_file, labels_file, gpu):
        self.caffemodel_file = caffemodel_file
        self.deploy_file = deploy_file
        self.mean_file = mean_file
        self.labels_file = labels_file
        self.gpu = gpu
        self.net = None
        self.transformer = None
        self.mean = None
        self.labels = None

        self.set_net()
        self.set_transformer()
        self.set_mean()
        self.set_labels()

    def set_net(self):
        if self.gpu:
            caffe.set_mode_gpu()

        self.net = caffe.Net(self.deploy_file, self.caffemodel_file, caffe.TEST)

    def set_transformer(self):
        network = caffe_pb2.NetParameter()
        with open(self.deploy_file) as f:
            text_format.Merge(f.read(), network)

        if network.input_shape:
            dims = network.input_shape[0].dim
        else:
            dims = network.input_dim[:4]

        self.transformer = caffe.io.Transformer(inputs={"data": dims})
        self.transformer.set_transpose("data", (2, 0, 1))

        if dims[1] == 3:
            self.transformer.set_channel_swap("data", (2, 1, 0))

    def set_mean(self):
        if self.mean_file:
            with open(self.mean_file, "rb") as f:
                blob = caffe_pb2.BlobProto()
                blob.MergeFromString(f.read())

                if blob.HasField("shape"):
                    blob_dims = blob.shape
                    if len(blob_dims) != 4:
                        pass
                elif blob.HasField("num") and blob.HasField("channels") and \
                        blob.HasField("height") and blob.HasField("width"):
                    blob_dims = (blob.num, blob.channels, blob.height, blob.width)
                else:
                    pass

                mean_pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
                self.transformer.set_mean("data", mean_pixel)

    def set_labels(self):
        if self.labels_file:
            with open(self.labels_file) as f:
                self.labels = []
                for line in f:
                    label = line.strip()
                    self.labels.append(label)

    def load_image(self, path):
        _, channels, height, width = self.transformer.inputs["data"]

        if channels == 3:
            mode = "RGB"
        elif channels == 1:
            mode = "L"
        else:
            return None

        try:
            image = PIL.Image.open(path)
            image = image.convert(mode)
            image = np.array(image)

            # resize transformation (squash)
            image = scipy.misc.imresize(image, (height, width), "bilinear")

            return image
        except Exception:
            return None

    def forward(self, image):
        image_data = self.transformer.preprocess("data", image)
        self.net.blobs["data"].data[0] = image_data
        output = self.net.forward()[self.net.outputs[-1]][0]
        return output

    def classify(self, image_path, top_k=5):
        image = self.load_image(image_path)

        score = None
        if image is not None:
            score = self.forward(image)
            indices = np.argsort(score)[::-1][:top_k]

        results = []

        if score is not None:
            for i in indices:
                if self.labels is not None:
                    results.append({"label": self.labels[i], "score": score[i]})
                else:
                    results.append({"label": i, "score": score[i]})

        return results
