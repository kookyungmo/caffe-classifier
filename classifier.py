#!/usr/bin/env python

import argparse
from python.model import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("caffemodel", help="")
    parser.add_argument("deploy", help="")

    # optional
    parser.add_argument("-m", "--mean", help="")
    parser.add_argument("-l", "--labels", help="")
    parser.add_argument("-g", "--gpu", dest="gpu", action="store_true", help="")
    parser.set_defaults(gpu=False)

    args = vars(parser.parse_args())

    model = Model(caffemodel_file=args["caffemodel"],
                  deploy_file=args["deploy"],
                  mean_file=args["mean"],
                  labels_file=args["labels"],
                  gpu=args["gpu"])

    print model.classify(image_path="")
