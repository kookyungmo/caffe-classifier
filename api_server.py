#!/usr/bin/env python

import os
import argparse
import urllib3
import shutil
import hashlib
import PIL.Image
import json

from flask import Flask, Response
from flask import request

from python.model import Model

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
pm = urllib3.PoolManager()


@app.route("/api/classify", methods=["GET"])
def classify_get():
    image_url = request.args.get("image_url", None)
    pos = request.args.get("pos", "")

    image_path = None
    results = []

    if image_url:
        image_path = download(image_url)

    if image_path:
        pos = get_pos(pos)
        image_path = crop_image(image_path, pos)

        results = model.classify(image_path)

    return Response(response=json.dumps(results, indent=4), status=200, mimetype="application/json")


@app.route("/api/classify/local", methods=["GET"])
def classify_local_get():
    image_path = request.args.get("image_path", None)
    pos = request.args.get("pos", "")

    results = []

    if image_path:
        pos = get_pos(pos)
        image_path = crop_image(image_path, pos)

        results = model.classify(image_path)

    return Response(response=json.dumps(results, indent=4), status=200, mimetype="application/json")


def make_download_folder():
    if not os.path.isdir(download_folder):
        os.makedirs(download_folder)


def download(url):
    image_path = os.path.join(download_folder, hashlib.md5(url).hexdigest())

    try:
        with pm.request('GET', url, preload_content=False) as r, open(image_path, 'wb') as f:
            shutil.copyfileobj(r, f)

            # check image
            try:
                image = PIL.Image.open(image_path)
            except Exception:
                return None

            return image_path
        return None
    except Exception as err:
        return None


def start(port):
    app.run(host="0.0.0.0", port=port, debug=False)


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def get_pos(pos):
    pos = pos.split(",")

    if len(pos) == 4:
        err = False
        _pos = []
        for p in pos:
            if not is_float(p):
                err = True
                break
            else:
                _pos.append(float(p))
        if not err:
            pos = _pos
        else:
            pos = None
    else:
        pos = None

    return pos


def crop_image(path, pos):
    try:
        image = PIL.Image.open(path)
        image_path = "%s-crop.%s" % (path, image.format.lower())
        image.crop((pos[0], pos[1], pos[2], pos[3])).save(image_path)
        return image_path
    except Exception:
        return None

if __name__ == "__main__":
    global download_folder
    global model

    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("caffemodel", help="")
    parser.add_argument("deploy", help="")

    # optional
    parser.add_argument("--port", type=int, help="")
    parser.add_argument("--mean", help="")
    parser.add_argument("--labels", help="")
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="")
    parser.add_argument("--download_folder", help="")

    parser.set_defaults(port=8080)
    parser.set_defaults(gpu=False)
    parser.set_defaults(download_folder="/tmp/caffe-classifier")

    args = vars(parser.parse_args())

    download_folder = args["download_folder"]
    make_download_folder()

    model = Model(caffemodel_file=args["caffemodel"],
                  deploy_file=args["deploy"],
                  mean_file=args["mean"],
                  labels_file=args["labels"],
                  gpu=args["gpu"])

    start(args["port"])
