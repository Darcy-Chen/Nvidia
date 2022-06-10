#!/usr/bin/env python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

#!/usr/bin/python3

import jetson.inference
import jetson.utils
import numpy as np

import argparse

import os

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect.")
args = parser.parse_args()

count = 0
x = np.array(['file', 'result', 'confidence'])

for file in os.listdir(args.filename):
    # load an image (into shared CPU/GPU memory)
    if file.endswith('.jpg') and count < 10:
        img = jetson.utils.loadImage(file)

        # load the recognition network
        net = jetson.inference.imageNet(args.network)

        # classify the image
        class_idx, confidence = net.Classify(img)

        # find the object description
        class_desc = net.GetClassDesc(class_idx)
        count += 1

        result = np.array([file, class_desc, confidence * 100])
        x = np.concatenate((x, result), axis=0)

np.savetxt("result.csv", x, delimiter=",")
