#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import Queue

import caffe

def getCutout(image, x1, y1, x2, y2, border):
    assert(x1 >= 0 and y1 >= 0)
    assert(x2 > x1 and y2 >y1)
    assert(border >= 0)
    return cv2.getRectSubPix(image, (y2-y1 + 2*border, x2-x1 + 2*border), (((y2-1)+y1) / 2.0, ((x2-1)+x1) / 2.0))


def fillRndData(data, net):
    inputLayer = 'data'
    randomChannels = net.blobs[inputLayer].data.shape[1]
    rndData = np.random.randn(data.shape[0], randomChannels, data.shape[2], data.shape[3]).astype(np.float32) * 0.2
    rndData[:,0:1,:,:] = data
    net.blobs[inputLayer].data[...] = rndData[:,0:1,:,:]

def mkdirp(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        help="Model definition file.",
        required=True
    )
    parser.add_argument(
        "--pretrained_model",
        help="Trained model weights file.",
        required=True
    )
    parser.add_argument(
        "--out_scale",
        help="Scale of the output image.",
        default=1.0,
        type=float
    )
    parser.add_argument(
        "--output_path",
        help="Output path.",
        default=''
    )
    parser.add_argument(
        "--tile_resolution",
        help="Resolution of processing tile.",
        required=True,
        type=int
    )
    parser.add_argument(
        "--suffix",
        help="Suffix of the output file.",
        default="-deblur",
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--grey_mean",
        action='store_true',
        help="Use grey mean RGB=127. Default is the VGG mean."
    )
    parser.add_argument(
        "--use_mean",
        action='store_true',
        help="Use mean."
    )
    parser.add_argument(
        "--adversarial",
        action='store_true',
        help="Use mean."
    )
    args = parser.parse_args()

    mkdirp(args.output_path)

    if hasattr(caffe, 'set_mode_gpu'):
        if args.gpu:
            print('GPU mode', file=sys.stderr)
            caffe.set_mode_gpu()
        net = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)
    else:
        if args.gpu:
            print('GPU mode', file=sys.stderr)
        net = caffe.Net(args.model_def, args.pretrained_model, gpu=args.gpu)


    inputs = [line.strip() for line in sys.stdin]

    print("Classifying %d inputs." % len(inputs), file=sys.stderr)


    inputBlob = net.blobs.keys()[0]
    outputBlob = net.blobs.keys()[-1]

    print( inputBlob, outputBlob)
    channelCount = net.blobs[inputBlob].data.shape[1]
    net.blobs[inputBlob].reshape(1, channelCount, args.tile_resolution, args.tile_resolution)
    net.reshape()

    if channelCount == 1 or channelCount > 3:
        color = 0
    else:
        color = 1

    outResolution = net.blobs[outputBlob].data.shape[2]
    inResolution = int(outResolution / args.out_scale)
    boundary = (net.blobs[inputBlob].data.shape[2] - inResolution) / 2

    for fileName in inputs:
        img = cv2.imread(fileName, flags=color).astype(np.float32)
        original = np.copy(img)
        img = img.reshape(img.shape[0], img.shape[1], -1)
        if args.use_mean:
            if args.grey_mean or channelCount == 1:
                img -= 127
            else:
                img[:,:,0] -= 103.939
                img[:,:,1] -= 116.779
                img[:,:,2] -= 123.68
        img *= 0.004

        outShape = [int(img.shape[0] * args.out_scale) ,
                    int(img.shape[1] * args.out_scale) ,
                    net.blobs[outputBlob].channels]
        imgOut = np.zeros(outShape)

        imageStartTime = time.time()
        for x, xOut in zip(range(0, img.shape[0], inResolution), range(0, imgOut.shape[0], outResolution)):
            for y, yOut in zip(range(0, img.shape[1], inResolution), range(0, imgOut.shape[1], outResolution)):

                start = time.time()

                region = getCutout(img, x, y, x+inResolution, y+inResolution, boundary)
                region = region.reshape(region.shape[0], region.shape[1], -1)
                data = region.transpose([2, 0, 1]).reshape(1, -1, region.shape[0], region.shape[1])

                if args.adversarial:
                    fillRndData(data, net)
                    out = net.forward()
                else:
                    out = net.forward_all(data=data)

                out = out[outputBlob].reshape(out[outputBlob].shape[1], out[outputBlob].shape[2], out[outputBlob].shape[3]).transpose(1, 2, 0)

                if imgOut.shape[2] == 3 or imgOut.shape[2] == 1:
                    out /= 0.004
                    if args.use_mean:
                        if args.grey_mean:
                            out += 127
                        else:
                            out[:,:,0] += 103.939
                            out[:,:,1] += 116.779
                            out[:,:,2] += 123.68

                if out.shape[0] != outResolution:
                    print("Warning: size of net output is %d px and it is expected to be %d px" % (out.shape[0], outResolution))
                if out.shape[0] < outResolution:
                    print("Error: size of net output is %d px and it is expected to be %d px" % (out.shape[0], outResolution))
                    exit()

                xRange = min((outResolution, imgOut.shape[0] - xOut))
                yRange = min((outResolution, imgOut.shape[1] - yOut))

                imgOut[xOut:xOut+xRange, yOut:yOut+yRange, :] = out[0:xRange, 0:yRange, :]
                imgOut[xOut:xOut+xRange, yOut:yOut+yRange, :] = out[0:xRange, 0:yRange, :]

                print(".", end="", file=sys.stderr)
                sys.stdout.flush()


        print(imgOut.min(), imgOut.max())
        print("IMAGE DONE %s" % (time.time() - imageStartTime))
        basename = os.path.basename(fileName)
        name = os.path.join(args.output_path, basename + args.suffix)
        print(name, imgOut.shape)
        cv2.imwrite( name, imgOut)

if __name__ == '__main__':
    main(sys.argv)
