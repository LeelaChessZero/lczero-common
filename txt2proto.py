#!/usr/bin/env python

import argparse
import gzip
import os
import numpy as np
import lc0net_pb2 as pb


def parse(filename):
    weights = []

    with open(filename, 'r') as f:
        version = int(f.readline())

        for e, line in enumerate(f):
            weights.append(list(map(float, line.split(' '))))

            if e == 1:
                filters = len(line.split(' '))

        blocks = e - (3 + 14)

        if blocks % 8 != 0:
            raise ValueError("Inconsistent number of weights in the file")

        blocks //= 8

    return weights, version, filters, blocks


def fill_conv_block(cb, w8b):
    cb.weights = w8b.pop().tobytes()
    cb.biases = w8b.pop().tobytes()
    cb.bn_means = w8b.pop().tobytes()
    cb.bn_stddivs = w8b.pop().tobytes()


def main(argv):
    weights, version, filters, blocks = parse(argv.input)

    w = pb.Weights()
    w.version = version
    flat = np.hstack(weights)
    w.min_val = np.min(flat)
    w.max_val = np.max(flat)
    print(w.min_val, w.max_val)

    w8b = []
    n = 0
    for wrow in weights:
        n += len(wrow)
        norm = (np.array(wrow) - w.min_val) / (w.max_val - w.min_val)
        norm *= 255
        norm = np.round(norm)
        w8b.append(norm.astype(np.uint8))

    w.ip2_val_b = w8b.pop().tobytes()
    w.ip2_val_w = w8b.pop().tobytes()
    w.ip1_val_b = w8b.pop().tobytes()
    w.ip1_val_w = w8b.pop().tobytes()
    fill_conv_block(w.value, w8b)

    w.ip_pol_b = w8b.pop().tobytes()
    w.ip_pol_w = w8b.pop().tobytes()
    fill_conv_block(w.policy, w8b)

    tower = []
    for i in range(blocks):
        tower.append(w.residual.add())

    for res in reversed(tower):
        fill_conv_block(res.conv2, w8b)
        fill_conv_block(res.conv1, w8b)

    fill_conv_block(w.input, w8b)

    filename = argv.output + ".pb.gz"
    with gzip.open(filename, 'wb') as f:
        data = w.SerializeToString()
        f.write(data)

	size = os.path.getsize(filename) / 1024
    print("saved {}x{} v{} as '{}' {}W {}K".format(filters, blocks, version, filename, n, round(size)))
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
    'Convert network textfile to proto.')
    argparser.add_argument('-i', '--input', type=str, 
        help='input network weight text file')
    argparser.add_argument('-o', '--output', type=str, 
        help='output filepath without extension')
    main(argparser.parse_args())
