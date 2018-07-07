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


def fill_layer(layer, weights):
    params = np.array(weights.pop(), dtype=np.float32)
    layer.min_val = 0 if len(params) == 1 else np.min(params)
    layer.max_val = np.max(params)
    params = (params - layer.min_val) / (layer.max_val - layer.min_val)
    params *= 255
    layer.params = np.round(params).astype(np.uint8).tobytes()


def fill_conv_block(convblock, weights):
    fill_layer(convblock.weights, weights)
    fill_layer(convblock.biases, weights)
    fill_layer(convblock.bn_means, weights)
    fill_layer(convblock.bn_stddivs, weights)


def main(argv):
    weights, version, filters, blocks = parse(argv.input)

    net = pb.Net()
    net.version = version

    fill_layer(net.ip2_val_b, weights)
    fill_layer(net.ip2_val_w, weights)
    fill_layer(net.ip1_val_b, weights)
    fill_layer(net.ip1_val_w, weights)
    fill_conv_block(net.value, weights)

    fill_layer(net.ip_pol_b, weights)
    fill_layer(net.ip_pol_w, weights)
    fill_conv_block(net.policy, weights)

    tower = []
    for i in range(blocks):
        tower.append(net.residual.add())

    for res in reversed(tower):
        fill_conv_block(res.conv2, weights)
        fill_conv_block(res.conv1, weights)

    fill_conv_block(net.input, weights)

    filename = argv.output + ".pb.gz"
    with gzip.open(filename, 'wb') as f:
        data = net.SerializeToString()
        f.write(data)

    size = os.path.getsize(filename) * 1e-6
    print("saved {}x{} v{} as '{}' {}M".format(filters, blocks, version, filename, round(size, 2)))
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
    'Convert network textfile to proto.')
    argparser.add_argument('-i', '--input', type=str, 
        help='input network weight text file')
    argparser.add_argument('-o', '--output', type=str, 
        help='output filepath without extension')
    main(argparser.parse_args())
