#!/usr/bin/env python

import argparse
import gzip
import os
import numpy as np
import lc0net_pb2 as pb


class Net:
    def __init__(self):
        self.params = pb.Net()
        self.weights = []


    def fill_layer(self, layer, weights):
        """Normalize and populate 8bit layer in protobuf"""
        params = np.array(weights.pop(), dtype=np.float32)
        layer.min_val = 0 if len(params) == 1 else np.min(params)
        layer.max_val = np.max(params)
        params = (params - layer.min_val) / (layer.max_val - layer.min_val)
        params *= 2**16-1
        params = np.round(params)
        layer.params = params.astype(np.uint16).tobytes()


    def fill_conv_block(self, convblock, weights):
        """Normalize and populate 8bit convblock in protobuf"""
        self.fill_layer(convblock.weights, weights)
        self.fill_layer(convblock.biases, weights)
        self.fill_layer(convblock.bn_means, weights)
        self.fill_layer(convblock.bn_stddivs, weights)


    def denorm_layer(self, layer, weights):
        """Denormalize a layer from protobuf"""
        params = np.frombuffer(layer.params, np.uint16).astype(np.float32)
        params /= 2**16-1
        weights.insert(0, params * (layer.max_val - layer.min_val) + layer.min_val)


    def denorm_conv_block(self, convblock, weights):
        """Denormalize a convblock from protobuf"""
        self.denorm_layer(convblock.weights, weights)
        self.denorm_layer(convblock.biases, weights)
        self.denorm_layer(convblock.bn_means, weights)
        self.denorm_layer(convblock.bn_stddivs, weights)


    def save_txt(self, filename):
        """Save weights as txt file"""
        weights = self.get_weights()

        if len(filename.split('.')) == 1:
            filename += ".txt.gz"

        with gzip.open(filename, 'wb') as f:
            f.write("{}\n".format(self.params.version))
            for row in weights:
                f.write(" ".join(map(str, row.tolist())) + "\n")

        size = os.path.getsize(filename) * 1e-6
        print("saved as '{}' {}M".format(filename, round(size, 2)))


    def save_proto(self, filename):
        """Save weights gzipped protobuf file"""
        if len(filename.split('.')) == 1:
            filename += ".pb.gz"

        with gzip.open(filename, 'wb') as f:
            data = self.params.SerializeToString()
            f.write(data)

        size = os.path.getsize(filename) * 1e-6
        print("saved as '{}' {}M".format(filename, round(size, 2)))



    def get_weights(self):
        """Returns the weights as floats per layer"""
        if self.weights == []:
            self.denorm_layer(self.params.ip2_val_b, self.weights)
            self.denorm_layer(self.params.ip2_val_w, self.weights)
            self.denorm_layer(self.params.ip1_val_b, self.weights)
            self.denorm_layer(self.params.ip1_val_w, self.weights)
            self.denorm_conv_block(self.params.value, self.weights)

            self.denorm_layer(self.params.ip_pol_b, self.weights)
            self.denorm_layer(self.params.ip_pol_w, self.weights)
            self.denorm_conv_block(self.params.policy, self.weights)

            for res in reversed(self.params.residual):
                self.denorm_conv_block(res.conv2, self.weights)
                self.denorm_conv_block(res.conv1, self.weights)

            self.denorm_conv_block(self.params.input, self.weights)
            
        return self.weights


    def parse_proto(self, filename):
        with gzip.open(filename, 'rb') as f:
            self.params = self.params.FromString(f.read())


    def parse_txt(self, filename):
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

        self.params.version = version
        self.fill_layer(self.params.ip2_val_b, weights)
        self.fill_layer(self.params.ip2_val_w, weights)
        self.fill_layer(self.params.ip1_val_b, weights)
        self.fill_layer(self.params.ip1_val_w, weights)
        self.fill_conv_block(self.params.value, weights)

        self.fill_layer(self.params.ip_pol_b, weights)
        self.fill_layer(self.params.ip_pol_w, weights)
        self.fill_conv_block(self.params.policy, weights)

        tower = []
        for i in range(blocks):
            tower.append(self.params.residual.add())

        for res in reversed(tower):
            self.fill_conv_block(res.conv2, weights)
            self.fill_conv_block(res.conv1, weights)

        self.fill_conv_block(self.params.input, weights)


def main(argv):
    net = Net()

    if argv.input.endswith(".txt"):
        net.parse_txt(argv.input)
        net.save_txt(argv.output)
        net.save_proto(argv.output)
    elif argv.input.endswith(".pb.gz"):
        net.parse_proto(argv.input)
        net.save_txt(argv.output)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
    'Convert network textfile to proto.')
    argparser.add_argument('-i', '--input', type=str, 
        help='input network weight text file')
    argparser.add_argument('-o', '--output', type=str, 
        help='output filepath without extension')
    main(argparser.parse_args())
