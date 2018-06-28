// protoc lc0net.proto --cpp_out=build
// clang++-6.0 txt2proto.cc build/lc0net.pb.cc $(pkg-config --cflags --libs protobuf) -std=c++14 -Wall -Wextra -O3 -march=native
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <exception>
#include <vector>

#include "build/lc0net.pb.h"

class Net {
  using FloatVector = std::vector<float>;
  using FloatVectors = std::vector<FloatVector>;

  public:
    Net() {
    }
    ~Net() {}

    void FromTxtFile(const std::string &filename) {
      std::ifstream in(filename.c_str());
      std::string buffer, line;

      // strip version
      std::getline(in, buffer);
      int version = std::stoi(buffer);
      std::cout << "version: " << version << std::endl;
      w_.set_version(version);

      // parse network data
      FloatVectors weights;

      while (in) {
        FloatVector weight_line;
        std::getline(in, line);
        std::stringstream ss(line);
        while (std::getline(ss, buffer, ' ')) {
          float v = std::stof(buffer);
          auto x = reinterpret_cast<std::uint32_t*>(&v);
          (*x) &= 0xffff0000;
          weight_line.emplace_back(v);
        }
        weights.emplace_back(weight_line);
      }

      for (auto v : weights.back()) { w_.add_ip2_val_b(v); } weights.pop_back();
      for (auto v : weights.back()) { w_.add_ip2_val_w(v); } weights.pop_back();
      for (auto v : weights.back()) { w_.add_ip1_val_b(v); } weights.pop_back();
      for (auto v : weights.back()) { w_.add_ip1_val_w(v); } weights.pop_back();

      w_.set_allocated_value(FillConvBlock(weights));
      for (auto v : weights.back()) { w_.add_ip_pol_b(v); } weights.pop_back();
      for (auto v : weights.back()) { w_.add_ip_pol_w(v); } weights.pop_back();
      w_.set_allocated_policy(FillConvBlock(weights));

      const int num_residual = (weights.size() - 5) / 8;
      if ((weights.size() - 5) % 8 != 0) {
        throw std::runtime_error("Bad input file");
      }

      std::vector<lc0::Weights_Residual*> tower(num_residual);
      for (int i = 0; i < num_residual; i++) { tower[i] = w_.add_residual(); }
      for (int i = num_residual - 1; i >= 0; i--) {
        tower[i]->set_allocated_conv2(FillConvBlock(weights));
        tower[i]->set_allocated_conv1(FillConvBlock(weights));
      }
      w_.set_allocated_input(FillConvBlock(weights));
    }

    void Save(const std::string &filename) {
      std::string s;
      std::ofstream output(filename.c_str(), std::ios::binary | std::ios::ate);
      w_.SerializeToString(&s);
      output.write(s.c_str(), s.size());
      output.close();
      std::cout << "written to " << filename << std::endl;
    }

  private:
    lc0::Weights w_;

    lc0::Weights_ConvBlock* FillConvBlock(FloatVectors &weights) {
      lc0::Weights_ConvBlock *block = new lc0::Weights_ConvBlock();
      for (auto v : weights.back()) { block->add_bn_stddivs(v); } weights.pop_back();
      for (auto v : weights.back()) { block->add_bn_means(v); } weights.pop_back();
      for (auto v : weights.back()) { block->add_biases(v); } weights.pop_back();
      for (auto v : weights.back()) { block->add_weights(v); } weights.pop_back();
      return block;
    }
};

int main(int argc, char **argv) {
  if (argc != 3) {
    return 1;
  }

  Net net;
  net.FromTxtFile(argv[1]);
  net.Save(argv[2]);

  return 0;
}
