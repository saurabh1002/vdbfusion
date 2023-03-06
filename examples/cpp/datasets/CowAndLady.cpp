// MIT License
//
// # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "CowAndLady.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::vector<std::string> GetVelodyneFiles(const fs::path& velodyne_path, int n_scans) {
    std::vector<std::string> velodyne_files;
    for (const auto& entry : fs::directory_iterator(velodyne_path)) {
        if (entry.path().extension() == ".bin") {
            velodyne_files.emplace_back(entry.path().string());
        }
    }
    if (velodyne_files.empty()) {
        std::cerr << velodyne_path << "path doesn't have any .bin" << std::endl;
        exit(1);
    }
    std::sort(velodyne_files.begin(), velodyne_files.end());
    if (n_scans > 0) {
        velodyne_files.erase(velodyne_files.begin() + n_scans, velodyne_files.end());
    }
    return velodyne_files;
}

std::vector<Eigen::Vector3d> ReadKITTIVelodyne(const std::string& path) {
    std::ifstream scan_input(path.c_str(), std::ios::binary);
    assert(scan_input.is_open() && "ReadPointCloud| not able to open file");

    scan_input.seekg(0, std::ios::end);
    uint32_t num_points = scan_input.tellg() / (3 * sizeof(double));
    scan_input.seekg(0, std::ios::beg);

    std::vector<double> values(3 * num_points);
    scan_input.read((char*)&values[0], 3 * num_points * sizeof(double));
    scan_input.close();

    std::vector<Eigen::Vector3d> points;
    points.resize(num_points);
    for (uint32_t i = 0; i < num_points; i++) {
        points[i].x() = values[i * 3];
        points[i].y() = values[i * 3 + 1];
        points[i].z() = values[i * 3 + 2];
    }
    return points;
}

void PreProcessCloud(std::vector<Eigen::Vector3d>& points, float min_range, float max_range) {
    points.erase(
        std::remove_if(points.begin(), points.end(), [&](auto p) { return p.norm() > max_range; }),
        points.end());
    points.erase(
        std::remove_if(points.begin(), points.end(), [&](auto p) { return p.norm() < min_range; }),
        points.end());
}
}  // namespace

namespace datasets {

CowAndLadyDataset::CowAndLadyDataset(const std::string& cow_root_dir, int n_scans) {
    // TODO: to be completed
    auto cow_root_dir_ = fs::absolute(fs::path(cow_root_dir));

    // Read data, cache it inside the class.
    scan_files_ = GetVelodyneFiles(fs::absolute(cow_root_dir_ / "binary/"), n_scans);
}

CowAndLadyDataset::CowAndLadyDataset(
    const std::string& cow_root_dir, int n_scans, bool preprocess, float min_range, float max_range)
    : preprocess_(preprocess), min_range_(min_range), max_range_(max_range) {
    auto cow_root_dir_ = fs::absolute(fs::path(cow_root_dir));

    // Read data, cache it inside the class.
    scan_files_ = GetVelodyneFiles(fs::absolute(cow_root_dir_ / "binary/"), n_scans);
}

std::vector<Eigen::Vector3d> CowAndLadyDataset::operator[](int idx) const {
    std::vector<Eigen::Vector3d> points = ReadKITTIVelodyne(scan_files_[idx]);
    if (preprocess_) PreProcessCloud(points, min_range_, max_range_);
    return points;
}
}  // namespace datasets
