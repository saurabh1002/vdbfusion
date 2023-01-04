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

#include <fmt/core.h>
#include <fmt/format.h>
#include <igl/write_triangle_mesh.h>
#include <openvdb/openvdb.h>
#include <vdbfusion/VDBVolume.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <string>

#include "datasets/KITTIOdometry.h"
#include "utils/Config.h"
#include "utils/Iterable.h"
#include "utils/Timers.h"

// Namespace aliases
using namespace fmt::literals;
using namespace utils;
namespace fs = std::filesystem;

namespace {

argparse::ArgumentParser ArgParse(int argc, char* argv[]) {
    argparse::ArgumentParser argparser("KITTIPipeline");
    argparser.add_argument("kitti_root_dir").help("The full path to the KITTI dataset");
    argparser.add_argument("mesh_output_dir").help("Directory to store the resultant mesh");
    argparser.add_argument("--sequence").help("KITTI Sequence");
    argparser.add_argument("--config")
        .help("Dataset specific config file")
        .default_value<std::string>("config/kitti.yaml")
        .action([](const std::string& value) { return value; });
    argparser.add_argument("--n_scans")
        .help("How many scans to map")
        .default_value(int(-1))
        .action([](const std::string& value) { return std::stoi(value); });

    try {
        argparser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << "Invalid Arguments " << std::endl;
        std::cerr << err.what() << std::endl;
        std::cerr << argparser;
        exit(0);
    }

    auto kitti_root_dir = argparser.get<std::string>("kitti_root_dir");
    if (!fs::exists(kitti_root_dir)) {
        std::cerr << kitti_root_dir << "path doesn't exists" << std::endl;
        exit(1);
    }
    return argparser;
}
}  // namespace

int main(int argc, char* argv[]) {
    auto argparser = ArgParse(argc, argv);

    // VDBVolume configuration
    auto vdbfusion_cfg =
        vdbfusion::VDBFusionConfig::LoadFromYAML(argparser.get<std::string>("--config"));
    // Dataset specific configuration
    auto kitti_cfg = datasets::KITTIConfig::LoadFromYAML(argparser.get<std::string>("--config"));

    openvdb::initialize();

    // Kitti stuff
    auto n_scans = argparser.get<int>("--n_scans");
    auto kitti_root_dir = fs::path(argparser.get<std::string>("kitti_root_dir"));
    auto sequence = argparser.get<std::string>("--sequence");

    // Initialize dataset
    const auto dataset =
        datasets::KITTIDataset(kitti_root_dir, sequence, n_scans, kitti_cfg.apply_pose_,
                               kitti_cfg.preprocess_, kitti_cfg.min_range_, kitti_cfg.max_range_);

    fmt::print("Integrating {} scans\n", dataset.size());
    vdbfusion::VDBVolume tsdf_volume(vdbfusion_cfg.voxel_size_, vdbfusion_cfg.sdf_trunc_,
                                     vdbfusion_cfg.space_carving_);
    timers::FPSTimer<10> timer;
    bool init_scan = true;
    Sophus::SE3d init_tf{};

    std::vector<Eigen::Matrix<double, 3, 4>> poses;
    poses.reserve(dataset.size());

    for (const auto& [scan, origin] : iterable(dataset)) {
        timer.tic();
        if (!init_scan) {
            auto [aligned_scan, T] = tsdf_volume.AlignScan(scan, init_tf);
            tsdf_volume.Integrate(aligned_scan, T.matrix(), [](float /*unused*/) { return 1.0; });
            poses.emplace_back(T.matrix3x4());
            // init_tf = T;
        } else {
            tsdf_volume.Integrate(scan, origin, [](float /*unused*/) { return 1.0; });
            poses.emplace_back(init_tf.matrix3x4());
            init_scan = false;
        }
        timer.toc();
    }

    std::filesystem::create_directory(fmt::format(
        "{out_dir}/kitti_odom_{seq}/", "out_dir"_a = argparser.get<std::string>("mesh_output_dir"),
        "seq"_a = sequence));

    // Store the grid results to disks
    std::string map_name = fmt::format("{out_dir}/kitti_odom_{seq}/{n_scans}_scans",
                                       "out_dir"_a = argparser.get<std::string>("mesh_output_dir"),
                                       "seq"_a = sequence, "n_scans"_a = n_scans);
    {
        timers::ScopeTimer timer("Writing VDB grid to disk");
        auto tsdf_grid = tsdf_volume.tsdf_;
        std::string filename = fmt::format("{map_name}.vdb", "map_name"_a = map_name);
        openvdb::io::File(filename).write({tsdf_grid});
    }

    std::string grad_name = fmt::format("{out_dir}/kitti_odom_{seq}/{n_scans}_scans_grad",
                                        "out_dir"_a = argparser.get<std::string>("mesh_output_dir"),
                                        "seq"_a = sequence, "n_scans"_a = n_scans);
    {
        timers::ScopeTimer timer("Writing VDB grid Gradient to disk");
        auto grad_grid = tsdf_volume.ComputeGradient();
        std::string filename = fmt::format("{grad_name}.vdb", "grad_name"_a = grad_name);
        openvdb::io::File(filename).write({grad_grid});
    }

    std::string pose_name = fmt::format("{out_dir}/kitti_odom_{seq}/{seq}",
                                        "out_dir"_a = argparser.get<std::string>("mesh_output_dir"),
                                        "seq"_a = sequence);
    {
        timers::ScopeTimer timer("Writing Poses to disk");
        std::string filename = fmt::format("{pose_name}.txt", "pose_name"_a = pose_name);
        std::ofstream pose_file(filename);
        std::for_each(poses.cbegin(), poses.cend(),
                      [&pose_file](const Eigen::Matrix<double, 3, 4>& pose) {
                          pose_file << pose(0, 0) << " " << pose(0, 1) << " " << pose(0, 2) << " "
                                    << pose(0, 3) << " " << pose(1, 0) << " " << pose(1, 1) << " "
                                    << pose(1, 2) << " " << pose(1, 3) << " " << pose(2, 0) << " "
                                    << pose(2, 1) << " " << pose(2, 2) << " " << pose(2, 3) << "\n";
                      });
        pose_file.close();
    }

    // Run marching cubes and save a .ply file
    {
        timers::ScopeTimer timer("Writing Mesh to disk");
        auto [vertices, triangles] =
            tsdf_volume.ExtractTriangleMesh(vdbfusion_cfg.fill_holes_, vdbfusion_cfg.min_weight_);

        // TODO: Fix this!
        Eigen::MatrixXd V(vertices.size(), 3);
        for (size_t i = 0; i < vertices.size(); i++) {
            V.row(i) = Eigen::VectorXd::Map(&vertices[i][0], vertices[i].size());
        }

        // TODO: Also this
        Eigen::MatrixXi F(triangles.size(), 3);
        for (size_t i = 0; i < triangles.size(); i++) {
            F.row(i) = Eigen::VectorXi::Map(&triangles[i][0], triangles[i].size());
        }
        std::string filename = fmt::format("{map_name}.ply", "map_name"_a = map_name);
        igl::write_triangle_mesh(filename, V, F, igl::FileEncoding::Binary);
    }

    return 0;
}
