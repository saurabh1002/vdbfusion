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
#include <vdbfusion/ImplicitRegistration.h>
#include <vdbfusion/Preprocessing.h>
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
using PointCloud = std::vector<Eigen::Vector3d>;

namespace fs = std::filesystem;

namespace {

argparse::ArgumentParser ArgParse(int argc, char* argv[]) {
    argparse::ArgumentParser argparser("KITTIPipeline");
    argparser.add_argument("kitti_root_dir").help("The full path to the KITTI dataset");
    argparser.add_argument("mesh_output_dir").help("Directory to store the resultant mesh");
    argparser.add_argument("--sequence").help("KITTI Sequence");
    argparser.add_argument("--config")
        .help("Dataset specific config file")
        .default_value<std::string>("../examples/cpp/config/kitti.yaml")
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

template <int N = 10>
class DataSaver {
public:
    explicit DataSaver(const argparse::ArgumentParser& argparser, const std::string& sequence)
        : argparser_(argparser), sequence_(sequence) {
        std::filesystem::create_directory(fmt::format(
            "{out_dir}/kitti_odom_{seq}/",
            "out_dir"_a = argparser_.get<std::string>("mesh_output_dir"), "seq"_a = sequence_));
    };

public:
    void operator()(vdbfusion::ImplicitRegistration& pipeline,
                    const Eigen::Matrix4d& T,
                    float min_weight) {
        auto map_name = fmt::format("{out_dir}/kitti_odom_{seq}/{n_scans}_scans",
                                    "out_dir"_a = argparser_.get<std::string>("mesh_output_dir"),
                                    "seq"_a = sequence_, "n_scans"_a = n_iters + 1);
        if ((n_iters + 1) % N == 0) {
            {
                timers::ScopeTimer timer("Writing VDB grid to disk");
                std::string filename = fmt::format("{map_name}.vdb", "map_name"_a = map_name);
                openvdb::io::File(filename).write({pipeline.vdb_volume_global_.tsdf_});
            }
            // {
            //     auto grad_name =
            //         fmt::format("{out_dir}/kitti_odom_{seq}/{n_scans}_scans_grad",
            //                     "out_dir"_a = argparser_.get<std::string>("mesh_output_dir"),
            //                     "seq"_a = sequence_, "n_scans"_a = n_iters + 1);
            //     timers::ScopeTimer timer("Writing VDB grid Gradient to disk");
            //     auto grad_grid = pipeline.ComputeGradient(pipeline.vdb_volume_global_.tsdf_, T);
            //     std::string filename = fmt::format("{grad_name}.vdb", "grad_name"_a = grad_name);
            //     openvdb::io::File(filename).write({grad_grid});
            // }
            {
                timers::ScopeTimer timer("Writing Mesh to disk");
                auto [vertices, triangles] =
                    pipeline.vdb_volume_global_.ExtractTriangleMesh(true, min_weight);

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
        }
        n_iters++;
    }

private:
    int n_iters = 0;
    argparse::ArgumentParser argparser_;
    std::string sequence_;
};

std::vector<Eigen::Vector3d> TransformPoints(const std::vector<Eigen::Vector3d>& points,
                                             const Eigen::Matrix4d& transformation) {
    std::vector<Eigen::Vector3d> points_new;
    for (auto& point : points) {
        Eigen::Vector4d new_point =
            transformation * Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        points_new.emplace_back(new_point.head<3>() / new_point(3));
    }
    return points_new;
}

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
    const auto dataset = datasets::KITTIDataset(kitti_root_dir, sequence, n_scans);
    fmt::print("Integrating {} scans\n", dataset.size());

    // Init VDB Volume
    vdbfusion::VDBVolume tsdf_volume(vdbfusion_cfg.voxel_size_, vdbfusion_cfg.sdf_trunc_,
                                     vdbfusion_cfg.space_carving_);

    // Init registration class
    vdbfusion::registrationConfigParams config;
    config.max_iters_ = 2000;
    config.convergence_threshold_ = 5e-3;
    config.clipping_range_ = kitti_cfg.max_range_;
    vdbfusion::ImplicitRegistration registration_pipeline(tsdf_volume, config);

    timers::FPSTimer<50> timer;
    DataSaver<10> datasaver(argparser, sequence);
    bool init_scan = true;
    Eigen::Matrix4d init_tf{};
    init_tf.setIdentity();

    std::vector<Eigen::Matrix<double, 3, 4>> poses;
    poses.reserve(dataset.size());

    int count = 50;
    int scan_nr = 0;
    for (const auto& [_, scan, pose] : iterable(dataset)) {
        auto scan_p = vdbfusion::CorrectKITTIScan(
            vdbfusion::Preprocess(scan, kitti_cfg.max_range_, kitti_cfg.min_range_));
        timer.tic();
        if (!init_scan) {
            auto [aligned_scan, T, n_iters] = registration_pipeline.AlignScan(scan_p, init_tf);
            tsdf_volume.Integrate(aligned_scan, T, [](float) { return 1.0; });
            poses.emplace_back(T.block<3, 4>(0, 0));
            init_tf = T;

            std::cout << "difference pose: " << (pose - T).norm() << "\n";
            std::cout << "scan idx: " << scan_nr << "\t iters = " << n_iters << "\n";
        } else {
            tsdf_volume.Integrate(TransformPoints(scan_p, pose), pose,
                                  [](float /*unused*/) { return 1.0; });
            poses.emplace_back(pose.block<3, 4>(0, 0));

            count--;
            if (count == 0) init_scan = false;
            init_tf = Sophus::SE3d(Sophus::makeRotationMatrix(pose.block<3, 3>(0, 0)),
                                   pose.block<3, 1>(0, 3))
                          .matrix();
        }
        scan_nr++;
        datasaver(registration_pipeline, init_tf, vdbfusion_cfg.min_weight_);
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
