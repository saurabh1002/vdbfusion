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
#include <vdbfusion/VDBVolume.h>

#include <Eigen/Core>
#include <algorithm>
#include <argparse/argparse.hpp>
#include <execution>
#include <filesystem>
#include <fstream>
#include <string>

#include "datasets/CowAndLady.h"
#include "utils/Config.h"
#include "utils/Iterable.h"
#include "utils/Timers.h"

// Namespace aliases
using namespace fmt::literals;
using namespace utils;
namespace fs = std::filesystem;

namespace {

argparse::ArgumentParser ArgParse(int argc, char* argv[]) {
    argparse::ArgumentParser argparser("CowAndLadyPipeline");
    argparser.add_argument("cow_and_lady_dir")
        .help("The full path to the Cow and Lady dataset directory");
    argparser.add_argument("mesh_output_dir").help("Directory to store the resultant mesh");
    argparser.add_argument("--config")
        .help("Dataset specific config file")
        .default_value<std::string>("../examples/cpp/config/cow_and_lady.yaml")
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
    return argparser;
}
}  // namespace

template <int N = 10>
class DataSaver {
public:
    explicit DataSaver(const argparse::ArgumentParser& argparser) : argparser_(argparser) {
        std::filesystem::create_directory(
            fmt::format("{out_dir}/cow_and_lady/",
                        "out_dir"_a = argparser_.get<std::string>("mesh_output_dir")));
    };

public:
    void operator()(vdbfusion::ImplicitRegistration& pipeline,
                    const Eigen::Matrix4d& T,
                    float min_weight) {
        auto map_name = fmt::format("{out_dir}/cow_and_lady/{n_scans}_scans",
                                    "out_dir"_a = argparser_.get<std::string>("mesh_output_dir"),
                                    "n_scans"_a = n_iters + 1);
        if ((n_iters + 1) % N == 0) {
            {
                timers::ScopeTimer timer("Writing VDB grid to disk");
                std::string filename = fmt::format("{map_name}.vdb", "map_name"_a = map_name);
                openvdb::io::File(filename).write({pipeline.vdb_volume_global_.tsdf_});
            }
            // {
            //     auto grad_name =
            //         fmt::format("{out_dir}/cow_and_lady/{n_scans}_scans_grad",
            //                     "out_dir"_a = argparser_.get<std::string>("mesh_output_dir"),
            //                     "n_scans"_a = n_iters + 1);
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
};

int main(int argc, char* argv[]) {
    auto argparser = ArgParse(argc, argv);

    // VDBVolume configuration
    auto vdbfusion_cfg =
        vdbfusion::VDBFusionConfig::LoadFromYAML(argparser.get<std::string>("--config"));

    // Dataset specific configuration
    auto cow_and_lady_cfg =
        datasets::CowAndLadyConfig::LoadFromYAML(argparser.get<std::string>("--config"));

    openvdb::initialize();

    auto n_scans = argparser.get<int>("--n_scans");
    auto cow_root_dir = fs::path(argparser.get<std::string>("cow_and_lady_dir"));

    // Initialize dataset
    const auto dataset =
        datasets::CowAndLadyDataset(cow_root_dir, n_scans, cow_and_lady_cfg.preprocess_,
                                    cow_and_lady_cfg.min_range_, cow_and_lady_cfg.max_range_);

    fmt::print("Integrating {} scans\n", n_scans);

    // Init VDB Volume
    vdbfusion::VDBVolume tsdf_volume(vdbfusion_cfg.voxel_size_, vdbfusion_cfg.sdf_trunc_,
                                     vdbfusion_cfg.space_carving_);

    // Init registration class
    vdbfusion::registrationConfigParams config;
    config.max_iters_ = 500;
    config.convergence_threshold_ = 5e-5;
    config.clipping_range_ = cow_and_lady_cfg.max_range_;

    vdbfusion::ImplicitRegistration registration_pipeline(tsdf_volume, config);

    timers::FPSTimer<10> timer;
    DataSaver<50> datasaver(argparser);
    bool init_scan = true;
    Eigen::Matrix4d init_tf{};
    init_tf.setIdentity();

    std::vector<Eigen::Matrix<double, 3, 4>> poses;
    poses.reserve(n_scans);

    int start_idx = 100;
    for (int idx = start_idx; idx < n_scans; idx++) {
        timer.tic();
        if (!init_scan) {
            auto [aligned_scan, T, n_iters] =
                registration_pipeline.AlignScan(dataset[idx], init_tf);
            tsdf_volume.Integrate(aligned_scan, T, [](float) { return 1.0; });
            poses.emplace_back(T.block<3, 4>(0, 0));
            init_tf = T;

            std::cout << "scan idx: " << idx << "\t iters = " << n_iters << "\n";
        } else {
            tsdf_volume.Integrate(dataset[idx], init_tf, [](float) { return 1.0; });
            poses.emplace_back(init_tf.block<3, 4>(0, 0));
            init_scan = false;
        }
        datasaver(registration_pipeline, init_tf, vdbfusion_cfg.min_weight_);
        timer.toc();
    }

    std::filesystem::create_directory(fmt::format(
        "{out_dir}/cow_and_lady/", "out_dir"_a = argparser.get<std::string>("mesh_output_dir")));

    // Store the grid results to disks
    std::string pose_name = fmt::format(
        "{out_dir}/cow_and_lady/", "out_dir"_a = argparser.get<std::string>("mesh_output_dir"));
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

    std::string map_name = fmt::format("{out_dir}/cow_and_lady/{n_scans}_scans",
                                       "out_dir"_a = argparser.get<std::string>("mesh_output_dir"),
                                       "n_scans"_a = n_scans);

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
