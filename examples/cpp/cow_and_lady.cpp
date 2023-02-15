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
#include <rosbag_reader/rosbag.h>
#include <vdbfusion/ImplicitRegistration.h>
#include <vdbfusion/VDBVolume.h>

#include <Eigen/Core>
#include <algorithm>
#include <argparse/argparse.hpp>
#include <execution>
#include <filesystem>
#include <fstream>
#include <string>

#include "utils/Config.h"
#include "utils/Iterable.h"
#include "utils/Timers.h"

// Namespace aliases
using namespace fmt::literals;
using namespace utils;
namespace fs = std::filesystem;

namespace {

std::vector<Eigen::Vector3d> vector2Eigen(const std::vector<std::vector<double>>& vec_mat) {
    std::vector<Eigen::Vector3d> scan(vec_mat.size());
    std::transform(
        std::execution::par_unseq, vec_mat.cbegin(), vec_mat.cend(), scan.begin(),
        [](const std::vector<double>& pt) { return Eigen::Vector3d(pt[0], pt[1], pt[2]); });
    // for (int i = 0; i < vec_mat.size(); i++) {
    //     scan.emplace_back(Eigen::Vector3d(vec_mat[i][0], vec_mat[i][1], vec_mat[i][2]));
    // }
    return std::move(scan);
}

argparse::ArgumentParser ArgParse(int argc, char* argv[]) {
    argparse::ArgumentParser argparser("CowAndLadyPipeline");
    argparser.add_argument("path_to_bag_file")
        .help("The full path to the Cow and Lady dataset rosbag");
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
                    const Sophus::SE3d& T,
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
            {
                auto grad_name =
                    fmt::format("{out_dir}/cow_and_lady/{n_scans}_scans_grad",
                                "out_dir"_a = argparser_.get<std::string>("mesh_output_dir"),
                                "n_scans"_a = n_iters + 1);
                timers::ScopeTimer timer("Writing VDB grid Gradient to disk");
                auto grad_grid = pipeline.ComputeGradient(pipeline.ClipVolume(T));
                std::string filename = fmt::format("{grad_name}.vdb", "grad_name"_a = grad_name);
                openvdb::io::File(filename).write({grad_grid});
            }
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

void PreProcessCloud(std::vector<Eigen::Vector3d>& points, float min_range, float max_range) {
    points.erase(
        std::remove_if(points.begin(), points.end(), [&](auto p) { return p.norm() > max_range; }),
        points.end());
    points.erase(
        std::remove_if(points.begin(), points.end(), [&](auto p) { return p.norm() < min_range; }),
        points.end());
}

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
    auto path_to_cow_and_lady_bag = fs::path(argparser.get<std::string>("path_to_bag_file"));

    // Initialize dataset
    auto rosbag = Rosbag(path_to_cow_and_lady_bag);
    rosbag.readData();

    int num_of_scans_on_topic = rosbag.getNumMsgsonTopic(cow_and_lady_cfg.topic_name_);
    if (n_scans == -1 || n_scans > num_of_scans_on_topic) {
        n_scans = num_of_scans_on_topic;
    }
    fmt::print("Integrating {} scans\n", n_scans);

    // Init VDB Volume
    vdbfusion::VDBVolume tsdf_volume(vdbfusion_cfg.voxel_size_, vdbfusion_cfg.sdf_trunc_,
                                     vdbfusion_cfg.space_carving_);

    // Init registration class
    vdbfusion::registrationConfigParams config;
    config.use_clipped_tsdf = true;
    config.max_iters_ = 500;
    config.convergence_threshold_ = 5e-5;
    config.clipping_range_ = cow_and_lady_cfg.max_range_;
    vdbfusion::ImplicitRegistration registration_pipeline(tsdf_volume, config);

    timers::FPSTimer<10> timer;
    DataSaver<50> datasaver(argparser);
    bool init_scan = true;
    Sophus::SE3d init_tf{};

    std::vector<Eigen::Matrix<double, 3, 4>> poses;
    poses.reserve(n_scans);

    float sigma = 50.0;
    float epsilon = -0.2;
    auto weghting_func = [&](float sdf) {
        // if (sdf > epsilon)
        //     return 1.0;
        // else if (sdf < -vdbfusion_cfg.sdf_trunc_)
        //     return 0.0;
        // else
        //     return std::exp(-sigma * std::pow(sdf - epsilon, 2));
        return 1.0;
    };

    for (int idx = 0; idx < n_scans; idx++) {
        if (idx < 100) continue;
        timer.tic();
        auto pcl = rosbag.extractPointCloud2(cow_and_lady_cfg.topic_name_, idx);
        auto scan = vector2Eigen(pcl.data);
        PreProcessCloud(scan, cow_and_lady_cfg.min_range_, cow_and_lady_cfg.max_range_);
        if (!init_scan) {
            auto [aligned_scan, T, n_iters] = registration_pipeline.AlignScan(scan, init_tf);
            std::cout << "idx: " << idx << "; niters = " << n_iters << "\n";
            tsdf_volume.Integrate(aligned_scan, T.matrix(), weghting_func);
            poses.emplace_back(T.matrix3x4());
            init_tf = T;
        } else {
            tsdf_volume.Integrate(scan, init_tf.matrix(), weghting_func);
            poses.emplace_back(init_tf.matrix3x4());
            init_scan = false;
        }
        datasaver(registration_pipeline, init_tf, vdbfusion_cfg.min_weight_);
        timer.toc();
    }

    std::filesystem::create_directory(fmt::format(
        "{out_dir}/cow_and_lady/", "out_dir"_a = argparser.get<std::string>("mesh_output_dir")));

    // Store the grid results to disks
    std::string map_name = fmt::format("{out_dir}/cow_and_lady/{n_scans}_scans",
                                       "out_dir"_a = argparser.get<std::string>("mesh_output_dir"),
                                       "n_scans"_a = n_scans);
    // {
    //     timers::ScopeTimer timer("Writing VDB grid to disk");
    //     auto tsdf_grid = tsdf_volume.tsdf_;
    //     std::string filename = fmt::format("{map_name}.vdb", "map_name"_a = map_name);
    //     openvdb::io::File(filename).write({tsdf_grid});
    // }

    // std::string grad_name = fmt::format("{out_dir}/cow_and_lady/{n_scans}_scans_grad",
    //                                     "out_dir"_a =
    //                                     argparser.get<std::string>("mesh_output_dir"),//
    //                                     "n_scans"_a = n_scans);
    // {
    //     timers::ScopeTimer timer("Writing VDB grid Gradient to disk");
    //     auto grad_grid = tsdf_volume.ComputeGradient(tsdf_volume.tsdf_);
    //     std::string filename = fmt::format("{grad_name}.vdb", "grad_name"_a = grad_name);
    //     openvdb::io::File(filename).write({grad_grid});
    // }

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
