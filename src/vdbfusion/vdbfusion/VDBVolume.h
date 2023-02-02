/*
 * MIT License
 *
 * # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>

#include <Eigen/Core>
#include <functional>
#include <tuple>

#include "sophus/se3.hpp"

namespace vdbfusion {

class VDBVolume {
public:
    VDBVolume(float voxel_size, float sdf_trunc, bool space_carving = false);
    ~VDBVolume() = default;

public:
    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume.
    void Integrate(const std::vector<Eigen::Vector3d>& points,
                   const Eigen::Vector3d& origin,
                   const std::function<float(float)>& weighting_function);

    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume.
    void inline Integrate(const std::vector<Eigen::Vector3d>& points,
                          const Eigen::Matrix4d& extrinsics,
                          const std::function<float(float)>& weighting_function) {
        const Eigen::Vector3d& origin = extrinsics.block<3, 1>(0, 3);
        Integrate(points, origin, weighting_function);
    }

    /// @brief Integrate incoming TSDF grid inside the current volume using the TSDF equations
    void Integrate(openvdb::FloatGrid::Ptr grid,
                   const std::function<float(float)>& weighting_function);

    /// @brief Fuse a new given sdf value at the given voxel location, thread-safe
    void UpdateTSDF(const float& sdf,
                    const openvdb::Coord& voxel,
                    const std::function<float(float)>& weighting_function);

    /// @brief Prune TSDF grids, ideal utility to cleanup a D(x) volume before exporting it
    openvdb::FloatGrid::Ptr Prune(float min_weight) const;

    /// @brief Extracts a TriangleMesh as the iso-surface in the actual volume
    [[nodiscard]] std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3i>>
    ExtractTriangleMesh(bool fill_holes = true, float min_weight = 0.5) const;

public:
    /// OpenVDB Grids modeling the signed distance field and the weight grid
    openvdb::FloatGrid::Ptr tsdf_;
    openvdb::FloatGrid::Ptr weights_;
    // openvdb::v9_0::tools::ScalarToVectorConverter<openvdb::FloatGrid>::Type::Ptr gradients_;

    /// VDBVolume public properties
    float voxel_size_;
    float sdf_trunc_;
    bool space_carving_;
};

struct registrationConfigParams {
    bool use_constant_velocity_model_;
    bool use_clipped_tsdf;
    int max_iters_;
    double convergence_threshold_;
    double clipping_range_;
};

class ImplicitRegistration {
public:
    ImplicitRegistration(VDBVolume& vdb_volume_global, const registrationConfigParams& config);
    ~ImplicitRegistration() = default;

public:
    Sophus::SE3d ConstantVelocityModel() const;

    /// @brief Compute the Gradients of the Signed Distance Field at each voxel location
    openvdb::tools::ScalarToVectorConverter<openvdb::FloatGrid>::Type::Ptr ComputeGradient(
        const openvdb::FloatGrid::Ptr grid) const;

    openvdb::FloatGrid::Ptr ClipVolume(const Sophus::SE3d& T) const;

    std::tuple<std::vector<Eigen::Vector3d>, Sophus::SE3d, int> AlignScan(
        const std::vector<Eigen::Vector3d>& points, const Sophus::SE3d& init_tf);

    void RMSError(const std::vector<Eigen::Vector3d>& points);

public:
    VDBVolume vdb_volume_global_;
    registrationConfigParams config_;
    Sophus::SE3d T_1{};
    Sophus::SE3d T_2{};
};

}  // namespace vdbfusion
