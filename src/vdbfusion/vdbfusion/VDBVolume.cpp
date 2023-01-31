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

#include "VDBVolume.h"

// OpenVDB
#include <openvdb/Types.h>
#include <openvdb/math/BBox.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Clip.h>
#include <openvdb/tools/GridOperators.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "utils/conversions.h"

typedef Eigen::Matrix<double, 3, 6> Matrix3x6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6x6d;
typedef Eigen::Matrix<double, 6, 1> Matrix6x1d;

namespace {
float ComputeSDF(const Eigen::Vector3d& origin,
                 const Eigen::Vector3d& point,
                 const Eigen::Vector3d& voxel_center) {
    const Eigen::Vector3d v_voxel_origin = voxel_center - origin;
    const Eigen::Vector3d v_point_voxel = point - voxel_center;
    const double dist = v_point_voxel.norm();
    const double proj = v_voxel_origin.dot(v_point_voxel);
    const double sign = proj / std::abs(proj);
    return static_cast<float>(sign * dist);
}

Eigen::Vector3d GetVoxelCenter(const openvdb::Coord& voxel, const openvdb::math::Transform& xform) {
    const float voxel_size = xform.voxelSize()[0];
    openvdb::math::Vec3d v_wf = xform.indexToWorld(voxel) + voxel_size / 2.0;
    return Eigen::Vector3d(v_wf.x(), v_wf.y(), v_wf.z());
}

std::vector<Eigen::Vector3d> ApplyTransform(const std::vector<Eigen::Vector3d>& pcl,
                                            const Sophus::SE3d& T) {
    auto R = T.rotationMatrix();
    auto tr = T.translation();

    std::vector<Eigen::Vector3d> pcl_t;
    pcl_t.reserve(pcl.size());

    std::for_each(pcl.cbegin(), pcl.cend(),
                  [&](const Eigen::Vector3d& pt) { pcl_t.emplace_back((R * pt) + tr); });
    return pcl_t;
}

}  // namespace

namespace vdbfusion {

VDBVolume::VDBVolume(float voxel_size, float sdf_trunc, bool space_carving /* = false*/)
    : voxel_size_(voxel_size), sdf_trunc_(sdf_trunc), space_carving_(space_carving) {
    tsdf_ = openvdb::FloatGrid::create(sdf_trunc_);
    tsdf_->setName("D(x): signed distance grid");
    tsdf_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    tsdf_->setGridClass(openvdb::GRID_LEVEL_SET);

    weights_ = openvdb::FloatGrid::create(0.0f);
    weights_->setName("W(x): weights grid");
    weights_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    weights_->setGridClass(openvdb::GRID_UNKNOWN);
}

void VDBVolume::UpdateTSDF(const float& sdf,
                           const openvdb::Coord& voxel,
                           const std::function<float(float)>& weighting_function) {
    using AccessorRW = openvdb::tree::ValueAccessorRW<openvdb::FloatTree>;
    if (sdf > -sdf_trunc_) {
        AccessorRW tsdf_acc = AccessorRW(tsdf_->tree());
        AccessorRW weights_acc = AccessorRW(weights_->tree());
        const float tsdf = std::min(sdf_trunc_, sdf);
        const float weight = weighting_function(sdf);
        const float last_weight = weights_acc.getValue(voxel);
        const float last_tsdf = tsdf_acc.getValue(voxel);
        const float new_weight = weight + last_weight;
        const float new_tsdf = (last_tsdf * last_weight + tsdf * weight) / (new_weight);
        tsdf_acc.setValue(voxel, new_tsdf);
        weights_acc.setValue(voxel, new_weight);
    }
}

void VDBVolume::Integrate(openvdb::FloatGrid::Ptr grid,
                          const std::function<float(float)>& weighting_function) {
    for (auto iter = grid->cbeginValueOn(); iter.test(); ++iter) {
        const auto& sdf = iter.getValue();
        const auto& voxel = iter.getCoord();
        this->UpdateTSDF(sdf, voxel, weighting_function);
    }
}

void VDBVolume::Integrate(const std::vector<Eigen::Vector3d>& points,
                          const Eigen::Vector3d& origin,
                          const std::function<float(float)>& weighting_function) {
    if (points.empty()) {
        std::cerr << "PointCloud provided is empty\n";
        return;
    }

    // Get some variables that are common to all rays
    const openvdb::math::Transform& xform = tsdf_->transform();
    const openvdb::Vec3R eye(origin.x(), origin.y(), origin.z());

    // Get the "unsafe" version of the grid acessors
    auto tsdf_acc = tsdf_->getUnsafeAccessor();
    auto weights_acc = weights_->getUnsafeAccessor();

    // Launch an for_each execution, use std::execution::par to parallelize this region
    std::for_each(points.cbegin(), points.cend(), [&](const auto& point) {
        // Get the direction from the sensor origin to the point and normalize it
        const Eigen::Vector3d direction = point - origin;
        openvdb::Vec3R dir(direction.x(), direction.y(), direction.z());
        dir.normalize();

        // Truncate the Ray before and after the source unless space_carving_ is specified.
        const auto depth = static_cast<float>(direction.norm());
        const float t0 = space_carving_ ? 0.0f : depth - sdf_trunc_;
        const float t1 = depth + sdf_trunc_;

        // Create one DDA per ray(per thread), the ray must operate on voxel grid coordinates.
        const auto ray = openvdb::math::Ray<float>(eye, dir, t0, t1).worldToIndex(*tsdf_);
        openvdb::math::DDA<decltype(ray)> dda(ray);
        do {
            const auto voxel = dda.voxel();
            const auto voxel_center = GetVoxelCenter(voxel, xform);
            const auto sdf = ComputeSDF(origin, point, voxel_center);
            if (sdf > -sdf_trunc_) {
                const float tsdf = std::min(sdf_trunc_, sdf);
                const float weight = weighting_function(sdf);
                const float last_weight = weights_acc.getValue(voxel);
                const float last_tsdf = tsdf_acc.getValue(voxel);
                const float new_weight = weight + last_weight;
                const float new_tsdf = (last_tsdf * last_weight + tsdf * weight) / (new_weight);
                tsdf_acc.setValue(voxel, new_tsdf);
                weights_acc.setValue(voxel, new_weight);
            }
        } while (dda.step());
    });
}

openvdb::FloatGrid::Ptr VDBVolume::Prune(float min_weight) const {
    const auto weights = weights_->tree();
    const auto tsdf = tsdf_->tree();
    const auto background = sdf_trunc_;
    openvdb::FloatGrid::Ptr clean_tsdf = openvdb::FloatGrid::create(sdf_trunc_);
    clean_tsdf->setName("D(x): Pruned signed distance grid");
    clean_tsdf->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    clean_tsdf->setGridClass(openvdb::GRID_LEVEL_SET);
    clean_tsdf->tree().combine2Extended(tsdf, weights, [=](openvdb::CombineArgs<float>& args) {
        if (args.aIsActive() && args.b() > min_weight) {
            args.setResult(args.a());
            args.setResultIsActive(true);
        } else {
            args.setResult(background);
            args.setResultIsActive(false);
        }
    });
    return clean_tsdf;
}

ImplicitRegistration::ImplicitRegistration(VDBVolume& vdb_volume_global, float clipping_range)
    : vdb_volume_global_(vdb_volume_global), clipping_range_(clipping_range) {}

openvdb::tools::ScalarToVectorConverter<openvdb::FloatGrid>::Type::Ptr
ImplicitRegistration::ComputeGradient(const openvdb::FloatGrid::Ptr grid) const {
    openvdb::tools::Gradient<openvdb::FloatGrid,
                             openvdb::tools::gridop::ToMaskGrid<openvdb::FloatGrid>::Type,
                             openvdb::util::NullInterrupter, openvdb::math::CD_4TH>
        grad_op(*grid);
    return grad_op.process(true);
}

openvdb::FloatGrid::Ptr ImplicitRegistration::ClipVolume(const Sophus::SE3d& T) const {
    auto center = Eigen2VDB::Vec3(T.translation());
    openvdb::BBoxd bbox(center - clipping_range_, center + clipping_range_);

    auto R = T.rotationMatrix();

    bbox.expand(center +
                Eigen2VDB::Vec3<double>(
                    R * Eigen::Vector3d{-clipping_range_, -clipping_range_, -clipping_range_}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(
                    R * Eigen::Vector3d{-clipping_range_, -clipping_range_, clipping_range_}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(
                    R * Eigen::Vector3d{-clipping_range_, clipping_range_, -clipping_range_}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(
                    R * Eigen::Vector3d{-clipping_range_, clipping_range_, clipping_range_}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(
                    R * Eigen::Vector3d{clipping_range_, -clipping_range_, -clipping_range_}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(
                    R * Eigen::Vector3d{clipping_range_, -clipping_range_, clipping_range_}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(
                    R * Eigen::Vector3d{clipping_range_, clipping_range_, -clipping_range_}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(
                    R * Eigen::Vector3d{clipping_range_, clipping_range_, clipping_range_}));
    return openvdb::tools::clip(*vdb_volume_global_.tsdf_, bbox);
}

Sophus::SE3d ImplicitRegistration::ConstantVelocityModel() const {
    Sophus::SE3d T_pred{};
    T_pred.rotationMatrix() = T_2.rotationMatrix().transpose() * T_1.rotationMatrix();
    T_pred.translation() =
        T_2.rotationMatrix().transpose() * (T_1.translation() - T_2.translation());
    return T_pred;
}

std::tuple<std::vector<Eigen::Vector3d>, Sophus::SE3d> ImplicitRegistration::AlignScan(
    const std::vector<Eigen::Vector3d>& points, const Sophus::SE3d& init_tf) {
    const openvdb::FloatGrid::Ptr local_tsdf = this->ClipVolume(init_tf);
    const auto local_tsdf_grad = this->ComputeGradient(local_tsdf);

    const auto local_tsdf_xform = local_tsdf->transform();
    const auto local_tsdf_grad_xform = local_tsdf_grad->transform();

    const auto tsdf_acc = local_tsdf->getConstUnsafeAccessor();
    const auto grad_acc = local_tsdf_grad->getConstUnsafeAccessor();

    // auto T_pred = ConstantVelocityModel();
    auto T = init_tf;

    Matrix6x6d A;
    Matrix6x1d b;

    // clang-format off
    Matrix3x6d grad_pt;
    grad_pt << 1, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0;
    // clang-format on

    int n_iters = 0;
    int max_iters = 150;
    double error_threshold = 5e-3;

    // LM related tuning
    // double lambda = 1e-2;
    // int lambda_up_factor = 10;
    // int lambda_down_factor = 10;

    // double prev_sum_sdf = std::numeric_limits<double>::max();
    double new_sum_sdf = 0;

    auto sdf_threshold = vdb_volume_global_.sdf_trunc_ / 2;

    while (true) {
        A.setConstant(0);
        b.setConstant(0);

        auto points_g = ApplyTransform(points, T);
        std::for_each(points_g.cbegin(), points_g.cend(), [&](const Eigen::Vector3d& point_g) {
            auto index_pos = local_tsdf_xform.worldToIndex(Eigen2VDB::Vec3(point_g));
            auto voxel_tsdf = openvdb::math::Coord(index_pos.x(), index_pos.y(), index_pos.z());
            float distance = tsdf_acc.getValue(voxel_tsdf);

            // Sensitive to this params
            if (std::abs(distance) < sdf_threshold) {
                // double weight = 1 - std::pow((distance / sdf_threshold), 4);
                auto index_pos = local_tsdf_xform.worldToIndex(Eigen2VDB::Vec3(point_g));
                auto voxel_grad = openvdb::math::Coord(index_pos.x(), index_pos.y(), index_pos.z());
                auto grad_sdf = VDB2Eigen::Vec3<double>(grad_acc.getValue(voxel_grad));

                grad_pt.block<3, 3>(0, 3) = -1 * Sophus::SO3d::hat(point_g);
                auto grad = grad_pt.transpose() * grad_sdf;

                A += grad * grad.transpose();
                b += distance * grad;
            }
        });
        auto se3_old = T.log();

        // auto B = Eigen::Matrix<double, 6, 6>::Identity();
        auto se3_new = se3_old - A.inverse() * b;
        T = Sophus::SE3d::exp(se3_new);

        T_2 = T_1;
        T_1 = T;
        // points_g = ApplyTransform(points, T);
        // std::for_each(points_g.cbegin(), points_g.cend(), [&](const Eigen::Vector3d& point_g) {
        //     auto voxel_tsdf = openvdb::math::Coord::round(
        //         local_tsdf_xform.worldToIndex(Eigen2VDB::Vec3(point_g)));
        //     new_sum_sdf += std::pow(tsdf_acc.getValue(voxel_tsdf), 2);
        // });
        // if (new_sum_sdf > prev_sum_sdf) {
        //     lambda *= lambda_up_factor;
        //     T = Sophus::SE3d::exp(se3_old);
        // } else {
        //     lambda /= lambda_down_factor;
        // }

        // prev_sum_sdf = new_sum_sdf;
        // new_sum_sdf = 0;

        n_iters++;

        if (n_iters > max_iters ||
            (se3_new - se3_old).lpNorm<Eigen::Infinity>() < error_threshold) {
            std::cout << "Scan Aligning converged; n_iters = " << n_iters << "\n";
            break;
        }
    }

    auto aligned_points = ApplyTransform(points, T);
    std::for_each(
        aligned_points.cbegin(), aligned_points.cend(), [&](const Eigen::Vector3d& point) {
            auto voxel_tsdf =
                openvdb::math::Coord::round(local_tsdf_xform.worldToIndex(Eigen2VDB::Vec3(point)));
            new_sum_sdf += std::pow(tsdf_acc.getValue(voxel_tsdf), 2);
        });
    std::cout << "RMS: " << std::sqrt(new_sum_sdf / aligned_points.size()) << "\n";

    return std::make_tuple(aligned_points, T);
}
}  // namespace vdbfusion
