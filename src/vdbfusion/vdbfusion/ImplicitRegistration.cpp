#include "ImplicitRegistration.h"

#include <openvdb/math/BBox.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Clip.h>
#include <openvdb/tools/GridOperators.h>

#include <Eigen/Core>
#include <algorithm>
#include <execution>
#include <tuple>

#include "VDBVolume.h"
#include "sophus/se3.hpp"
#include "utils/conversions.h"

using Eigen::Matrix4d;
using Sophus::SE3d;
using PointCloud = std::vector<Eigen::Vector3d>;

typedef Eigen::Matrix<double, 6, 7> Matrix6x7d;
typedef Eigen::Matrix<double, 6, 1> Matrix6x1d;

auto square = [](auto val) { return val * val; };

namespace {
PointCloud ApplyTransform(const PointCloud& pcl, const Matrix4d& T) {
    auto R = T.block<3, 3>(0, 0);
    auto tr = T.block<3, 1>(0, 3);

    PointCloud pcl_t(pcl.size());

    std::transform(std::execution::par_unseq, pcl.cbegin(), pcl.cend(), pcl_t.begin(),
                   [&](const Eigen::Vector3d& pt) { return (R * pt) + tr; });

    return pcl_t;
}

openvdb::BBoxd generateBBox(const Matrix4d& T, const double clip_val) {
    auto center = Eigen2VDB::Vec3<double>(T.block<3, 1>(0, 3));
    openvdb::BBoxd bbox(center - clip_val, center + clip_val);

    auto R = T.block<3, 3>(0, 0);

    bbox.expand(center +
                Eigen2VDB::Vec3<double>(R * Eigen::Vector3d{-clip_val, -clip_val, -clip_val}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(R * Eigen::Vector3d{-clip_val, -clip_val, clip_val}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(R * Eigen::Vector3d{-clip_val, clip_val, -clip_val}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(R * Eigen::Vector3d{-clip_val, clip_val, clip_val}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(R * Eigen::Vector3d{clip_val, -clip_val, -clip_val}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(R * Eigen::Vector3d{clip_val, -clip_val, clip_val}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(R * Eigen::Vector3d{clip_val, clip_val, -clip_val}));
    bbox.expand(center +
                Eigen2VDB::Vec3<double>(R * Eigen::Vector3d{clip_val, clip_val, clip_val}));
    return bbox;
}
}  // namespace

namespace vdbfusion {

ImplicitRegistration::ImplicitRegistration(VDBVolume& vdb_volume_global,
                                           const registrationConfigParams& config)
    : vdb_volume_global_(vdb_volume_global), config_(config) {
    T_minus_1.setIdentity();
    T_minus_2.setIdentity();
}

ImplicitRegistration::ImplicitRegistration(VDBVolume& vdb_volume_global,
                                           const int max_iters,
                                           const float convergence_threshold,
                                           const float clipping_range)
    : vdb_volume_global_(vdb_volume_global),
      config_({max_iters, convergence_threshold, clipping_range}) {
    T_minus_1.setIdentity();
    T_minus_2.setIdentity();
}

openvdb::tools::ScalarToVectorConverter<openvdb::FloatGrid>::Type::Ptr
ImplicitRegistration::ComputeGradient(const openvdb::FloatGrid::Ptr grid, const Matrix4d& T) const {
    auto bbox = generateBBox(T, config_.clipping_range_);

    openvdb::Vec3d idxMin, idxMax;
    openvdb::math::calculateBounds(grid->constTransform(), bbox.min(), bbox.max(), idxMin, idxMax);
    openvdb::CoordBBox region(openvdb::Coord::floor(idxMin), openvdb::Coord::floor(idxMax));

    openvdb::MaskGrid mask(*grid);
    mask.fill(region, true, true);

    openvdb::tools::Gradient<openvdb::FloatGrid,
                             openvdb::tools::gridop::ToMaskGrid<openvdb::FloatGrid>::Type,
                             openvdb::util::NullInterrupter, openvdb::math::CD_4TH>

        op(*grid, mask, nullptr);
    return op.process(true);
}

openvdb::tools::ScalarToVectorConverter<openvdb::FloatGrid>::Type::Ptr
ImplicitRegistration::ComputeGradient(const openvdb::FloatGrid::Ptr grid) const {
    openvdb::tools::Gradient<openvdb::FloatGrid,
                             openvdb::tools::gridop::ToMaskGrid<openvdb::FloatGrid>::Type,
                             openvdb::util::NullInterrupter, openvdb::math::CD_4TH>

        op(*grid);
    return op.process(true);
}

std::tuple<PointCloud, Matrix4d, int> ImplicitRegistration::AlignScan(const PointCloud& pcl_local,
                                                                      const Matrix4d& T_init) {
    const auto xform = vdb_volume_global_.tsdf_->transform();
    const auto tsdf_grad = this->ComputeGradient(vdb_volume_global_.tsdf_, T_init);

    const auto tsdf_acc = vdb_volume_global_.tsdf_->getConstUnsafeAccessor();
    const auto grad_acc = tsdf_grad->getConstUnsafeAccessor();
    Matrix4d T_constant_velocity_model = T_minus_2.inverse() * T_minus_1;

    Matrix4d T = T_init * T_constant_velocity_model;

    Matrix6x7d Hg;

    int n_iters = 0;

    while (true) {
        Hg.setZero();

        auto pcl_global_d = ApplyTransform(pcl_local, T);

        auto sum_sqrd_sdf = 0.0;

        Hg = std::transform_reduce(
            std::execution::par, pcl_global_d.cbegin(), pcl_global_d.cend(), Hg,
            [](auto Hg_sum, auto Hg_new) { return Hg_sum + Hg_new; },
            [&](const Eigen::Vector3d& point) {
                auto index_pos = xform.worldToIndex(Eigen2VDB::Vec3(point));
                auto voxel = openvdb::math::Coord(index_pos.x(), index_pos.y(), index_pos.z());
                Matrix6x7d Hg_new;
                Hg_new.setZero();

                if (tsdf_acc.tree().isValueOn(voxel)) {
                    float distance = tsdf_acc.getValue(voxel);
                    auto grad_sdf = VDB2Eigen::Vec3<double>(grad_acc.getValue(voxel));

                    Matrix6x1d J;
                    J.setZero();

                    J.head<3>() = grad_sdf;
                    J.tail<3>() = point.cross(grad_sdf);
                    Hg_new.block<6, 6>(0, 0) = J * J.transpose();
                    Hg_new.block<6, 1>(0, 6) = -distance * J;

                    sum_sqrd_sdf += square(distance);
                }
                return Hg_new;
            });

        auto H = Hg.block<6, 6>(0, 0);
        auto g = Hg.block<6, 1>(0, 6);

        // Left Perturbation Model
        auto dx = H.ldlt().solve(g);
        auto dT = SE3d::exp(dx).matrix();
        T = dT * T;

        T_minus_2 = T_minus_1;
        T_minus_1 = T;

        n_iters++;

        if (n_iters > config_.max_iters_ || dx.norm() < config_.convergence_threshold_) {
            std::cout << "Root Mean Signed Distance after convergence: "
                      << std::sqrt(sum_sqrd_sdf / pcl_global_d.size()) << "\n";
            break;
        }
    }

    auto aligned_points = ApplyTransform(pcl_local, T);
    return std::make_tuple(aligned_points, T, n_iters);
}
}  // namespace vdbfusion