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

using Sophus::SE3d;
using PointCloud = std::vector<Eigen::Vector3d>;

typedef Eigen::Matrix<double, 3, 6> Matrix3x6d;
typedef Eigen::Matrix<double, 6, 7> Matrix6x7d;
typedef Eigen::Matrix<double, 6, 6> Matrix6x6d;

typedef Eigen::Matrix<double, 6, 1> Matrix6x1d;

auto square = [](auto val) { return val * val; };

namespace {
PointCloud ApplyTransform(const PointCloud& pcl, const SE3d& T) {
    auto R = T.rotationMatrix();
    auto tr = T.translation();

    PointCloud pcl_t(pcl.size());

    std::transform(std::execution::par_unseq, pcl.cbegin(), pcl.cend(), pcl_t.begin(),
                   [&](const Eigen::Vector3d& pt) { return (R * pt) + tr; });

    return pcl_t;
}
}  // namespace

namespace vdbfusion {

ImplicitRegistration::ImplicitRegistration(VDBVolume& vdb_volume_global,
                                           const registrationConfigParams& config)
    : vdb_volume_global_(vdb_volume_global), config_(config) {}

openvdb::tools::ScalarToVectorConverter<openvdb::FloatGrid>::Type::Ptr
ImplicitRegistration::ComputeGradient(const openvdb::FloatGrid::Ptr grid) const {
    openvdb::tools::Gradient<openvdb::FloatGrid,
                             openvdb::tools::gridop::ToMaskGrid<openvdb::FloatGrid>::Type,
                             openvdb::util::NullInterrupter, openvdb::math::CD_4TH>
        op(*grid);
    return op.process(true);
}

openvdb::FloatGrid::Ptr ImplicitRegistration::ClipVolume(const SE3d& T) const {
    auto center = Eigen2VDB::Vec3(T.translation());
    auto clip_val = config_.clipping_range_;
    openvdb::BBoxd bbox(center - clip_val, center + clip_val);

    auto R = T.rotationMatrix();

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
    return openvdb::tools::clip(*vdb_volume_global_.tsdf_, bbox);
}

SE3d ImplicitRegistration::ConstantVelocityModel() const {
    SE3d T_pred{};
    T_pred.rotationMatrix() = T_minus_2.rotationMatrix().transpose() * T_minus_1.rotationMatrix();
    T_pred.translation() = T_minus_2.rotationMatrix().transpose() *
                           (T_minus_1.translation() - T_minus_2.translation());
    return T_pred;
}

std::tuple<PointCloud, SE3d, int> ImplicitRegistration::AlignScan(const PointCloud& pcl_local,
                                                                  const SE3d& T_init) {
    const auto xform = vdb_volume_global_.tsdf_->transform();
    const auto tsdf_grad = this->ComputeGradient(vdb_volume_global_.tsdf_);

    const auto tsdf_acc = vdb_volume_global_.tsdf_->getConstUnsafeAccessor();
    const auto grad_acc = tsdf_grad->getConstUnsafeAccessor();

    auto T = T_init * ConstantVelocityModel();

    Matrix6x6d H;
    Matrix6x1d g;
    Matrix6x1d J;

    int n_iters = 0;
    while (true) {
        H.setZero();
        g.setZero();

        auto pcl_global = ApplyTransform(pcl_local, T);

        auto sum_sqrd_sdf = 0.0;
        std::for_each(pcl_global.cbegin(), pcl_global.cend(), [&](const Eigen::Vector3d& point) {
            auto index_pos = xform.worldToIndex(Eigen2VDB::Vec3(point));
            auto voxel = openvdb::math::Coord(index_pos.x(), index_pos.y(), index_pos.z());

            float distance = tsdf_acc.getValue(voxel);
            auto grad_sdf = VDB2Eigen::Vec3<double>(grad_acc.getValue(voxel));

            J.setZero();
            if (grad_sdf.norm() > 1e-8) {
                J.head<3>() = grad_sdf;
                J.tail<3>() = point.cross(grad_sdf);
                H += J * J.transpose();
                g -= distance * J;
            }

            sum_sqrd_sdf += square(distance);
        });

        auto dx = H.ldlt().solve(g);
        auto dT = SE3d::exp(dx);
        T = dT * T;

        T_minus_2 = T_minus_1;
        T_minus_1 = T;

        n_iters++;

        if (n_iters > config_.max_iters_ || dx.norm() < config_.convergence_threshold_) {
            std::cout << "Scan Aligning converged, n_iters = " << n_iters << "\n";
            std::cout << "Root Mean Signed Distance after convergence: "
                      << std::sqrt(sum_sqrd_sdf / pcl_global.size()) << "\n";
            break;
        }
    }

    auto aligned_points = ApplyTransform(pcl_local, T);
    return std::make_tuple(aligned_points, T, n_iters);
}
}  // namespace vdbfusion