#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>

#include <Eigen/Core>

#include "VDBVolume.h"
#include "sophus/se3.hpp"

namespace vdbfusion {
struct registrationConfigParams {
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
        const std::vector<Eigen::Vector3d>& pcl_local, const Sophus::SE3d& T_init);

public:
    VDBVolume vdb_volume_global_;
    registrationConfigParams config_;
    Sophus::SE3d T_minus_1{};
    Sophus::SE3d T_minus_2{};
};
}  // namespace vdbfusion