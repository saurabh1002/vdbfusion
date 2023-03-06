#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>

#include <Eigen/Core>

#include "VDBVolume.h"
#include "sophus/se3.hpp"

namespace vdbfusion {
struct registrationConfigParams {
    int max_iters_;
    float convergence_threshold_;
    float clipping_range_;
};

class ImplicitRegistration {
public:
    ImplicitRegistration(VDBVolume& vdb_volume_global, const registrationConfigParams& config);
    ImplicitRegistration(VDBVolume& vdb_volume_global,
                         const int max_iters_,
                         const float convergence_threshold_,
                         const float clipping_range_);

    ~ImplicitRegistration() = default;

public:
    /// @brief Compute the Gradients of the Signed Distance Field at each voxel location
    openvdb::tools::ScalarToVectorConverter<openvdb::FloatGrid>::Type::Ptr ComputeGradient(
        const openvdb::FloatGrid::Ptr grid, const Eigen::Matrix4d& T) const;

    openvdb::tools::ScalarToVectorConverter<openvdb::FloatGrid>::Type::Ptr ComputeGradient(
        const openvdb::FloatGrid::Ptr grid) const;

    std::tuple<std::vector<Eigen::Vector3d>, Eigen::Matrix4d, int> AlignScan(
        const std::vector<Eigen::Vector3d>& pcl_local, const Eigen::Matrix4d& T_init);

public:
    VDBVolume vdb_volume_global_;
    registrationConfigParams config_;
    Eigen::Matrix4d T_minus_1;
    Eigen::Matrix4d T_minus_2;
};
}  // namespace vdbfusion