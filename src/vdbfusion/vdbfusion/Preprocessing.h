#include <Eigen/Core>
#include <vector>

namespace vdbfusion {
std::vector<Eigen::Vector3d> VoxelDownsample(const std::vector<Eigen::Vector3d>& frame,
                                             double voxel_size);

std::vector<Eigen::Vector3d> Preprocess(const std::vector<Eigen::Vector3d>& frame,
                                        double max_range,
                                        double min_range);

std::vector<Eigen::Vector3d> CorrectKITTIScan(const std::vector<Eigen::Vector3d>& frame);
}  // namespace vdbfusion