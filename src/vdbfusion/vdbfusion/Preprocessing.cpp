#include "Preprocessing.h"

#include <Eigen/Core>
#include <algorithm>
#include <execution>
#include <sophus/se3.hpp>
#include <unordered_map>
#include <vector>

using Voxel = Eigen::Vector3i;

struct VoxelHash {
    size_t operator()(const Voxel& voxel) const {
        const uint32_t* vec = reinterpret_cast<const uint32_t*>(voxel.data());
        return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
    }
};

std::vector<Eigen::Vector3d> vdbfusion::VoxelDownsample(const std::vector<Eigen::Vector3d>& frame,
                                                        double voxel_size) {
    std::unordered_map<Voxel, Eigen::Vector3d, VoxelHash> grid;
    grid.reserve(frame.size());
    for (const auto& point : frame) {
        const auto voxel = Voxel((point / voxel_size).cast<int>());
        if (grid.find(voxel) != grid.end()) continue;
        grid.insert({voxel, point});
    }
    std::vector<Eigen::Vector3d> frame_dowsampled;
    frame_dowsampled.reserve(grid.size());
    for (const auto& [_, point] : grid) {
        frame_dowsampled.emplace_back(point);
    }
    return frame_dowsampled;
}

std::vector<Eigen::Vector3d> vdbfusion::Preprocess(const std::vector<Eigen::Vector3d>& frame,
                                                   double max_range,
                                                   double min_range) {
    std::vector<Eigen::Vector3d> inliers;
    std::copy_if(frame.cbegin(), frame.cend(), std::back_inserter(inliers), [&](const auto& pt) {
        const double norm = pt.norm();
        return norm < max_range && norm > min_range;
    });
    return inliers;
}

std::vector<Eigen::Vector3d> vdbfusion::CorrectKITTIScan(
    const std::vector<Eigen::Vector3d>& frame) {
    constexpr double VERTICAL_ANGLE_OFFSET = (0.205 * M_PI) / 180.0;
    std::vector<Eigen::Vector3d> corrected_frame(frame.size());
    std::transform(
        std::execution::par, frame.cbegin(), frame.cend(), corrected_frame.begin(),
        [&](const auto& pt) {
            const Eigen::Vector3d rotationVector = pt.cross(Eigen::Vector3d(0, 0, 1));
            return Eigen::AngleAxisd(VERTICAL_ANGLE_OFFSET, rotationVector.normalized()) * pt;
        });
    return corrected_frame;
}