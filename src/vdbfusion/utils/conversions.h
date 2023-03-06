#include <openvdb/openvdb.h>

#include <Eigen/Core>

namespace VDB2Eigen {
template <typename T>
inline Eigen::Matrix<T, 3, 1> Vec3(const openvdb::math::Vec3<T>& vdb_vec) {
    return Eigen::Matrix<T, 3, 1>{vdb_vec.x(), vdb_vec.y(), vdb_vec.z()};
}
}  // namespace VDB2Eigen

namespace Eigen2VDB {
template <typename T>
inline openvdb::math::Vec3<T> Vec3(const Eigen::Matrix<T, 3, 1>& eigen_vec) {
    return openvdb::math::Vec3<T>{eigen_vec.x(), eigen_vec.y(), eigen_vec.z()};
}
}  // namespace Eigen2VDB

template <typename T, int nRows, int nCols>
Eigen::Matrix<T, nRows, nCols> extractDiagonal(const Eigen::Matrix<T, nRows, nCols> mat) {
    Eigen::Matrix<T, nRows, nCols> diag;
    // clang-format off
    diag << mat(0, 0),     0    ,     0    ,
                0    , mat(1, 1),     0    ,
                0    ,     0    , mat(2, 2);
    // clang-format on
    return diag;
}