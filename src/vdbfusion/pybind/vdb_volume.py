# MIT License
#
# # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Any, Optional, Tuple, Callable, overload

import numpy as np

from . import vdbfusion_pybind


class VDBVolume:
    def __init__(
        self,
        voxel_size: float,
        sdf_trunc: float,
        space_carving: bool = False,
    ):
        self._volume = vdbfusion_pybind._VDBVolume(
            voxel_size=np.float32(voxel_size),
            sdf_trunc=np.float32(sdf_trunc),
            space_carving=space_carving,
        )
        # Passthrough all data members from the C++ API
        self.voxel_size = self._volume._voxel_size
        self.sdf_trunc = self._volume._sdf_trunc
        self.space_carving = self._volume._space_carving
        self.pyopenvdb_support_enabled = self._volume.PYOPENVDB_SUPPORT_ENABLED
        if self.pyopenvdb_support_enabled:
            self.tsdf = self._volume._tsdf
            self.weights = self._volume._weights

    def __repr__(self) -> str:
        return (
            f"VDBVolume with:\n"
            f"voxel_size    = {self.voxel_size}\n"
            f"sdf_trunc     = {self.sdf_trunc}\n"
            f"space_carving = {self.space_carving}\n"
        )

    @overload
    def integrate(
        self,
        points: np.ndarray,
        extrinsic: np.ndarray,
        weighting_function: Callable[[float], float],
    ) -> None:
        ...

    @overload
    def integrate(self, points: np.ndarray, extrinsic: np.ndarray, weight: float) -> None:
        ...

    @overload
    def integrate(self, points: np.ndarray, extrinsic: np.ndarray) -> None:
        ...

    @overload
    def integrate(self, grid, weighting_function: Callable[[float], float]) -> None:
        ...

    @overload
    def integrate(self, grid, weight: float) -> None:
        ...

    @overload
    def integrate(self, grid) -> None:
        ...

    def integrate(
        self,
        points: Optional[np.ndarray] = None,
        extrinsic: Optional[np.ndarray] = None,
        grid: Optional[Any] = None,
        weight: Optional[float] = None,
        weighting_function: Optional[Callable[[float], float]] = None,
    ) -> None:
        if grid is not None:
            if not self.pyopenvdb_support_enabled:
                raise NotImplementedError("Please compile with PYOPENVDB_SUPPORT_ENABLED")
            if weighting_function is not None:
                return self._volume._integrate(grid, weighting_function)
            if weight is not None:
                return self._volume._integrate(grid, weight)
            return self._volume._integrate(grid)
        else:
            assert isinstance(points, np.ndarray), "points must by np.ndarray(n, 3)"
            assert points.dtype == np.float64, "points dtype must be np.float64"
            assert isinstance(extrinsic, np.ndarray), "origin/extrinsic must by np.ndarray"
            assert extrinsic.dtype == np.float64, "origin/extrinsic dtype must be np.float64"
            assert extrinsic.shape in [
                (3,),
                (3, 1),
                (4, 4),
            ], "origin/extrinsic must be a (3,) array or a (4,4) matrix"

            _points = vdbfusion_pybind._VectorEigen3d(points)
            if weighting_function is not None:
                return self._volume._integrate(_points, extrinsic, weighting_function)
            if weight is not None:
                return self._volume._integrate(_points, extrinsic, weight)
            self._volume._integrate(_points, extrinsic)

    @overload
    def update_tsdf(
        self, sdf: float, ijk: np.ndarray, weighting_function: Optional[Callable[[float], float]]
    ) -> None:
        ...

    @overload
    def update_tsdf(self, sdf: float, ijk: np.ndarray) -> None:
        ...

    def update_tsdf(
        self,
        sdf: float,
        ijk: np.ndarray,
        weighting_function: Optional[Callable[[float], float]] = None,
    ) -> None:
        if weighting_function is not None:
            return self._volume._update_tsdf(sdf, ijk, weighting_function)
        return self._volume._update_tsdf(sdf, ijk)

    def extract_triangle_mesh(self, fill_holes: bool = True, min_weight: float = 0.0) -> Tuple:
        """Returns a the vertices and triangles representing the constructed the TriangleMesh.

        If you can afford to use Open3D as dependency just pass the output of this function to the
        TriangleMesh constructor from Open3d.

        vertices, triangles = integrator.extract_triangle_mesh()
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(triangles),
        )
        """
        vertices, triangles = self._volume._extract_triangle_mesh(fill_holes, min_weight)
        return np.asarray(vertices), np.asarray(triangles)

    def extract_vdb_grids(self, out_file: str) -> None:
        """For now, write the internal map representation to a file.

        Contains both D(x) and W(x) grids.
        """
        self._volume._extract_vdb_grids(out_file)

    def prune(self, min_weight: float):
        """Use the W(x) weights grid to cleanup the generated signed distance field according to a
        minimum weight threshold.

        This function is ideal to cleanup the TSDF grid:D(x) before exporting it.
        """
        return self._volume._prune(min_weight)

class ImplicitRegistration:
    def __init__(
        self,
        vdb_volume_global: VDBVolume,
        max_iters: int,
        convergence_threshold: float,
        clipping_range: float
    ):
        self._registration_pipeline = vdbfusion_pybind._ImplicitRegistration(
            vdb_volume_global=vdb_volume_global._volume,
            max_iters=np.int32(max_iters),
            convergence_threshold=np.float32(convergence_threshold),
            clipping_range=np.float32(clipping_range)
        )

        self.vdb_volume_global = self._registration_pipeline._vdb_volume_global
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.clipping_range = clipping_range

    def __repr__(self) -> str:
        return (
            f"Implicit Registration Pipeline with:\n"
            f"global_vdb_volume = {self.vdb_volume_global.__repr__()}\n"
            f"convergence_threshold    = {self.convergence_threshold}\n"
            f"max_iters     = {self.max_iters}\n"
            f"clipping_range = {self.clipping_range}\n"
        )
    
    def alignScan(self, points: np.ndarray, T_init: np.ndarray) -> Tuple:
        assert isinstance(points, np.ndarray), "points must by np.ndarray(n, 3)"
        assert points.dtype == np.float64, "points dtype must be np.float64"
        assert isinstance(T_init, np.ndarray), "initial guess tf must by np.ndarray"
        assert T_init.dtype == np.float64, "initial guess tf dtype must be np.float64"
        assert T_init.shape == (4, 4) , "initial guess tf must be a (4,4) matrix"
        _points = vdbfusion_pybind._VectorEigen3d(points)
        aligned_points, T, n_iters = self._registration_pipeline._align_scan(_points, T_init)
        return np.asarray(aligned_points), T, n_iters