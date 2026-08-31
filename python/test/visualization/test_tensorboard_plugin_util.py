# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
import numpy as np
import pytest

pytest.importorskip("tensorboard")

import open3d as o3d
from open3d.visualization.tensorboard_plugin.util import _normalize


@pytest.mark.parametrize(
    "tensor, expected, expected_min, expected_max",
    [
        (o3d.core.Tensor([0, 127, 255], dtype=o3d.core.uint8),
         np.array([0, 127, 255], dtype=np.uint8), 0, 1),
        (o3d.core.Tensor([1.0, 3.0], dtype=o3d.core.float32),
         np.array([0.0, 1.0], dtype=np.float32), 1.0, 3.0),
        (np.array([0, 127, 255], dtype=np.uint8),
         np.array([0, 127, 255], dtype=np.uint8), 0, 1),
        (np.array([1.0, 3.0], dtype=np.float32),
         np.array([0.0, 1.0], dtype=np.float32), 1.0, 3.0),
    ],
)
def test_normalize_dtype_handling(tensor, expected, expected_min, expected_max):
    normalized, min_value, max_value = _normalize(tensor)
    if isinstance(normalized, o3d.core.Tensor):
        normalized = normalized.numpy()

    np.testing.assert_allclose(normalized, expected)
    assert min_value == expected_min
    assert max_value == expected_max
