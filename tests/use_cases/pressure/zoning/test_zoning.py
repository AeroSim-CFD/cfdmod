import unittest

import numpy as np
from nassu.lnas import LagrangianGeometry

from cfdmod.use_cases.pressure.zoning.processing import get_indexing_mask
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel


class TestGetRegionIndexMask(unittest.TestCase):
    def test_get_indexing_mask(self):
        vertices = np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]])
        triangles = np.array([[0, 1, 2], [1, 3, 2]])
        mesh = LagrangianGeometry(vertices, triangles)

        zoning = ZoningModel(x_intervals=[0, 5, 10], y_intervals=[0, 10], z_intervals=[0, 10])
        zoning.offset_limits(0.1)

        df_regions = zoning.get_regions_df()

        # Test the function to get region index mask
        region_mask = get_indexing_mask(mesh, df_regions)

        # Check if the region mask has the correct length
        self.assertEqual(len(region_mask), len(triangles))

        # Check if the triangles have been correctly assigned to regions
        self.assertEqual(region_mask[0], 0)  # First triangle should be in region 0
        self.assertEqual(region_mask[1], 1)  # Second triangle should be in region 1


if __name__ == "__main__":
    unittest.main()
