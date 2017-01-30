from __future__ import absolute_import

import tempfile
import shutil

import numpy as np
import pandas as pd
import pandas.util.testing as pd_util
from shapely.geometry import Point

from geopandas import GeoDataFrame, read_file, base
from geopandas.tests.util import unittest, download_nybb
from geopandas import sjoin
from distutils.version import LooseVersion

pandas_0_16_problem = 'fails under pandas < 0.17 due to issue 251,'\
                      'not problem with sjoin.'


@unittest.skipIf(not base.HAS_SINDEX, 'Rtree absent, skipping')
class TestSpatialJoin(unittest.TestCase):

    def setUp(self):
        nybb_filename, nybb_zip_path = download_nybb()
        self.polydf = read_file(nybb_zip_path, vfs='zip://' + nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.crs = self.polydf.crs
        N = 20
        b = [int(x) for x in self.polydf.total_bounds]

        x_range = range(b[0], b[2], int((b[2] - b[0]) / N))
        y_range = range(b[1], b[3], int((b[3] - b[1]) / N))

        self.pointdf = GeoDataFrame(
            [
                {'geometry': Point(x, y), 'attr1': x + y, 'attr2': x - y}
                for x, y in zip(x_range, y_range)
            ], crs=self.crs)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_geometry_name(self):
        # test sjoin is working with other geometry name
        polydf_old_geom_name = self.polydf.geometry.name
        self.polydf = (self.polydf.rename(columns={'geometry': 'new_geom'})
                                  .set_geometry('new_geom'))
        self.assertNotEqual(polydf_old_geom_name, self.polydf.geometry.name)
        res = sjoin(self.polydf, self.pointdf, how="left")
        self.assertEqual(self.polydf.geometry.name, res.geometry.name)

    def test_sjoin_when_dfs_have_complex_indexes(self):
        # relates to bug report 351, 352 and pull request
        polydf_indx = ["BoroName", "BoroCode"]
        self.polydf.set_index(polydf_indx, inplace=True)
        self.pointdf.index = self.pointdf.index.map(lambda x: "x%s" % x)

        self.assertNotIsInstance(self.polydf.index[0], int)
        self.assertNotIsInstance(self.pointdf.index[0], int)
        self.assertTrue(set(polydf_indx).isdisjoint(set(self.polydf.columns)))

        res = sjoin(self.polydf, self.pointdf, how="left")
        expected_cols = (
            set(self.polydf.columns).union(set(self.pointdf.columns)))
        self.assertTrue(expected_cols, set(res.columns))

    def test_index_name_conflicts_with_column_name(self):
        self.polydf.index.set_names("BoroName", inplace=True)
        res = sjoin(self.polydf, self.pointdf, how="left")
        expected_cols = (
            set(self.polydf.columns).union(set(self.pointdf.columns)))
        self.assertTrue(expected_cols, set(res.columns))


    def test_sjoin_left(self):
        df = sjoin(self.pointdf, self.polydf, how='left')
        self.assertEquals(df.shape, (21, 8))
        for i, row in df.iterrows():
            self.assertEquals(row.geometry.type, 'Point')
        self.assertTrue('attr1' in df.columns)
        self.assertTrue('BoroCode' in df.columns)

    def test_sjoin_right(self):
        # the inverse of left
        df = sjoin(self.pointdf, self.polydf, how="right")
        df2 = sjoin(self.polydf, self.pointdf, how="left")
        self.assertEquals(df.shape, (12, 8))
        self.assertEquals(df.shape, df2.shape)
        for i, row in df.iterrows():
            self.assertEquals(row.geometry.type, 'MultiPolygon')
        for i, row in df2.iterrows():
            self.assertEquals(row.geometry.type, 'MultiPolygon')

    def test_sjoin_inner(self):
        df = sjoin(self.pointdf, self.polydf, how="inner")
        self.assertEquals(df.shape, (11, 8))

    def test_sjoin_op(self):
        # points within polygons
        df = sjoin(self.pointdf, self.polydf, how="left", op="within")
        self.assertEquals(df.shape, (21, 8))
        self.assertEquals(df.iloc[1]['BoroName'], 'Staten Island')

        # points contain polygons? never happens so we should have nulls
        df = sjoin(self.pointdf, self.polydf, how="left", op="contains")
        self.assertEquals(df.shape, (21, 8))
        self.assertTrue(np.isnan(df.iloc[1]['Shape_Area']))

    def test_sjoin_bad_op(self):
        # AttributeError: 'Point' object has no attribute 'spandex'
        self.assertRaises(ValueError, sjoin, self.pointdf, self.polydf,
                          how="left", op="spandex")

    def test_sjoin_duplicate_column_name(self):
        pointdf2 = self.pointdf.rename(columns={'attr1': 'Shape_Area'})
        df = sjoin(pointdf2, self.polydf, how="left")
        self.assertTrue('Shape_Area_left' in df.columns)
        self.assertTrue('Shape_Area_right' in df.columns)

    def test_sjoin_values(self):
        # GH190
        self.polydf.index = [1, 3, 4, 5, 6]
        df = sjoin(self.pointdf, self.polydf, how='left')
        self.assertEquals(df.shape, (21, 8))
        df = sjoin(self.polydf, self.pointdf, how='left')
        self.assertEquals(df.shape, (12, 8))

    @unittest.skipIf(str(pd.__version__) < LooseVersion('0.17'),
                     pandas_0_16_problem)
    def test_left_sjoin_when_empty(self):
        to_concat = [
            self.pointdf.iloc[17:],
            pd.Series(name='index_right', dtype='int64'),
            self.polydf.iloc[:0].drop('geometry', axis=1)
        ]
        actual = sjoin(self.pointdf.iloc[17:], self.polydf, how='left')
        expected = GeoDataFrame(pd.concat(to_concat, axis=1), crs=actual.crs)
        pd_util.assert_frame_equal(expected,
                                   actual.reindex(columns=expected.columns))

    @unittest.skipIf(str(pd.__version__) < LooseVersion('0.17'),
                     pandas_0_16_problem)
    def test_inner_sjoin_when_empty(self):
        to_concat = [
            self.pointdf.iloc[:0],
            pd.Series(name='index_right', dtype='int64'),
            self.polydf.drop('geometry', axis=1).iloc[:0]
        ]
        actual = sjoin(self.pointdf.iloc[17:], self.polydf, how='inner')
        expected = GeoDataFrame(pd.concat(to_concat, axis=1), crs=actual.crs)
        pd_util.assert_frame_equal(expected,
                                   actual.reindex(columns=expected.columns))

    @unittest.skipIf(str(pd.__version__) < LooseVersion('0.17'),
                     pandas_0_16_problem)
    def test_right_sjoin_when_empty(self):

        right_idxs = pd.Series(range(0, 5), name='index_right', dtype='int64')
        to_concat = [
            self.pointdf.drop('geometry', axis=1).iloc[:0],
            pd.Series(name='index_left', dtype='int64'),
            right_idxs, self.polydf
        ]
        actual = sjoin(self.pointdf.iloc[17:], self.polydf, how='right')

        expected = GeoDataFrame(pd.concat(to_concat, axis=1), crs=actual.crs)
        expected.set_index('index_right', inplace=True)

        pd_util.assert_frame_equal(expected,
                                   actual.reindex(columns=expected.columns))

    @unittest.skip("Not implemented")
    def test_sjoin_outer(self):
        df = sjoin(self.pointdf, self.polydf, how="outer")
        self.assertEquals(df.shape, (21, 8))
