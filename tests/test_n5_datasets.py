"""Unit tests for :mod:`dl_utils.n5_datasets`.

Driven entirely by a YAML holding the parameters of a real run — a ``dataset`` name and
the ``datasetConfig`` kwargs splatted into it, the same shape ``training.py`` consumes::

    python test_n5_datasets.py --config_file config/n5_datasets_test.yaml -v

Everything the tests assert is derived from the mapper that config points at, so no
expectation is hardcoded to a particular dataset.  Arguments after ``--config_file`` are
forwarded to ``unittest``, so ``-v`` and ``-k test_sampleKey`` work as usual.

Two things keep a run bounded and non-destructive:

* conversion is exercised on the **first mapper row only**, so runtime tracks one volume
  rather than the whole dataset;
* that conversion writes to a ``_unittest``-suffixed dataset directory under the
  configured ``save_dir``.  Sharing the real dataset's name would make ``run()`` overwrite
  its ``<name>_sample_mapper.csv`` with a one-row index.
"""

import argparse
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from typing import Any, Dict
from unittest import mock

import numpy as np
import pandas as pd
import skimage.io
import yaml
import z5py

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dl_utils import n5_processing  # noqa: E402
from dl_utils.n5_processing import (  # noqa: E402
    BaseMaskDataset,
    BaseVolumeDataset,
    MultiDirectoryN5Dataset,
)

#: Only what the YAML may omit. Everything else must come from the configuration, because
#: the point of this suite is to test the parameters of a real run.
DEFAULTS: Dict[str, Any] = {
    "dataset": "MultiDirectoryN5Dataset",
    "datasetConfig": {
        "sampleMapperDFPath": None,
        "save_dir": None,
        "volumePathColumn": "sample_path",
        "maskPathColumn": "mask_path",
        "channelColumn": "channel",
        "channelKey": "dapi",
        "resolution": None,
        "datasetName": None,
    },
}

CONFIG: Dict[str, Any] = DEFAULTS
MAPPER: pd.DataFrame          # the configured sample mapper, loaded once
TEMP_SAVE_DIR = None          # only when the config leaves save_dir unset


def mergeConfig(base: dict, override: dict) -> dict:
    """*base* with *override* laid over it, recursing into nested dicts."""
    merged = dict(base)
    for key, value in (override or {}).items():
        nested = isinstance(value, dict) and isinstance(base.get(key), dict)
        merged[key] = mergeConfig(base[key], value) if nested else value
    return merged


def datasetClass() -> type:
    """The class under test, resolved by name out of :mod:`dl_utils.n5_datasets`."""
    name = CONFIG["dataset"]
    if not hasattr(n5_datasets, name):
        raise KeyError(f"dl_utils.n5_datasets has no class {name!r}")
    return getattr(n5_datasets, name)


def datasetConfig(**overrides) -> dict:
    """The configured constructor kwargs, with *overrides* applied."""
    return {**CONFIG["datasetConfig"], **overrides}


def setUpModule() -> None:
    """Load the configured mapper and fail with a pointed message if it is unusable."""
    global MAPPER, TEMP_SAVE_DIR
    config = CONFIG["datasetConfig"]

    mapperPath = config["sampleMapperDFPath"]
    if not mapperPath or not os.path.isfile(str(mapperPath)):
        raise unittest.SkipTest(
            f"datasetConfig.sampleMapperDFPath is not a readable file: {mapperPath!r}. "
            f"Point it at the sample mapper of a real run.")

    if config["save_dir"] is None:
        TEMP_SAVE_DIR = tempfile.TemporaryDirectory(prefix="n5_unittest_")
        config["save_dir"] = TEMP_SAVE_DIR.name

    MAPPER = BaseVolumeDataset.loadDataFrame(str(mapperPath))
    required = [column for column in (config["volumePathColumn"],
                                      config["maskPathColumn"],
                                      config["channelColumn"]) if column]
    missing = [column for column in required if column not in MAPPER.columns]
    if missing:
        raise unittest.SkipTest(
            f"{mapperPath} has no {missing} column(s); it has {list(MAPPER.columns)}. "
            f"Correct the column names in datasetConfig.")


def tearDownModule() -> None:
    if TEMP_SAVE_DIR is not None:
        TEMP_SAVE_DIR.cleanup()


class N5TestCase(unittest.TestCase):
    """The three dataset fixtures, plus the scratch paths the write tests need."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="n5_unittest_in_")
        self.addCleanup(tmp.cleanup)
        self.inputDir = tmp.name          # temp mapper CSVs: inputs, not artefacts
        self.volumeColumn = CONFIG["datasetConfig"]["volumePathColumn"]
        self.maskColumn = CONFIG["datasetConfig"]["maskPathColumn"]

    # ---- Fixtures ----------------------------------------------------

    def fullDataset(self, **overrides):
        """The configured dataset, verbatim. Read-only: never ``run()`` this one."""
        return datasetClass()(**datasetConfig(**overrides))

    def subsetDataset(self, rows: int = 1, **overrides):
        """The first *rows* of the mapper, converted into a ``_unittest`` sibling.

        Namespaced deliberately: ``run()`` rewrites ``sampleMapperOutPath``, so a
        one-row dataset sharing the real name would replace that dataset's index.
        """
        return datasetClass()(**datasetConfig(
            sampleMapperDFPath=self.writeMapper(MAPPER.head(rows), "subset.csv"),
            datasetName=f"{self.fullDataset().datasetDirName}_unittest",
            **overrides))

    def probeDataset(self, **overrides):
        """A dataset under a name that cannot already exist on disk."""
        return self.fullDataset(datasetName=f"_unittest_probe_{uuid.uuid4().hex[:8]}",
                                **overrides)

    # ---- Scratch -----------------------------------------------------

    def writeMapper(self, table: pd.DataFrame, name: str) -> str:
        path = os.path.join(self.inputDir, name)
        table.to_csv(path, index=False)
        return path

    def scratchN5(self) -> str:
        """A throwaway N5 under the configured save_dir, removed when the test ends."""
        path = os.path.join(CONFIG["datasetConfig"]["save_dir"],
                            f"_unittest_primitives_{uuid.uuid4().hex[:8]}.n5")
        self.addCleanup(shutil.rmtree, path, True)
        return path

    # ---- Derived expectations ----------------------------------------

    def firstVolumePath(self) -> str:
        return str(MAPPER[self.volumeColumn].iloc[0])

    def firstMaskPath(self) -> str:
        return str(MAPPER[self.maskColumn].iloc[0])

    @staticmethod
    def resampledShape(dataset, shape) -> tuple:
        """The shape ``resampleVolume`` produces — the isotropic case included.

        Mirrors ``int(round(size * zoom))``; when the resolution is already isotropic
        every zoom is 1 and this reduces to *shape*, which is what the short-circuit
        returns.
        """
        zoom = np.array(dataset.resolution, dtype=float) / min(dataset.resolution)
        return tuple(int(round(size * factor)) for size, factor in zip(shape, zoom))


class TestPathsAndKeys(N5TestCase):
    """Path resolution and key derivation — pure arithmetic, nothing is written."""

    def setUp(self) -> None:
        super().setUp()
        self.dataset = self.fullDataset()
        self.volumePath = self.firstVolumePath()

    def test_sampleRoot_is_the_common_ancestor_of_every_sample(self) -> None:
        directories = [os.path.dirname(os.path.abspath(str(path)))
                       for path in MAPPER[self.volumeColumn]]
        self.assertEqual(self.dataset.sampleRoot, os.path.commonpath(directories))
        for directory in directories:
            self.assertTrue(directory.startswith(self.dataset.sampleRoot))

    def test_sampleRoot_rejects_an_empty_mapper(self) -> None:
        self.dataset.sampleMapperDF = self.dataset.sampleMapperDF.iloc[:0]
        self.dataset.__dict__.pop("sampleRoot", None)        # drop the cached_property
        with self.assertRaises(ValueError):
            _ = self.dataset.sampleRoot

    def test_datasetDirName_prefers_an_explicit_name(self) -> None:
        self.assertEqual(self.fullDataset(datasetName="explicit").datasetDirName,
                         "explicit")
        self.assertEqual(self.fullDataset(datasetName=None).datasetDirName,
                         os.path.basename(self.dataset.sampleRoot))

    def test_datasetDir_is_resolved_without_being_created(self) -> None:
        # R8: creating it is run()'s job. Asserted on a name that cannot pre-exist, so a
        # previous suite run cannot make this pass or fail by accident.
        probe = self.probeDataset()
        self.assertEqual(probe.datasetDir,
                         os.path.join(probe.save_dir, probe.datasetDirName))
        self.assertFalse(os.path.isdir(probe.datasetDir))
        self.assertFalse(probe.confirmDatasetN5File)

    def test_n5Path_and_sampleMapperOutPath_sit_in_datasetDir(self) -> None:
        name = self.dataset.datasetDirName
        self.assertEqual(self.dataset.n5Path,
                         os.path.join(self.dataset.datasetDir, f"{name}.n5"))
        self.assertEqual(self.dataset.sampleMapperOutPath,
                         os.path.join(self.dataset.datasetDir, f"{name}_sample_mapper.csv"))

    def test_getSampleDatasetDir_mirrors_the_key_and_creates(self) -> None:
        probe = self.probeDataset()
        path = probe.getSampleDatasetDir(self.volumePath, "_unittest_probe")
        self.addCleanup(shutil.rmtree, path, True)
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(os.path.relpath(path, probe.save_dir).replace(os.sep, "/"),
                         f"{probe.sampleKey(self.volumePath)}/_unittest_probe")

    def test_sampleKey_diverges_from_the_root_and_is_posix(self) -> None:
        key = self.dataset.sampleKey(self.volumePath)
        self.assertEqual(key, os.path.relpath(os.path.splitext(self.volumePath)[0],
                                              self.dataset.sampleRoot).replace(os.sep, "/"))
        self.assertNotIn("\\", key)
        self.assertFalse(os.path.isabs(key))

    def test_sampleKey_strips_only_known_extensions(self) -> None:
        root = self.dataset.sampleRoot
        self.assertEqual(self.dataset.sampleKey(os.path.join(root, "a", "img.ome.tif")),
                         "a/img.ome")
        self.assertEqual(self.dataset.sampleKey(os.path.join(root, "a", "img.n5")),
                         "a/img.n5")

    def test_sampleKey_is_unique_across_the_mapper(self) -> None:
        keys = [self.dataset.datasetKey(str(row[self.volumeColumn]),
                                        self.dataset.RAW_KEY,
                                        self.dataset.channelOf(row))
                for _, row in self.dataset.sampleMapperDF.iterrows()]
        self.assertEqual(len(set(keys)), len(keys), "the mapper collides on an N5 key")

    def test_channelOf_reads_the_column_and_falls_back(self) -> None:
        column = CONFIG["datasetConfig"]["channelColumn"]
        if column:
            channel = str(MAPPER[column].iloc[0])
            self.assertEqual(self.dataset.channelOf(pd.Series({column: channel})), channel)
        self.assertEqual(self.dataset.channelOf(pd.Series({"_absent": 1})),
                         self.dataset.channelKey)
        self.assertEqual(self.dataset.channelOf(pd.Series({column or "c": np.nan})),
                         self.dataset.channelKey)

    def test_channelOf_rejects_unusable_key_segments(self) -> None:
        column = CONFIG["datasetConfig"]["channelColumn"]
        if not column:
            self.skipTest("no channelColumn configured, so no value can be rejected")
        for channel in ("nested/channel", "   "):
            with self.subTest(channel=channel), self.assertRaises(ValueError):
                self.dataset.channelOf(pd.Series({column: channel}))

    def test_getN5GroupKey_and_datasetKey_compose(self) -> None:
        stem = self.dataset.sampleKey(self.volumePath)
        self.assertEqual(self.dataset.getN5GroupKey(self.volumePath, "gfp"), f"{stem}/gfp")
        self.assertEqual(self.dataset.getN5GroupKey(self.volumePath),
                         f"{stem}/{self.dataset.channelKey}")
        self.assertEqual(self.dataset.datasetKey(self.volumePath, "raw", "gfp"),
                         f"{stem}/gfp/raw")


class TestVolumePreparation(N5TestCase):
    """``toUint16`` over arrays, ``prepareSampleVolume`` over the configured volumes."""

    def test_toUint16_normalises_intensities(self) -> None:
        volume = np.linspace(10.0, 20.0, 27).reshape(3, 3, 3)
        out = BaseVolumeDataset.toUint16(volume, normalise=True)
        self.assertEqual(out.dtype, np.uint16)
        self.assertEqual((out.min(), out.max()), (0, np.iinfo(np.uint16).max))

    def test_toUint16_zeroes_a_constant_volume(self) -> None:
        out = BaseVolumeDataset.toUint16(np.full((2, 2, 2), 7.0), normalise=True)
        self.assertTrue((out == 0).all())

    def test_toUint16_leaves_labels_untouched(self) -> None:
        labels = np.array([[[0, 1], [2, 3]]], dtype=np.uint16)
        np.testing.assert_array_equal(
            BaseVolumeDataset.toUint16(labels, normalise=False), labels)

    def test_toUint16_rejects_values_outside_uint16(self) -> None:
        for volume in (np.array([70000.0]), np.array([-1.0])):
            with self.subTest(volume=volume), self.assertRaises(ValueError):
                BaseVolumeDataset.toUint16(volume, normalise=False)

    def test_prepareSampleVolume_resamples_a_raw_volume_to_uint16(self) -> None:
        dataset = self.fullDataset()
        source = dataset.loadFromSamplePath(self.firstVolumePath())
        volume = dataset.prepareSampleVolume(self.firstVolumePath(),
                                             dataset.RAW_INTERPOLATION_ORDER)
        self.assertEqual(volume.dtype, np.uint16)
        self.assertEqual(volume.shape, self.resampledShape(dataset, source.shape))

    def test_prepareSampleVolume_keeps_every_label_of_a_mask(self) -> None:
        dataset = self.fullDataset()
        source = dataset.loadFromSamplePath(self.firstMaskPath())
        mask = dataset.prepareSampleVolume(self.firstMaskPath(),
                                           dataset.MASK_INTERPOLATION_ORDER)
        self.assertEqual(mask.shape, self.resampledShape(dataset, source.shape))
        # Nearest-neighbour invents no label; upsampling loses none either.
        self.assertTrue(set(np.unique(mask).tolist()) <= set(np.unique(source).tolist()))

    def test_prepareSampleVolume_rejects_a_non_3d_volume(self) -> None:
        flat = os.path.join(self.inputDir, "flat.tif")
        skimage.io.imsave(flat, np.zeros((4, 4), dtype=np.uint16), check_contrast=False)
        with self.assertRaises(ValueError):
            self.fullDataset().prepareSampleVolume(flat, 3)


class TestN5Primitives(N5TestCase):
    """The group/dataset primitives every write goes through."""

    def setUp(self) -> None:
        super().setUp()
        self.dataset = self.fullDataset()
        self.path = self.scratchN5()

    def test_requireGroup_creates_every_missing_level(self) -> None:
        with z5py.File(self.path, "a", use_zarr_format=False) as n5File:
            self.dataset.requireGroup(n5File, "a/b/c")
            self.assertIn("a", n5File)
            self.assertIn("c", n5File["a/b"])
            self.dataset.requireGroup(n5File, "a/b/c")          # idempotent

    def test_containsKey_walks_nested_keys(self) -> None:
        with z5py.File(self.path, "a", use_zarr_format=False) as n5File:
            self.dataset.requireGroup(n5File, "a/b/c")
            self.assertTrue(self.dataset.containsKey(n5File, "a/b/c"))
            self.assertFalse(self.dataset.containsKey(n5File, "a/b/absent"))
            self.assertFalse(self.dataset.containsKey(n5File, "missing/branch/leaf"))

    def test_writeN5Dataset_roundtrips_and_clamps_chunks(self) -> None:
        volume = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
        with z5py.File(self.path, "a", use_zarr_format=False) as n5File:
            self.dataset.writeN5Dataset(n5File, "s/dapi/raw", volume, "dapi")
            stored = n5File["s/dapi/raw"]
            np.testing.assert_array_equal(stored[:], volume)
            # A chunk may not exceed the axis it chunks.
            self.assertEqual(tuple(stored.chunks), volume.shape)
            self.assertEqual(stored.attrs["channel"], "dapi")


class TestConversion(N5TestCase):
    """``run`` / ``createDatasetN5File`` / ``writeKeyColumn`` / ``assertPairsWithRaw``."""

    def test_run_creates_the_n5_and_the_key_annotated_index(self) -> None:
        dataset = self.subsetDataset()
        dataset.run()
        self.assertTrue(dataset.confirmDatasetN5File)
        self.assertTrue(os.path.isfile(dataset.sampleMapperOutPath))
        index = dataset.read_index(dataset.sampleMapperOutPath)
        self.assertIn(dataset.N5_VOLUME_KEY_COLUMN, index.columns)
        self.assertIn(dataset.N5_MASK_KEY_COLUMN, index.columns)
        self.assertEqual(len(index), len(dataset.sampleMapperDF))

    def test_run_writes_into_its_own_dataset_directory(self) -> None:
        # The guard that keeps the suite off the real dataset's N5 and index.
        self.assertNotEqual(self.subsetDataset().datasetDir, self.fullDataset().datasetDir)

    def test_run_is_idempotent(self) -> None:
        dataset = self.subsetDataset()
        dataset.run()
        with mock.patch.object(datasetClass(), "prepareSampleVolume") as prepare:
            dataset.run()
        prepare.assert_not_called()

    def test_createDatasetN5File_writes_both_raw_and_mask(self) -> None:
        dataset = self.subsetDataset()
        dataset.run()
        with z5py.File(dataset.n5Path, "r") as n5File:
            for _, row in dataset.sampleMapperDF.iterrows():
                volumeKey = row[dataset.N5_VOLUME_KEY_COLUMN]
                maskKey = row[dataset.N5_MASK_KEY_COLUMN]
                self.assertTrue(dataset.containsKey(n5File, volumeKey))
                self.assertTrue(dataset.containsKey(n5File, maskKey))
                # Raw and mask share one group, and one shape.
                self.assertEqual(tuple(n5File[volumeKey].shape),
                                 tuple(n5File[maskKey].shape))

    def test_writeKeyColumn_rejects_a_missing_column(self) -> None:
        with self.assertRaises(KeyError):
            self.fullDataset().__writeKeyVolumes__("_absent", "raw", "key", 3)

    def test_writeKeyColumn_rejects_colliding_keys(self) -> None:
        # The same row twice: one key, two samples.
        collided = pd.concat([MAPPER.head(1), MAPPER.head(1)], ignore_index=True)
        dataset = datasetClass()(**datasetConfig(
            sampleMapperDFPath=self.writeMapper(collided, "collided.csv"),
            datasetName=f"_unittest_collision_{uuid.uuid4().hex[:8]}"))
        with self.assertRaises(ValueError):
            dataset.run()

    def test_assertPairsWithRaw_requires_a_raw_volume(self) -> None:
        dataset = self.fullDataset()
        with z5py.File(self.scratchN5(), "a", use_zarr_format=False) as n5File:
            with self.assertRaises(KeyError):
                dataset.assertPairsWithRaw(n5File, "s/dapi/mask",
                                           np.zeros((2, 2, 2), np.uint16))

    def test_assertPairsWithRaw_rejects_a_shape_mismatch(self) -> None:
        dataset = self.fullDataset()
        with z5py.File(self.scratchN5(), "a", use_zarr_format=False) as n5File:
            dataset.writeN5Dataset(n5File, "s/dapi/raw",
                                   np.zeros((4, 4, 4), np.uint16), "dapi")
            with self.assertRaises(ValueError):
                dataset.assertPairsWithRaw(n5File, "s/dapi/mask",
                                           np.zeros((2, 2, 2), np.uint16))

    def test_assertPairsWithRaw_checks_an_already_stored_mask(self) -> None:
        dataset = self.fullDataset()
        with z5py.File(self.scratchN5(), "a", use_zarr_format=False) as n5File:
            for subkey in ("raw", "mask"):
                dataset.writeN5Dataset(n5File, f"paired/dapi/{subkey}",
                                       np.zeros((4, 4, 4), np.uint16), "dapi")
            dataset.assertPairsWithRaw(n5File, "paired/dapi/mask", None)   # must not raise

            dataset.writeN5Dataset(n5File, "odd/dapi/raw",
                                   np.zeros((4, 4, 4), np.uint16), "dapi")
            dataset.writeN5Dataset(n5File, "odd/dapi/mask",
                                   np.zeros((2, 2, 2), np.uint16), "dapi")
            with self.assertRaises(ValueError):
                dataset.assertPairsWithRaw(n5File, "odd/dapi/mask", None)


class TestReads(N5TestCase):
    """The accessor family. One conversion of one sample serves the whole class."""

    @classmethod
    def setUpClass(cls) -> None:
        inputDir = tempfile.TemporaryDirectory(prefix="n5_unittest_reads_")
        cls.addClassCleanup(inputDir.cleanup)
        mapperPath = os.path.join(inputDir.name, "subset.csv")
        MAPPER.head(1).to_csv(mapperPath, index=False)

        fullName = datasetClass()(**datasetConfig()).datasetDirName
        cls.dataset = datasetClass()(**datasetConfig(
            sampleMapperDFPath=mapperPath, datasetName=f"{fullName}_unittest"))
        cls.dataset.run()
        cls.shape = tuple(cls.dataset.openKey(cls.dataset.keyOf(0)).shape)

    def test_keyOf_returns_the_mapper_keys(self) -> None:
        row = self.dataset.sampleDFRow(0)
        self.assertEqual(self.dataset.keyOf(0), row[self.dataset.N5_VOLUME_KEY_COLUMN])
        self.assertEqual(self.dataset.keyOf(0, self.dataset.N5_MASK_KEY_COLUMN),
                         row[self.dataset.N5_MASK_KEY_COLUMN])
        self.assertTrue(self.dataset.keyOf(0).endswith(f"/{self.dataset.RAW_KEY}"))

    def test_openKey_and_ensure_open_return_handles(self) -> None:
        # Metadata only: a handle knows its shape without reading a voxel.
        self.assertEqual(tuple(self.dataset._ensure_open(0).shape), self.shape)
        self.assertEqual(
            tuple(self.dataset._ensure_open(0, self.dataset.N5_MASK_KEY_COLUMN).shape),
            self.shape)

    def test_get_hr_vol_reads_whole_volumes_and_bboxes(self) -> None:
        bbox = tuple(slice(0, max(size // 2, 1)) for size in self.shape)
        volume = self.dataset.get_hr_vol(0)
        self.assertEqual(volume.shape, self.shape)
        np.testing.assert_array_equal(self.dataset.get_hr_vol(0), volume[bbox])

    def test_get_hr_mask_isolates_a_single_label(self) -> None:
        mask = self.dataset.get_hr_mask(0)
        self.assertEqual(mask.shape, self.shape)
        labels = [label for label in np.unique(mask).tolist() if label]
        if not labels:
            self.skipTest("the first sample's mask carries no instance label")
        isolated = self.dataset.get_hr_mask(0)
        self.assertEqual(set(np.unique(isolated).tolist()), {0, labels[0]})

    def test_get_masked_hr_vol_zeroes_the_background(self) -> None:
        mask = self.dataset.get_hr_mask(0)
        volume = self.dataset.get_hr_vol(0)
        masked = self.dataset.get_masked_hr_vol(0)
        self.assertTrue((masked[mask == 0] == 0).all())
        np.testing.assert_array_equal(masked[mask != 0], volume[mask != 0])

    def test_maskOfPath_matches_the_positional_read(self) -> None:
        row = self.dataset.sampleDFRow(0)
        np.testing.assert_array_equal(
            self.dataset.maskOfPath(row[self.dataset.SAMPLE_PATH_COLUMN],
                                    self.dataset.channelOf(row)),
            self.dataset.get_hr_mask(0))

    def test_getitem_returns_the_raw_mask_pair(self) -> None:
        """``MultiDirectoryN5Dataset.__getitem__``, reached through subscripting."""
        raw, mask = self.dataset[0]
        np.testing.assert_array_equal(raw, self.dataset.get_hr_vol(0))
        np.testing.assert_array_equal(mask, self.dataset.get_hr_mask(0))
        self.assertEqual(raw.shape, mask.shape)


class TestObjectClouds(N5TestCase):
    """Per-object point extraction.

    Both methods take an array, so they are exercised on synthetic labels: meshing a real
    organoid mask is minutes of marching cubes per instance and would say nothing extra
    about the code under test.
    """

    @staticmethod
    def block(shape=(16, 16, 16)) -> np.ndarray:
        """One solid object, big enough to clear the mesh and skeleton filters."""
        mask = np.zeros(shape, dtype=bool)
        mask[3:13, 3:13, 3:13] = True
        return mask

    @staticmethod
    def labelled(count: int = 3, shape=(16, 16, 24)) -> np.ndarray:
        """*count* separated slabs labelled ``1..count`` — distinct bounding boxes."""
        mask = np.zeros(shape, dtype=np.uint16)
        for label in range(1, count + 1):
            start = label * 6 - 3
            mask[3:13, 3:13, start:start + 3] = label
        return mask

    def test_objectPoints_skeletonises_when_asked(self) -> None:
        points = BaseMaskDataset.objectPoints(self.block(), surfacePoints=False)
        self.assertEqual(points.ndim, 2)
        self.assertEqual(points.shape[1], 3)
        self.assertGreater(points.shape[0], 0)

    def test_objectPoints_meshes_the_surface(self) -> None:
        try:
            import trimesh  # noqa: F401
        except ImportError:
            self.skipTest("trimesh is not installed")
        points = BaseMaskDataset.objectPoints(self.block(), surfacePoints=True)
        self.assertEqual(points.shape[1], 3)
        self.assertGreater(points.shape[0], 0)

    def test_iter_object_clouds_yields_every_label(self) -> None:
        mask = self.labelled()
        clouds = dict(self.fullDataset().iter_object_clouds(mask, surfacePoints=False))
        self.assertEqual(sorted(clouds), list(range(1, 4)))
        for label, points in clouds.items():
            with self.subTest(label=label):
                self.assertEqual(points.shape[1], 3)
                self.assertEqual(points.dtype, np.float32)
                # Shifted back out of the crop, so coordinates address the whole volume.
                self.assertTrue((points >= 0).all())


class TestMultiDirectoryContract(N5TestCase):
    """``__init__`` and ``validateSamplePaths`` — the wider contract of the subclass."""

    def test_init_requires_both_path_columns(self) -> None:
        parameters = datasetConfig()
        for column in ("volumePathColumn", "maskPathColumn"):
            with self.subTest(column=column), self.assertRaises(TypeError):
                MultiDirectoryN5Dataset(
                    **{key: value for key, value in parameters.items() if key != column})

    def test_validateSamplePaths_accepts_the_configured_layout(self) -> None:
        dataset = self.fullDataset()
        self.assertEqual(len(dataset.sampleMapperDF), len(MAPPER))
        for column in (dataset.SAMPLE_PATH_COLUMN, dataset.SAMPLE_MASK_PATH_COLUMN):
            self.assertIn(column, dataset.sampleMapperDF.columns)

    def test_validateSamplePaths_rejects_a_missing_column(self) -> None:
        mapper = self.writeMapper(MAPPER.drop(columns=[self.maskColumn]), "nomask.csv")
        with self.assertRaises(KeyError):
            self.fullDataset(sampleMapperDFPath=mapper)

    def test_validateSamplePaths_rejects_a_missing_file(self) -> None:
        table = MAPPER.head(1).copy()
        table.loc[0, self.volumeColumn] = os.path.join(self.inputDir, "absent.tif")
        with self.assertRaises(FileNotFoundError):
            self.fullDataset(sampleMapperDFPath=self.writeMapper(table, "absent.csv"))

    def test_validateSamplePaths_allows_no_channel_column(self) -> None:
        # channelOf documents None as supported; validation must not look it up.
        dataset = self.fullDataset(channelColumn=None)
        row = dataset.sampleDFRow(0)
        self.assertEqual(dataset.channelOf(row), dataset.channelKey)
        path = str(row[dataset.SAMPLE_PATH_COLUMN])
        self.assertEqual(
            dataset.datasetKey(path, dataset.RAW_KEY, dataset.channelOf(row)),
            f"{dataset.sampleKey(path)}/{dataset.channelKey}/{dataset.RAW_KEY}")


def parseArgs(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config_file", required=True,
                        help="YAML holding `dataset` and the `datasetConfig` kwargs of a "
                             "real run.")
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    args, unittestArgv = parseArgs(sys.argv[1:])
    with open(args.config_file, encoding="utf-8") as handle:
        CONFIG = mergeConfig(DEFAULTS, yaml.safe_load(handle) or {})
    unittest.main(argv=[sys.argv[0], *unittestArgv])
