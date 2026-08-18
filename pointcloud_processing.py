import os

import pandas as pd
from matplotlib import pyplot as plt

from dl_utils import MOBIE_LABEL_KEY, util
from dl_utils.SampleLoader import SampleLoaderBioImage
from dl_utils.cell_geometry import mask_to_surface_points
from dl_utils.n5_processing import BaseObjectDataset
from skimage.morphology import skeletonize
import numpy as np

from dl_utils.vizualizations import CustomMatplotlib


# TODO: The baseclass deals with N5 processing, here it is point cloud processing
    # TODO: Later, create imaginary class baseclassing both PointObject and N5 classes to keep naming conventions the same

class PointObjectDataset(BaseObjectDataset):
    def __init__(self, *args, **kwargs):

        super().__init__(*args,**kwargs)

        assert os.path.exists(self.objectDFPath), f"Expected objectMapperDF to be in {self.objectDFPath}, adjust save_dir to dynamic finding."

        self.objectMapperDF : pd.DataFrame = SampleLoaderBioImage.loadData(self.objectDFPath)

    def mask2Pointcloud(self, idx: int) -> np.ndarray:
        return mask_to_surface_points(self.get_hr_mask(idx))

    def showTestVolumes(self):
        fig, ax = plt.subplots(1, 3, figsize=(5, 15))
        rng = np.random.default_rng(42)
        object_index = rng.choice(
            self.objectMapperDF.index[
                self.objectMapperDF[self.channelColumn].str.contains("Nuclei", na=False)
            ].to_numpy()
        )

        hr_vol = self.get_hr_vol(object_index)
        masked_hr_vol = self.get_masked_hr_vol(object_index)
        mid_slice = hr_vol.shape[0] // 2
        CustomMatplotlib.subImshow(ax[0], hr_vol[mid_slice], label="Object", vmin=0, vmax=hr_vol[mid_slice].max())
        CustomMatplotlib.subImshow(ax[1], masked_hr_vol[mid_slice], label="Masked object", vmin=0, vmax=masked_hr_vol[mid_slice].max())

        hr_pc = self.mask2Pointcloud(object_index)
        CustomMatplotlib.subScatter(ax=[2], points={"x": hr_pc[:,0], "y": hr_pc[:,1]}, title="Pointcloud object")


    def singleObjectPointcloudDFPath(self, object_label : int, sampleRow : pd.Series) -> str:
        samplePath = sampleRow[self.SAMPLE_PATH_COLUMN]
        singleObjectDFPath = self.singVolumeObjectDFPath(int(sampleRow.name),samplePath) #training_processed/tables/experiment/bbox_sampleFileName.json
        dirs = os.path.dirname(singleObjectDFPath) # ..../experiment/
        sampleFileName = os.path.splitext(os.path.basename(samplePath))[0] # ..../experiment/sampleFileName/
        dirs = os.path.join(dirs, sampleFileName)

        return util.get_savedf_path(str(dirs), typie=f"{object_label}", prefix="object", fileExtension="csv")


    def __writePointcloud__(self) -> None:
        """Write one dataset per sample under *subkey*; record its key in *keyColumn*.

        Keys always derive from the sample *volume* path, so every role of a sample
        (raw, mask, ...) lands in that sample's one group whatever column the pixels
        came from.  Samples already present are skipped, which is what makes
        construction idempotent and an interrupted run resumable.  The N5 is opened
        once for the whole column.
        """


        pathColumn = self.SAMPLE_MASK_PATH_COLUMN
        subkey= self.MASK_KEY
        keyColumn= self.N5_MASK_KEY_COLUMN
        validate = self.assertPairsWithRaw
        interpolationOrder = self.MASK_INTERPOLATION_ORDER


        if pathColumn not in self.sampleMapperDF.columns:
            raise KeyError(
                f"Sample mapper has no {pathColumn!r} column: "
                f"{list(self.sampleMapperDF.columns)}")

        # Keys are pure path arithmetic, so they are derived — and checked for
        # collisions — before a single voxel is written and overwrites its neighbour.
        # NOTE: Constructed from sample path, channel and volume type (e.g. raw, mask etc.) indicator subkey
        keys = pd.Index([self.singleVolumePointcloudDFPath(i, channel=self.channelOf(row), subkey=subkey)
                         for i, row in self.objectMapperDF.iterrows()])

        # Note: Test if duplicated keys exist
        duplicates = keys[keys.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(
                f"Sample mapper maps several rows onto the same N5 key: {duplicates[:5]}")

        with z5py.File(self.n5Path, "a", use_zarr_format=False) as n5File:

            for key, (i, row) in zip(keys, self.sampleMapperDF.iterrows()):
                if self.containsKey(n5File, key):
                    # Still validated: the check reads metadata only, and skipping it on
                    # resume meant an existing dataset was never checked against its pair.
                    if validate is not None:
                        validate(n5File, key, None)
                    logger.debug("Already in N5, skipping: %s", key)
                    continue
                print(f"Writing {key} of indx {i} keys to N5 file {self.n5Path}")
                prepared_dict: dict = self.prepareSampleVolume(row[pathColumn], interpolationOrder)

                volume: np.ndarray = prepared_dict.pop("volume")

                # NOTE: Only can compute after resampled volume. If placed before, computation will be wrong
                dataFrameObject: pd.DataFrame = mask2BBOXDF(volume)

                if validate is not None:
                    validate(n5File, key, volume)

                self.writeN5Dataset(n5File, key, volume, self.channelOf(row))

                logger.info("Wrote %s <- %s", key, row[pathColumn])

                # NOTE: Save each key to map objects to samples
                self.objectMapperDF.append(self.checkpointSingleVolumeObjectDF(dataFrameObject,
                                                                      keyColumn, key, pathColumn, row))

        self.sampleMapperDF[keyColumn] = keys.to_numpy()

        self.objectMapperDF: pd.DataFrame = self.sampleDF2ObjectDFMerge(objectDF=pd.concat(self.objectMapperDF, axis=0),
                                                                        sampleDF=self.sampleMapperDF, on=[keyColumn, BaseDataset.SAMPLE_LABEL_COLUMN])

        self.removeSegmentationErrors()
    #
    # def run(self) -> None:
    #


    def createSampleMapperDFObjects(self) -> None:


        for vol_path, organoid_rows in dataDF.groupby(self.datasetPathColumn):
            mask_path = str(organoid_rows[self.maskPathColumn].iloc[0])
            # Rooted at the VOLUME, so single- and dual-channel runs produce one layout.
            # Resolved per volume so sub-datasets keep their own leaf and same-basename
            # volumes cannot overwrite each other.
            pc_dir = self.getSampleDatasetDir(str(vol_path), "point_clouds")
            pc_path = self._build_point_cloud(str(vol_path), pc_dir)
            self.instances[str(vol_path)] = pc_path
            row: dict = {
                self.datasetPathColumn: vol_path,
                self.maskPathColumn: mask_path,
                self.SAMPLE_PATH_COLUMN: pc_path,
                self.keptIndicesColumn: 0,
            }
            for col in dataDF.columns:
                if col not in row:
                    row[col] = organoid_rows[col].iloc[0]
            rows.append(row)
        df_out = pd.DataFrame(rows)
        df_out[self.SAMPLE_LABEL_COLUMN] = df_out.index
        self.write_index(df_out, self.dfPath)
        assert len(df_out) == len(self.instances), (
            f"Index rows ({len(df_out)}) and registered instances "
            f"({len(self.instances)}) disagree — a sample was dropped."
        )


    def _summary_csv_path(self, raw_path: str) -> str:
        """Tier-2 per-organoid summary path: ``point_clouds/<volstem>_summary.csv``."""
        pc_dir = self.getSampleDatasetDir(raw_path, "point_clouds")
        stem = os.path.splitext(os.path.basename(raw_path))[0]
        return os.path.join(pc_dir, f"{stem}_summary.csv")

    @staticmethod
    def _dataset_name(raw_path: str) -> str:
        """Leaf folder (e.g. ``100uM`` / ``14j``) acting as the 'dataset' grouping."""
        return os.path.basename(os.path.dirname(raw_path))

    def _register_channel(self, channel_row, channel: str) -> pd.DataFrame:
        """Tier 1+2: write one CSV per object and the per-organoid summary; return it.

        Each object's cloud goes to ``point_clouds/objects/<volstem>_<labelid>.csv``
        (columns ``x, y, z``). The returned summary DataFrame has one row per object
        file. Idempotent: if the summary exists and all object files it lists are
        present, it is reused without re-extracting.
        """
        raw_path = str(channel_row[self.datasetPathColumn])
        mask_path = str(channel_row[self.maskPathColumn])
        volstem = os.path.splitext(os.path.basename(raw_path))[0]
        dataset = self._dataset_name(raw_path)
        summary_csv = self._summary_csv_path(raw_path)
        obj_dir = os.path.join(os.path.dirname(summary_csv), "objects")
        summary_cols = [
            self.SAMPLE_PATH_COLUMN, self.labelColumn, self.channel_column,
            self.datasetPathColumn, self.maskPathColumn,
            self.organoid_stem_column, self.organoidColumn,
        ]

        empty = pd.DataFrame(columns=summary_cols)

        # Reuse a valid, complete summary (resumability within wipe-and-rebuild).
        # An unparseable / empty file is treated as no summary -> re-extract.
        if os.path.exists(summary_csv):
            try:
                df = pd.read_csv(summary_csv)
            except pd.errors.EmptyDataError:
                df = None
            if (df is not None and self.SAMPLE_PATH_COLUMN in df.columns
                    and len(df) and df[self.SAMPLE_PATH_COLUMN].map(os.path.exists).all()):
                logger.info("Summary complete, skipping (%d objects): %s", len(df), summary_csv)
                return df

        os.makedirs(obj_dir, exist_ok=True)
        # The N5 was built by run(); reading it by key needs no per-sample conversion.
        rows: List[dict] = []
        for label_id, xyz in self.iter_object_clouds(self.maskOfPath(raw_path)):
            obj_csv = os.path.join(obj_dir, f"{volstem}_{label_id}.csv")
            pd.DataFrame(xyz, columns=["x", "y", "z"]).to_csv(obj_csv, index=False)
            rows.append({
                self.SAMPLE_PATH_COLUMN: obj_csv,
                self.labelColumn: label_id,
                self.channel_column: channel,
                self.datasetPathColumn: raw_path,
                self.maskPathColumn: mask_path,
                self.organoid_stem_column: volstem,
                self.organoidColumn: dataset,
            })

        # Drop samples with no extractable objects: write NO summary file (so a
        # future run re-checks the mask) and return an empty frame, which
        # _build_supermaster excludes from the masters / super-master.
        if not rows:
            if os.path.exists(summary_csv):
                os.remove(summary_csv)
            logger.warning("No objects for %s (%s); skipping sample.", volstem, channel)
            return empty

        df = pd.DataFrame(rows, columns=summary_cols)
        os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
        df.to_csv(summary_csv, index=False)
        logger.info("Wrote %d object clouds + summary → %s", len(df), summary_csv)
        return df

    def _build_supermaster(self, channel_rows_iter) -> None:
        """Tiers 3+4: per-dataset master (concat of organoid summaries) and the
        super-master (concat of masters). Always rebuilt from what is on disk.
        """
        from collections import defaultdict
        per_dataset: dict = defaultdict(list)   # dataset name -> [summary DataFrame, ...]
        dataset_dir: dict = {}                   # dataset name -> leaf folder path

        for channel_row, channel in channel_rows_iter:
            summary = self._register_channel(channel_row, channel)
            if not len(summary):
                continue
            ds = str(summary[self.organoidColumn].iloc[0])
            per_dataset[ds].append(summary)
            dataset_dir[ds] = os.path.dirname(str(summary[self.datasetPathColumn].iloc[0]))

        master_frames: List[pd.DataFrame] = []
        for ds, frames in per_dataset.items():
            master = pd.concat(frames, ignore_index=True)
            # TODO: tier 3 is the ONE index tier that escapes save_dir — dataset_dir[ds] is
            # dirname(source_volume), so the master lands in the raw-data tree while tiers
            # 1, 2 and 4 go under the artefact root. On a read-only source mount this raises
            # PermissionError only AFTER every object has been meshed. Take the directory
            # from artefactDir and keep ds purely as the grouping key.
            master_path = os.path.join(dataset_dir[ds], f"{ds}_objects_master.csv")
            master.to_csv(master_path, index=False)
            logger.info("Wrote dataset master (%d objects) → %s", len(master), master_path)
            master_frames.append(master)

        supermaster = (
            pd.concat(master_frames, ignore_index=True)
            if master_frames else pd.DataFrame(columns=[
                self.SAMPLE_PATH_COLUMN, self.labelColumn, self.channel_column,
                self.datasetPathColumn, self.maskPathColumn,
                self.organoid_stem_column, self.organoidColumn,
            ])
        )
        supermaster.reset_index(drop=True, inplace=True)
        supermaster[self.SAMPLE_LABEL_COLUMN] = supermaster.index
        self.write_index(supermaster, self.dfPath)
        logger.info("Wrote super-master (%d objects) → %s", len(supermaster), self.dfPath)

    # ------------------------------------------------------------------
    # Index building — overrides PointCloudOrganoids
    # ------------------------------------------------------------------

    def _iter_channel_rows(self, dataDF: pd.DataFrame):
        """Yield ``(channel_row, channel)`` for every (organoid, channel) to index.

        Channel rows are resolved exactly as in
        :meth:`PointCloudOrganoids._create_dual_channel_dataset` (group by
        ``_sample_key``; pick the row whose channel flag column == 1). With no channel
        filter, every volume row is treated as a single ``"Nuclei"`` channel.
        """
        groups = self._channel_groups()
        if groups is None:
            dataDF = self.preprocessDataFrame(dataDF)
            for _, organoid_rows in dataDF.groupby(self.datasetPathColumn):
                yield organoid_rows.iloc[0], "Nuclei"
            return

        dataDF = dataDF.copy()
        dataDF["_sample_key"] = dataDF[self.datasetPathColumn].map(self._sample_key)
        for sample_key, sample_rows in dataDF.groupby("_sample_key"):
            channel_rows: dict = {}
            for channel in self.CHANNEL_ALGORITHMS:
                if channel in sample_rows.columns:
                    sel = sample_rows.loc[sample_rows[channel] == 1]
                    if len(sel):
                        channel_rows[channel] = sel.iloc[0]
            for group in groups:
                for channel in group:
                    if channel not in channel_rows:
                        logger.warning(
                            "Sample %s missing channel %s; skipping", sample_key, channel,
                        )
                        continue
                    yield channel_rows[channel], channel

    @staticmethod
    def _mask_to_point_cloud(mask: np.ndarray) -> np.ndarray:
        return PointCloudOrganoids.skeletonization(mask)

    # TODO: no live caller — its only caller was label_volume_to_skeleton_array, which is
    # itself unreachable (see CHANNEL_ALGORITHMS), and the surface path's downsample is
    # commented out. Remove, or reinstate the surface downsample.
    @staticmethod
    def _downsample_to(arr: np.ndarray, n_points: int = 2048) -> np.ndarray:
        """Stride-downsample ``arr`` (P, F) to roughly ``n_points`` rows.

        Mirrors the stride logic used for surface points. Guards against the
        zero-step slice when ``P < n_points`` (returns ``arr`` unchanged).
        """
        if arr.shape[0] <= n_points:
            return arr
        dfactor = max(1, arr.shape[0] // n_points)
        return arr[::dfactor]
    # TODO: dead — CHANNEL_ALGORITHMS maps both channels to skeletonization,
    # so getattr never resolves here (grep confirms no call site anywhere in the repo).
    # It also iterates np.unique(label_volume) WITHOUT skipping label 0, unlike
    # skeletonization, so it would skeletonise the background if reached.
    @staticmethod
    def label_volume_to_skeleton_array(label_volume):
        """Skeletonize a label volume into a downsampled point cloud.

        Returns: (P, 4) array of ``[x, y, z, instance_id]`` skeleton points,
        downsampled to roughly ``n_points`` (xyz order, matching the surface
        array, so both channels share one coordinate convention).
        """
        import numpy as np
        from skimage.morphology import skeletonize
        labels = np.unique(label_volume)

        vols : list[np.ndarray] = []
        for label in labels:
            binary = label_volume == label

            skeleton = skeletonize(binary, method= 'lee')                         # (D, H, W) boolean

            coords = np.argwhere(skeleton).astype(np.float32)      # (P, 3) [z, y, x]

            # Look up the instance ID at each skeleton voxel
            z, y, x = coords.astype(np.int64).T
            ids = label_volume[z, y, x].astype(np.float32)         # (P,)

            xyz = coords[:, [2, 1, 0]]                             # -> [x, y, z]
            vols.append(np.concatenate([xyz, ids[:, None]], axis=-1))      # (P, 4)

        out = np.concatenate(vols, axis=0)

        return PointCloudOrganoids._downsample_to(out, n_points=4000)

    def _build_point_cloud(self, vol_path: str, pc_dir: str) -> str:
        """Read the N5 mask → extract point cloud → save CSV; return CSV path.

        The mask comes from the N5 rather than the raw TIFF, so the volume must already
        be converted (see :meth:`~dl_utils.n5_datasets.BaseVolumeDataset.run`).
        """
        stem = os.path.splitext(os.path.basename(vol_path))[0]
        pc_path = os.path.join(pc_dir, f"{stem}_pointcloud.csv")
        if os.path.exists(pc_path):
            logger.info("Point cloud already exists, skipping: %s", pc_path)
            return pc_path
        mask: np.ndarray = self.maskOfPath(vol_path)
        coords = self._mask_to_point_cloud(mask)
        pd.DataFrame(coords, columns=["z", "y", "x", LABEL_KEY]).to_csv(pc_path, index=False)
        logger.info("Saved point cloud (%d points) → %s", len(coords), pc_path)
        return pc_path


    def createSampleMapperDF(self) -> None:
        dataDF = self.loadDataFrame(manualDFPath).drop_duplicates().copy()
        self._build_supermaster(self._iter_channel_rows(dataDF))

    def createSamplemapperDFfromDatasetDir(self, **kwargs: str) -> None:
        """Single-channel directory scan: each volume is one ``"Nuclei"`` channel."""
        vol_paths: List[str] = sorted(glob.glob(os.path.join(directory, f"*{fileExtension}")))
        mask_paths: List[str] = sorted(glob.glob(os.path.join(directory, "mask", f"*{fileExtension}")))
        assert mask_paths, (
            f"No mask files found. Volumes: {len(vol_paths)}, "
            f"Masks: {len(mask_paths)} in {directory}"
        )
        assert len(vol_paths) == len(mask_paths), (
            f"Volume/mask count mismatch: {len(vol_paths)} vs {len(mask_paths)}"
        )
        rows = [
            (pd.Series({self.datasetPathColumn: v, self.maskPathColumn: m}), "Nuclei")
            for v, m in zip(vol_paths, mask_paths)
        ]
        self._build_supermaster(iter(rows))

    # ------------------------------------------------------------------
    # Data access — read one object's own CSV directly
    # ------------------------------------------------------------------


    def __getstate__(self) -> dict:
        # Don't pickle cached clouds into worker processes; rebuilt per worker.
        # No super() call: the base class no longer defines one (see run()/openKey).
        state = self.__dict__.copy()
        state["instances"] = {}
        return state

