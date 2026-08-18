from collections import deque
from functools import cached_property, partial
from logging import Logger
from pathlib import Path
from typing import Any, Generator, Iterable, Union

import pandas as pd
import skimage.io
from plyfile import PlyData

import data_loader.data_loader as data_loader
from data_loader import DataLoader, Extensions, DLoaderException
from data_loader.data_loader import _SpecialDictRepr


def load_ply(path):
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
    return plydata


class ExtensionsBioimage(Extensions):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def defaults(self) -> dict:
        """
        Returns a dictionary with default file extensions and their loader methods,
        extending the base defaults with a ``.ply`` loader.

        #### Returns:
            - `dict`: A dictionary with default file extensions and their loader methods.
        """
        # ``Extensions.defaults`` is a property; access (don't call) it, then merge.
        return {**super().defaults, "ply": self._ext_tuple("ply", load_ply)}


class SampleLoaderBioImage(DataLoader):
    def __init__(self, path, total_workers=1, **kwargs):
        super().__init__(path=path, total_workers=total_workers, **kwargs)
        self.path = path
        self._file = None

    def _execute_path(self):
        try:
            files = self.base_executor(
                self._load_file, self._get_file, total_workers=self._total_workers
            )
        except Exception as error:
            raise DLoaderException(error)
        return (
            (f.name if not self._full_posix else f.as_posix(), v)
            for f, v in files
            if v is not None
        )

    @classmethod
    def dataObject(cls, path, **kwargs) -> "SampleLoaderBioImage":
        # `data_loader.Extensions.customize` expects each ext_loaders value to be a
        # nested dict {<callable>: {param: value}} (it pulls the callable via
        # next(iter(...)) then introspects its signature). PALMTracer .txt files are
        # tab-delimited with a 2-line metadata header before the real column header,
        # so skip those two rows.
        return cls(path, generator=False, full_posix=False,
                   ext_loaders={"txt": {pd.read_csv: {"sep": "\t", "skiprows": 2, "header": 0}},
                                "json": {pd.read_json: {}},
                                "csv": {pd.read_csv: {}},
                                "tiff": {skimage.io.imread: {}},
                                "Tiff": {skimage.io.imread: {}},
                                "TIFF": {skimage.io.imread: {}},
                                "tif": {skimage.io.imread: {}}}, **kwargs)

    @classmethod
    def loadData(cls, path) -> Any:
        files: _SpecialDictRepr = cls.dataObject(path, total_workers=1).file
        # `path` resolves to a single file, so there is exactly one entry.
        output = next(iter(files.values()), None)
        assert output is not None, f"The loaded data is of None type for path: {path!r}"
        return output

    @cached_property
    def file(self):
        """
        Get loaded files as a dictionary or generator, depending on the `generator` parameter.

        #### Returns:
            - `_SpecialDictRepr` or `_SpecialGenRepr`: Loaded files.
        """
        if self._file is None:
            self._file = self._execute_path()
        return self._repr_files(self._file)

    @cached_property
    def _get_file(self) -> Union[Generator[Any, None, None], Any]:
        return self.get_file(
            self._path,
            self.all_exts["empty"].suffix_
            if not self._default_exts
            else self._validate_exts(self._default_exts),
            self._verbose,
        )

    @classmethod
    def get_file(
        cls,
        directory: data_loader.P,
        defaults: Iterable = "",
        startswith: str = "",
        verbose: bool = False,
        files_only: bool = True,
        generator: bool = True,
        log: Logger = None,
    ) -> Generator:
        """
        Get files from a directory based on default extensions.

        #### Args:
            - `directory` (Path): Directory path.
            - `defaults` (Iterable): Default file extensions to be processed.
            - `startswith` (str): File name prefix to filter files.
            - `verbose` (bool): Indicates whether to display verbose output.

        #### Returns:
            - `Generator`: Files generator.
        """
        # `directory` is actually a single validated file path in every current
        # caller (loadData is called per-file), so we yield it directly. The check
        # guards against a directory being passed by mistake.
        file_path = Path(directory)
        assert not file_path.is_dir(), \
            f"Expected a file path, got a directory: {file_path!r}"

        validated_files = (fp for fp in (file_path,))

        return validated_files if generator else deque(validated_files)
