from typing import NamedTuple, Any, Tuple, List, Optional, Literal, Union
from data_loader.data_loader import _SpecialDictRepr, _SpecialGenRepr
import numpy as np
from plyfile import PlyData
import os
import pandas as pd
from data_loader import DataLoader, Extensions, DLoaderException

import pickle
import json
from functools import cached_property, partial, wraps
from collections.abc import Mapping
import json
import os
import pandas as pd
import pickle
from collections import deque, namedtuple
from functools import cached_property, partial, wraps
from logging import Logger
from typing import Any, Generator, Iterable, Iterator, NamedTuple, TypeVar, Union
import data_loader.data_loader as data_loader
from pathlib import Path


def load_ply(path):
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
    return plydata


class ExtensionsBioimage(Extensions):
    def __init__(**kwargs):
        super().__init__(**kwargs)

    @property
    def defaults(self) -> dict:
        """
        Returns a dictionary with default file extensions and their loader methods.

        #### Returns:
            - `dict`: A dictionary with default file extensions and their loader methods.
        """
        methods_dict = self.super().defaults()
        my_methods = {"ply": self._ext_tuple("ply", load_ply)}
        return methods_dict.update(my_methods)


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
                   ext_loaders={"txt": {pd.read_csv: {"sep": "\t", "skiprows": 2, "header": 0}}}, **kwargs)

    @classmethod
    def loadData(cls, path) -> Any:
        dloader = cls.dataObject(path, total_workers=1)
        files: _SpecialDictRepr = dloader.file
        del dloader
        output = None
        for k, v in files.items():
            output = v

        assert output is not None, "The loaded data is of None type"

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
        validate_file = partial(cls._validate_file, verbose=verbose)
        directory: Path = validate_file(directory)
        no_dirs = lambda p: p.is_file() and not p.is_dir()
        ext_pattern = partial(cls._compiler, defaults, escape_k=False)
        filter_files = lambda fp: all(
            (
                no_dirs(fp) if files_only else True,
                ext_pattern(cls._rm_period(fp.suffix)),
                validate_file(fp),
            )
        )

        assert not directory.is_dir(), "Is not a directory"

        validated_files = (f for f in (directory,))

        return deque(validated_files) if not generator else validated_files
