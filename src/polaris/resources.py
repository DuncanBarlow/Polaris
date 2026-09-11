from __future__ import annotations

from contextlib import contextmanager
from importlib import resources
from pathlib import Path


@contextmanager
def ifriit_run_files_path():
    resource = resources.files("polaris").joinpath("ifriit_run_files")
    with resources.as_file(resource) as path:
        yield Path(path)


@contextmanager
def facility_config_files_path():
    resource = resources.files("polaris").joinpath("facility_config_files")
    with resources.as_file(resource) as path:
        yield Path(path)


def get_ifriit_run_files_root() -> Path:
    return Path(__file__).resolve().parent / "ifriit_run_files"


def get_facility_config_files_root() -> Path:
    return Path(__file__).resolve().parent / "facility_config_files"
