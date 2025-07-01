from pandas import DataFrame

from funcs.preproc import _read_original_csv


def add_birthplace(birthplace_path: str) -> DataFrame:
    return _read_original_csv(birthplace_path)
