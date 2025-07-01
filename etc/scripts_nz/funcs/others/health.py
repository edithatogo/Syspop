from pandas import DataFrame

from funcs.preproc import _read_original_csv


def add_mmr(mmr_data_path: str) -> DataFrame:
    return _read_original_csv(mmr_data_path)
