import os
from typing import Dict
import pandas as pd


class DataLoader:
    """class for loading data with needed format"""

    @staticmethod
    def load_frequency_dictionary(path: str, to_lowercase=False) -> Dict:
        freq_dict = {}

        for filename in os.listdir(path):
            key = os.path.splitext(filename)[0]
            freq_dict[key] = pd.read_csv(f'{os.path.join(path, filename)}', dtype=object)
            freq_dict[key].fillna('', inplace=True)
            if to_lowercase:
                freq_dict[key]['value'] = freq_dict[key]['value'].str.lower()

        return freq_dict

    @staticmethod
    def load_csv_data(path: str, fillna=True) -> pd.DataFrame:
        data = pd.read_csv(path, dtype=object)

        if fillna:
            data.fillna('', inplace=True)

        return data

    @staticmethod
    def load_excel_data(path: str, fillna=True) -> pd.DataFrame:
        data = pd.read_excel(path)

        if fillna:
            data.fillna('', inplace=True)

        return data

    @staticmethod
    def preprocess(
            data: pd.DataFrame,
            fill_bottle_size: float = None,
            only_completed_rows=False,
            drop_not_add_columns=False
    ) -> pd.DataFrame:

        new_data = data.copy()

        if fill_bottle_size:
            new_data.loc[new_data['Add_BottleSize'] == '', 'Add_BottleSize'] = fill_bottle_size

        if only_completed_rows:
            new_data = new_data.loc[new_data['IsCompleted'] == True]

        if drop_not_add_columns:
            new_data.drop(
                columns=[col for col in new_data.columns if not col.startswith('Add')],
                inplace=True
            )

        new_data = new_data.applymap(
            lambda x: ''.join(
                char for char in str(x).replace('} {', '; ') if char not in '{}[]'
            )
        )

        vintage = 'Add_Vintage'
        if vintage in new_data.columns:
            new_data.loc[(new_data[vintage] == 'Non Vintage') | (new_data[vintage] == 'Nv'), vintage] = 'nv'

        return new_data
