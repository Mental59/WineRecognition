import os
from typing import Dict
import pandas as pd


class DataSaver:
    """class for saving generated data structures"""

    @staticmethod
    def save_frequency_dictionary(frequency_dictionary: Dict, excel_path, csv_folder):
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            for key in frequency_dictionary:
                frequency_dictionary[key].sort_values(by='count', inplace=True, ascending=False)
                frequency_dictionary[key].to_excel(writer, sheet_name=key)
                frequency_dictionary[key].to_csv(os.path.join(csv_folder, f'{key}.csv'), index=False)
