import os
import sys
import json
import pandas as pd
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.split(current_dir)[0]
if upper_dir not in sys.path:
    sys.path.insert(0, upper_dir)
from data_master import DataGenerator, DataLoader, ComplexGenerator


OUTPUT_PATH = r'G:\PythonProjects\WineRecognition2\data\text\exp2_datasets'

data_info = json.load(
    open(os.path.join(upper_dir, 'data_info.json'))
)

halliday = DataLoader.load_csv_data(
    r'G:\PythonProjects\WineRecognition2\data\csv\Halliday_Wine_AU-only_completed_rows.csv'
)
wine_searcher = DataLoader.load_csv_data(
    r'G:\PythonProjects\WineRecognition2\data\csv\WineSearcher_Wine_AU-only_completed_rows.csv'
)

percent = 5
# 5% of rows in wine_searcher
n_rows = wine_searcher.shape[0] * percent / 100
first_n_rows = n_rows
complex_generator = ComplexGenerator(data_info['all_keys_probs'])

for _ in tqdm(range(100 // percent)):
    data = pd.concat(
        [halliday, wine_searcher.head(int(first_n_rows))]
    ).fillna('')

    first_n_rows += n_rows
    with open(os.path.join(OUTPUT_PATH, f'Halliday_WineSearcher_{percent}.txt'), 'w', encoding='utf-8') as file:
        file.write(
            DataGenerator.generate_data_text_complex(
                data=data,
                complex_generator=complex_generator,
                write_labels=True
            )
        )
    percent += 5
