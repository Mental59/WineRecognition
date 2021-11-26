import os
import json
from data_master import DataGenerator, DataLoader, DataSaver


config = json.load(open('generate_freq_dict_config.json'))

freq_dict = DataGenerator.generate_freq_dict(
    *[DataLoader.load_csv_data(path) for path in config['data_paths']],
    keys=json.load(open(config['data_info_path']))['keys']['all'],
    byword=config['byword']
)

if not os.path.exists(config['output_csv_folder']):
    os.mkdir(config['output_csv_folder'])

DataSaver.save_frequency_dictionary(
    frequency_dictionary=freq_dict,
    excel_path=config['output_excel_path'],
    csv_folder=config['output_csv_folder']
)
