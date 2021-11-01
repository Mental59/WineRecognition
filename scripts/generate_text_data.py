import os
import sys
import json
import argparse


"""
python .\scripts\generate_text_data.py -c -l .\data\csv\Halliday_Wine_AU-only_completed_rows-all_columns.csv ./data/text/Halliday_Wine_AU-only_completed_rows-complex.txt
python .\scripts\generate_text_data.py -l -k all .\data\csv\Halliday_Wine_AU-only_completed_rows-all_columns.csv ./data/text/Halliday_Wine_AU-only_completed_rows-all_keys.txt
python .\scripts\generate_text_data.py -l -k all .\data\csv\Halliday_Wine_AU-all_rows-all_columns.csv ./data/text/Halliday_Wine_AU-all_rows-all_keys.txt
python .\scripts\generate_text_data.py -l -k all .\data\csv\WineSearcher_Wine_AU-all_rows-all_columns.csv ./data/text/WineSearcher_Wine_AU-all_rows-all_keys.txt
python .\scripts\generate_text_data.py -l -k origin_full_name .\data\csv\WineSearcher_Wine_AU-all_rows-all_columns.csv ./data/text/WineSearcher_Wine_AU-all_rows-origin_fullname.txt
python .\scripts\generate_text_data.py -l -k without_add .\data\csv\WineSearcher_Wine_AU-all_rows-all_columns.csv ./data/text/WineSearcher_Wine_AU-all_rows-without_add.txt
"""


current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.split(current_dir)[0]
if upper_dir not in sys.path:
    sys.path.insert(0, upper_dir)


from data_master import DataGenerator, DataLoader, ComplexGenerator


parser = argparse.ArgumentParser()
parser.add_argument(
    'input',
    type=str,
    help='Path to input csv data file'
)
parser.add_argument(
    'output',
    type=str,
    help='Path where new txt file will be stored'
)
parser.add_argument(
    '-k', '--keys',
    type=str,
    help='Which key set to use, might be one of this: "main", "all", "withoud_add", "origin_full_name"'
         ' (for more details see file data_info.json)'
)
parser.add_argument(
    '-c', '--complex',
    action='store_true',
    help='Use complex generator'
)
parser.add_argument(
    '-l', '--labels',
    action='store_true',
    help='Write labels in output file'
)
args = parser.parse_args()

data_info = json.load(
    open(os.path.join(upper_dir, 'data_info.json'))
)

data = DataLoader.load_csv_data(args.input)

with open(args.output, 'w', encoding='utf-8') as file:

    if args.complex:
        file.write(
            DataGenerator.generate_data_text_complex(
                data=data,
                complex_generator=ComplexGenerator(data_info['all_keys_probs']),
                write_labels=args.labels
            )
        )
    else:
        file.write(
            DataGenerator.generate_data_text(
                data=data,
                keys=data_info['keys'][args.keys],
                write_labels=args.labels
            )
        )
