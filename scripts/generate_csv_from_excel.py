import os
import sys
import argparse


current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.split(current_dir)[0]
if upper_dir not in sys.path:
    sys.path.insert(0, upper_dir)

from data_master import DataLoader


"""
python .\scripts\generate_csv_from_excel.py --drop-not-add --fill-bottle-size 750.0 --only-completed-rows .\data\excel\Halliday_Wine_AU.xlsx ./data/csv/Halliday_Wine_AU-only_completed_rows.csv
python .\scripts\generate_csv_from_excel.py --drop-not-add --fill-bottle-size 750.0 .\data\excel\Halliday_Wine_AU.xlsx ./data/csv/Halliday_Wine_AU-all_rows.csv
python .\scripts\generate_csv_from_excel.py --drop-not-add --fill-bottle-size 750.0 .\data\excel\WineSearcher_Wine_AU.xlsx ./data/csv/WineSearcher_Wine_AU-all_rows.csv
python .\scripts\generate_csv_from_excel.py --drop-not-add --fill-bottle-size 750.0 --only-completed-rows  .\data\excel\WineSearcher_Wine_AU.xlsx ./data/csv/WineSearcher_Wine_AU-only_completed_rows.csv
"""


parser = argparse.ArgumentParser()
parser.add_argument(
    'path',
    type=str,
    help='Path to input excel file'
)
parser.add_argument(
    'output',
    type=str,
    help='Path where new csv file will be stored'
)
parser.add_argument(
    '--drop-not-add',
    help="Drop columns which names don't start with Add",
    action='store_true'
)
parser.add_argument(
    '--fill-bottle-size',
    type=float,
    help='Fill empty bottle size cells with this float number'
)
parser.add_argument(
    '--only-completed-rows',
    help='Store only rows with True on column IsCompleted',
    action='store_true'
)
args = parser.parse_args()

data = DataLoader.load_excel_data(
    path=args.path,
    fillna=True
)

data = DataLoader.preprocess(
    data,
    fill_bottle_size=args.fill_bottle_size,
    only_completed_rows=args.only_completed_rows,
    drop_not_add_columns=args.drop_not_add
)

data.to_csv(args.output, index=False)
