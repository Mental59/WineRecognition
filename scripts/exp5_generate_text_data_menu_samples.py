from os import mkdir
from os.path import join, exists
import json
import pandas as pd
from data_master import DataGenerator, DataLoader, ComplexGeneratorMain, ComplexGeneratorMenu

OUTPUT_PATH = r'G:\PythonProjects\WineRecognition2\data\text\halliday_winesearcher_menu_gen_samples'
HALLIDAY_PATH = r'G:\PythonProjects\WineRecognition2\data\csv\Halliday_Wine_AU-only_completed_rows.csv'
WINESEARCHER_PATH = r'G:\PythonProjects\WineRecognition2\data\csv\WineSearcher_Wine_AU-only_completed_rows.csv'
PERCENT = 0.05  # percent of each menu in final dataset


def split_price(price: str):
    price = price.replace(',', '')
    divided_price = str(int(float(price) / 5))
    return divided_price + '/' + str(int(float(price)))


def cfg_value(column, prob=1.0, prepr_func=None, values=None):
    return column, prob, prepr_func, values


def main():
    if not exists(OUTPUT_PATH):
        mkdir(OUTPUT_PATH)
    data_info = json.load(open('../data_info.json'))
    halliday = DataLoader.load_csv_data(HALLIDAY_PATH)
    winesearcher = DataLoader.load_csv_data(WINESEARCHER_PATH)
    halliday_winesearcher = pd.concat((halliday, winesearcher))

    cfgs = [
        [
            cfg_value('Add_Vintage'),
            cfg_value('Add_TradeName'),
            cfg_value('Add_GeoIndication'),
            cfg_value('Add_GrapeVarieties'),
            cfg_value('Add_Price')
        ],
        [
            cfg_value('Add_TradeName'),
            cfg_value('Add_GrapeVarieties'),
            cfg_value('Add_GeoIndication'),
            cfg_value('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC']),
            cfg_value('Add_Price', prepr_func=split_price)
        ],
        [
            cfg_value('Add_Vintage'),
            cfg_value('Add_TradeName'),
            cfg_value('Add_GrapeVarieties'),
            cfg_value('Add_Price', prepr_func=split_price),
            cfg_value('Add_Sweetness', 0.5),
            cfg_value('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC'])
        ],
        [
            cfg_value('Add_TradeName'),
            cfg_value('Add_GrapeVarieties'),
            cfg_value('Punctuation', values=['-']),
            cfg_value('Add_KeyWordFalse', values=['Glass', 'Bottle'])
        ],
        [
            cfg_value('Add_Vintage'),
            cfg_value('Add_TradeName'),
            cfg_value('Add_WineType'),
            cfg_value('Add_GeoIndication'),
            cfg_value('Add_GeoIndication', 0.6, values=['NV', 'NSW', 'SA', 'VIC']),
            cfg_value('Add_WineColor', 0.5),
            cfg_value('Add_BottleSize'),
            cfg_value('Add_BottleSize', values=['mL']),
            cfg_value('Add_Price')

        ]
    ]

    complex_generators = {
        'main': ComplexGeneratorMain(data_info['all_keys_probs']),
        'menu': [ComplexGeneratorMenu(cfg) for cfg in cfgs]
    }

    n_rows = int(len(halliday_winesearcher) * PERCENT)

    with open(join(OUTPUT_PATH, 'Halliday_WineSearcher_MenuGenSamples.txt'), 'w', encoding='utf-8') as file:
        final_dataset = [
            DataGenerator.generate_data_text_complex(halliday_winesearcher, complex_generators['main'])
        ]
        for complex_generator in complex_generators['menu']:
            # Items in cfg are Tuples of (column, prob, prepr_func, values)
            column_names = [val[0] for val in complex_generator.cfg if val[3] is None]
            samples = halliday_winesearcher[column_names].dropna().sample(n_rows)
            final_dataset.append(DataGenerator.generate_data_text_menu(samples, complex_generator))
        file.write(''.join(final_dataset))


if __name__ == '__main__':
    main()
