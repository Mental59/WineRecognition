from os import mkdir
from os.path import join, exists
import json
import pandas as pd
from data_master import DataGenerator, DataLoader, ComplexGeneratorMain, ComplexGeneratorMenu, CfgValue

OUTPUT_PATH = r'G:\PythonProjects\WineRecognition2\data\text\halliday_winesearcher_menu_gen_samplesv3'
HALLIDAY_PATH = r'G:\PythonProjects\WineRecognition2\data\csv\Halliday_Wine_AU-only_completed_rows.csv'
WINESEARCHER_PATH = r'G:\PythonProjects\WineRecognition2\data\csv\WineSearcher_Wine_AU-only_completed_rows.csv'
PERCENT = 0.05  # percent of each pattern in final dataset


def split_price(price: str):
    price = price.replace(',', '')
    divided_price = str(int(float(price) / 5))
    return divided_price + '/' + str(int(float(price)))


def main():
    if not exists(OUTPUT_PATH):
        mkdir(OUTPUT_PATH)
    data_info = json.load(open('../data_info.json'))
    halliday = DataLoader.load_csv_data(HALLIDAY_PATH)
    winesearcher = DataLoader.load_csv_data(WINESEARCHER_PATH)
    halliday_winesearcher = pd.concat((halliday, winesearcher))

    cfgs = [
        # 1 menu
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_TradeName'),
            CfgValue('Add_Brand', prob=0.5),
            CfgValue('Add_GeoIndication'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Add_Price')
        ],
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_TradeName'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Add_GeoIndication'),
            CfgValue('Add_Price')
        ],
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_TradeName'),
            CfgValue('Add_KeyWordTrue', prob=0.4),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Add_Price')
        ],
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_TradeName'),
            CfgValue('Add_KeyWordTrue'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Add_Price')
        ],

        # 2 menu
        [
            CfgValue('Add_TradeName'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Add_GeoIndication'),
            CfgValue('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC'], prob=0.625),
            CfgValue('Add_Price', prepr_func=split_price)
        ],
        [
            CfgValue('Add_TradeName'),
            CfgValue('Add_WineType'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Add_GeoIndication'),
            CfgValue('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC'], prob=0.625),
            CfgValue('Add_Price')
        ],
        [
            CfgValue('Add_TradeName'),
            CfgValue('Add_KeyWordTrue'),
            CfgValue('Add_WineColor', prob=0.5),
            CfgValue('Add_GeoIndication'),
            CfgValue('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC'], prob=0.625),
            CfgValue('Add_Price', prepr_func=split_price)
        ],
        [
            CfgValue('Add_TradeName'),
            CfgValue('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC']),
            CfgValue('Add_Sweetness'),
            CfgValue('Add_GeoIndication'),
            CfgValue('Add_Price')
        ],
        [
            CfgValue('Add_TradeName'),
            CfgValue('Add_KeyWordTrue'),
            CfgValue('Add_Sweetness'),
            CfgValue('Add_Price')
        ],

        # 3 menu
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_TradeName'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Add_Price', prepr_func=split_price),
            CfgValue('Add_GeoIndication'),
            CfgValue('Punctuation', values=[',']),
            CfgValue('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC'])
        ],
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_TradeName'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Add_Price', prepr_func=split_price),
            CfgValue('Add_KeyWordTrue'),
            CfgValue('Punctuation', values=[',']),
            CfgValue('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC'])
        ],
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_TradeName'),
            CfgValue('Add_WineColor'),
            CfgValue('Add_Price', prepr_func=split_price),
            CfgValue('Add_KeyWordTrue'),
            CfgValue('Punctuation', values=[',']),
            CfgValue('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC'])
        ],
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_TradeName'),
            CfgValue('Add_KeyWordTrue'),
            CfgValue('Add_Price', prepr_func=split_price),
            CfgValue('Add_GeoIndication'),
            CfgValue('Punctuation', values=[',']),
            CfgValue('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC'])
        ],
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_TradeName'),
            CfgValue('Add_Brand'),
            CfgValue('Add_KeyWordTrue'),
            CfgValue('Add_Price', prepr_func=split_price),
            CfgValue('Add_GeoIndication'),
            CfgValue('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC'])
        ],
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_Brand'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Add_Price', prepr_func=split_price),
            CfgValue('Add_Sweetness'),
            CfgValue('Punctuation', values=[',']),
            CfgValue('Add_KeyWordTrue'),
            CfgValue('Punctuation', values=[',']),
            CfgValue('Other', values=['Locally']),
            CfgValue('Other', values=['Grown'])
        ],


        # 4 menu
        [
            CfgValue('Add_TradeName'),
            CfgValue('Add_GeoIndication'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Punctuation', values=['-']),
            CfgValue('Other', values=['Glass', 'Bottle'])
        ],
        [
            CfgValue('Add_Brand'),
            CfgValue('Add_WineColor'),
            CfgValue('Punctuation', values=['-']),
            CfgValue('Other', values=['Glass', 'Bottle'])
        ],
        [
            CfgValue('Add_TradeName'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Punctuation', values=['-']),
            CfgValue('Other', values=['Glass', 'Bottle'])
        ],
        [
            CfgValue('Add_TradeName'),
            CfgValue('Add_Brand'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Punctuation', values=['-']),
            CfgValue('Other', values=['Glass', 'Bottle'])
        ],

        # 5 menu
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_TradeName'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Punctuation', values=['-']),
            CfgValue('Add_GeoIndication'),
            CfgValue('Punctuation', values=[',']),
            CfgValue('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC']),
            CfgValue('Add_WineColor'),
            CfgValue('Add_BottleSize'),
            CfgValue('Add_BottleSize', values=['mL']),
            CfgValue('Add_Price')
        ],
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Add_TradeName'),
            CfgValue('Punctuation', values=['-']),
            CfgValue('Add_GeoIndication'),
            CfgValue('Add_BottleSize'),
            CfgValue('Add_BottleSize', values=['mL']),
            CfgValue('Add_Price')
        ],
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Add_TradeName'),
            CfgValue('Add_Brand'),
            CfgValue('Punctuation', values=['-']),
            CfgValue('Add_GeoIndication'),
            CfgValue('Punctuation', values=[',']),
            CfgValue('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC']),
            CfgValue('Add_BottleSize'),
            CfgValue('Add_BottleSize', values=['mL']),
            CfgValue('Add_Price')
        ],
        [
            CfgValue('Add_Vintage'),
            CfgValue('Add_TradeName'),
            CfgValue('Add_KeyWordTrue'),
            CfgValue('Punctuation', values=['-']),
            CfgValue('Add_GeoIndication'),
            CfgValue('Punctuation', values=[',']),
            CfgValue('Add_GeoIndication', values=['NV', 'NSW', 'SA', 'VIC']),
            CfgValue('Add_GrapeVarieties'),
            CfgValue('Add_BottleSize'),
            CfgValue('Add_BottleSize', values=['mL']),
            CfgValue('Add_Price')
        ],
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
            column_names = [value.column for value in complex_generator.cfg if value.values is None]
            samples = halliday_winesearcher[column_names].dropna().sample(n_rows)
            final_dataset.append(DataGenerator.generate_data_text_menu(samples, complex_generator))
        file.write(''.join(final_dataset))


if __name__ == '__main__':
    main()
