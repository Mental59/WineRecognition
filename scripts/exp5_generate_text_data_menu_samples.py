from os import mkdir
from os.path import join, exists
import json
import pandas as pd
from data_master import DataGenerator, DataLoader, ComplexGeneratorMain, ComplexGeneratorMenu, CfgValue

OUTPUT_PATH = r'G:\PythonProjects\WineRecognition2\data\text\data_and_menu_gen_samples'
OUTPUT_NAME = 'Halliday_WineSearcher_Bruxelles_MenuGenSamples.txt'
DATA_PATHS = [
    r'G:\PythonProjects\WineRecognition2\data\csv\Halliday_Wine_AU-only_completed_rows.csv',
    r'G:\PythonProjects\WineRecognition2\data\csv\WineSearcher_Wine_AU-only_completed_rows.csv',
    r'G:\PythonProjects\WineRecognition2\data\csv\Bruxelles_Wine_ES.csv'
]
PERCENT = 0.15  # percent of each pattern in final dataset


def main():
    def split_price(price: str):
        price = price.replace(',', '')
        divided_price = str(int(float(price) / 5))
        return divided_price + '/' + str(int(float(price)))

    if not exists(OUTPUT_PATH):
        mkdir(OUTPUT_PATH)
    data_info = json.load(open('../data_info.json'))
    data_sources = [DataLoader.load_csv_data(data_path) for data_path in DATA_PATHS]
    df = pd.concat(data_sources)

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
            CfgValue('Add_Brand'),
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

    n_rows = int(len(df) * PERCENT)
    total_number_of_lines = len(df) + n_rows * len(cfgs)

    with open(join(OUTPUT_PATH, OUTPUT_NAME), 'w', encoding='utf-8') as file:
        final_dataset = [
            DataGenerator.generate_data_text_complex(df, complex_generators['main'])
        ]
        for complex_generator in complex_generators['menu']:
            column_names = [value.column for value in complex_generator.cfg if value.values is None]
            samples = df[column_names].dropna().sample(n_rows)
            final_dataset.append(DataGenerator.generate_data_text_menu(samples, complex_generator))
        file.write(''.join(final_dataset))

    print(f'Total number of lines: {total_number_of_lines}')


if __name__ == '__main__':
    main()
