from os import mkdir
from os.path import join, exists
import json
import pandas as pd
from data_master import DataGenerator, DataLoader, ComplexGeneratorMain, ComplexGeneratorMenu, CfgValue

OUTPUT_PATH = r'G:\PythonProjects\WineRecognition2\data\text\data_and_menu_gen_samples'
OUTPUT_NAME = 'Halliday_WineSearcher_Bruxelles_MenuGenSamples_v2.txt'
DATA_PATHS = [
    r'G:\PythonProjects\WineRecognition2\data\csv\Halliday_Wine_AU-only_completed_rows.csv',
    r'G:\PythonProjects\WineRecognition2\data\csv\WineSearcher_Wine_AU-only_completed_rows.csv',
    r'G:\PythonProjects\WineRecognition2\data\csv\Bruxelles_Wine_ES.csv'
]
PERCENT = 0.10  # percent of each pattern in final dataset

# possible labels
TRADENAME = 'Add_TradeName'
BRAND = 'Add_Brand'
KEYWORD_TRUE = 'Add_KeyWordTrue'
KEYWORD_FALSE = 'Add_KeyWordFalse'
GRAPE = 'Add_GrapeVarieties'
GEO = 'Add_GeoIndication'
TYPE = 'Add_WineType'
BOTTLESIZE = 'Add_BottleSize'
SWEETNESS = 'Add_Sweetness'
COLOR = 'Add_WineColor'
CLOSURE = 'Add_ClosureType'
CERTIFICATE = 'Add_Certificate'
VINTAGE = 'Add_Vintage'
PRICE = 'Add_Price'
PUNCTUATION = 'Punctuation'
OTHER = 'Other'


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
            CfgValue(VINTAGE),
            CfgValue(TRADENAME),
            CfgValue(BRAND, prob=0.5),
            CfgValue(GEO),
            CfgValue(GRAPE),
            CfgValue(PRICE)
        ],
        [
            CfgValue(VINTAGE),
            CfgValue(TRADENAME),
            CfgValue(GRAPE),
            CfgValue(GEO),
            CfgValue(PRICE)
        ],
        [
            CfgValue(VINTAGE),
            CfgValue(TRADENAME),
            CfgValue(KEYWORD_TRUE, prob=0.4),
            CfgValue(GRAPE),
            CfgValue(PRICE)
        ],
        [
            CfgValue(VINTAGE),
            CfgValue(TRADENAME),
            CfgValue(KEYWORD_TRUE),
            CfgValue(GRAPE),
            CfgValue(PRICE)
        ],

        # 2 menu
        [
            CfgValue(TRADENAME),
            CfgValue(GRAPE),
            CfgValue(GEO),
            CfgValue(GEO, values=['NV', 'NSW', 'SA', 'VIC'], prob=0.625),
            CfgValue(PRICE, prepr_func=split_price)
        ],
        [
            CfgValue(TRADENAME),
            CfgValue(TYPE),
            CfgValue(GRAPE),
            CfgValue(GEO),
            CfgValue(GEO, values=['NV', 'NSW', 'SA', 'VIC'], prob=0.625),
            CfgValue(PRICE)
        ],
        [
            CfgValue(TRADENAME),
            CfgValue(KEYWORD_TRUE),
            CfgValue(COLOR, prob=0.5),
            CfgValue(GEO),
            CfgValue(GEO, values=['NV', 'NSW', 'SA', 'VIC'], prob=0.625),
            CfgValue(PRICE, prepr_func=split_price)
        ],
        [
            CfgValue(TRADENAME),
            CfgValue(GEO, values=['NV', 'NSW', 'SA', 'VIC']),
            CfgValue(SWEETNESS),
            CfgValue(GEO),
            CfgValue(PRICE)
        ],
        [
            CfgValue(TRADENAME),
            CfgValue(KEYWORD_TRUE),
            CfgValue(SWEETNESS),
            CfgValue(PRICE)
        ],

        # 3 menu
        [
            CfgValue(VINTAGE),
            CfgValue(TRADENAME),
            CfgValue(GRAPE),
            CfgValue(PRICE, prepr_func=split_price),
            CfgValue(GEO),
            CfgValue(PUNCTUATION, values=[',']),
            CfgValue(GEO, values=['NV', 'NSW', 'SA', 'VIC'])
        ],
        [
            CfgValue(VINTAGE),
            CfgValue(TRADENAME),
            CfgValue(GRAPE),
            CfgValue(PRICE, prepr_func=split_price),
            CfgValue(KEYWORD_TRUE),
            CfgValue(PUNCTUATION, values=[',']),
            CfgValue(GEO, values=['NV', 'NSW', 'SA', 'VIC'])
        ],
        [
            CfgValue(VINTAGE),
            CfgValue(TRADENAME),
            CfgValue(COLOR),
            CfgValue(PRICE, prepr_func=split_price),
            CfgValue(KEYWORD_TRUE),
            CfgValue(PUNCTUATION, values=[',']),
            CfgValue(GEO, values=['NV', 'NSW', 'SA', 'VIC'])
        ],
        [
            CfgValue(VINTAGE),
            CfgValue(TRADENAME),
            CfgValue(KEYWORD_TRUE),
            CfgValue(PRICE, prepr_func=split_price),
            CfgValue(GEO),
            CfgValue(PUNCTUATION, values=[',']),
            CfgValue(GEO, values=['NV', 'NSW', 'SA', 'VIC'])
        ],
        [
            CfgValue(VINTAGE),
            CfgValue(TRADENAME),
            CfgValue(BRAND),
            CfgValue(KEYWORD_TRUE),
            CfgValue(PRICE, prepr_func=split_price),
            CfgValue(GEO),
            CfgValue(GEO, values=['NV', 'NSW', 'SA', 'VIC'])
        ],
        [
            CfgValue(VINTAGE),
            CfgValue(BRAND),
            CfgValue(GRAPE),
            CfgValue(PRICE, prepr_func=split_price),
            CfgValue(SWEETNESS),
            CfgValue(PUNCTUATION, values=[',']),
            CfgValue(KEYWORD_TRUE),
            CfgValue(PUNCTUATION, values=[',']),
            CfgValue(OTHER, values=['Locally']),
            CfgValue(OTHER, values=['Grown'])
        ],


        # 4 menu
        [
            CfgValue(TRADENAME),
            CfgValue(GEO),
            CfgValue(GRAPE),
            CfgValue(PUNCTUATION, values=['-']),
            CfgValue(OTHER, values=['Glass', 'Bottle'])
        ],
        [
            CfgValue(BRAND),
            CfgValue(COLOR),
            CfgValue(PUNCTUATION, values=['-']),
            CfgValue(OTHER, values=['Glass', 'Bottle'])
        ],
        [
            CfgValue(BRAND),
            CfgValue(GRAPE),
            CfgValue(PUNCTUATION, values=['-']),
            CfgValue(OTHER, values=['Glass', 'Bottle'])
        ],
        [
            CfgValue(TRADENAME),
            CfgValue(BRAND),
            CfgValue(GRAPE),
            CfgValue(PUNCTUATION, values=['-']),
            CfgValue(OTHER, values=['Glass', 'Bottle'])
        ],

        # 5 menu
        [
            CfgValue(VINTAGE),
            CfgValue(TRADENAME),
            CfgValue(GRAPE),
            CfgValue(PUNCTUATION, values=['-']),
            CfgValue(GEO),
            CfgValue(PUNCTUATION, values=[',']),
            CfgValue(GEO, values=['NV', 'NSW', 'SA', 'VIC']),
            CfgValue(COLOR),
            CfgValue(BOTTLESIZE),
            CfgValue(BOTTLESIZE, values=['mL']),
            CfgValue(PRICE)
        ],
        [
            CfgValue(VINTAGE),
            CfgValue(GRAPE),
            CfgValue(TRADENAME),
            CfgValue(PUNCTUATION, values=['-']),
            CfgValue(GEO),
            CfgValue(BOTTLESIZE),
            CfgValue(BOTTLESIZE, values=['mL']),
            CfgValue(PRICE)
        ],
        [
            CfgValue(VINTAGE),
            CfgValue(GRAPE),
            CfgValue(TRADENAME),
            CfgValue(BRAND),
            CfgValue(PUNCTUATION, values=['-']),
            CfgValue(GEO),
            CfgValue(PUNCTUATION, values=[',']),
            CfgValue(GEO, values=['NV', 'NSW', 'SA', 'VIC']),
            CfgValue(BOTTLESIZE),
            CfgValue(BOTTLESIZE, values=['mL']),
            CfgValue(PRICE)
        ],
        [
            CfgValue(VINTAGE),
            CfgValue(TRADENAME),
            CfgValue(KEYWORD_TRUE),
            CfgValue(PUNCTUATION, values=['-']),
            CfgValue(GEO),
            CfgValue(PUNCTUATION, values=[',']),
            CfgValue(GEO, values=['NV', 'NSW', 'SA', 'VIC']),
            CfgValue(GRAPE),
            CfgValue(BOTTLESIZE),
            CfgValue(BOTTLESIZE, values=['mL']),
            CfgValue(PRICE)
        ],

        # NEW MENUS

        # England
        # 1
        [
            CfgValue(GRAPE),
            CfgValue(OTHER, values=['-']),
            CfgValue(OTHER, values=['Bottle']),
            CfgValue(OTHER, values=['-']),
            CfgValue(OTHER, values=['£']),
            CfgValue(PRICE)
        ],
        # 2
        [
            CfgValue(GRAPE),
            CfgValue(COLOR),
            CfgValue(BOTTLESIZE),
            CfgValue(OTHER, values=['ml']),
            CfgValue(PRICE)
        ],
        [
            CfgValue(GRAPE),
            CfgValue(COLOR),
            CfgValue(OTHER, values=['bottle']),
            CfgValue(PRICE)
        ],

        # USA
        # 1
        [
            CfgValue(BRAND),
            CfgValue(GRAPE),
            CfgValue(OTHER, values=['$']),
            CfgValue(PRICE)
        ],
        # 2
        [
            CfgValue(BRAND),
            CfgValue(GRAPE),
            CfgValue(GEO),
            CfgValue(PRICE)
        ],
        # [
        #     CfgValue(BRAND),
        #     CfgValue(PRICE)
        # ],
        [
            CfgValue(BRAND),
            CfgValue(GEO),
            CfgValue(PRICE)
        ],
        [
            CfgValue(TRADENAME),
            CfgValue(SWEETNESS),
            CfgValue(TYPE, prob=0.1),
            CfgValue(GEO),
            CfgValue(PRICE)
        ],
        [
            CfgValue(TRADENAME),
            CfgValue(TYPE),
            CfgValue(COLOR),
            CfgValue(BRAND),
            CfgValue(GEO),
            CfgValue(PRICE)
        ],

        # Canada
        # 1
        [
            CfgValue(BRAND),
            CfgValue(GRAPE),
            CfgValue(PUNCTUATION, values=['(']),
            CfgValue(BOTTLESIZE),
            CfgValue(PUNCTUATION, values=[')']),
            CfgValue(PUNCTUATION, values=['|']),
            CfgValue(OTHER, values=['$']),
            CfgValue(PRICE)
        ],

        # 2
        [
            CfgValue(VINTAGE),
            CfgValue(GRAPE),
            CfgValue(TYPE),
            CfgValue(PRICE)
        ],
        [
            CfgValue(VINTAGE),
            CfgValue(BRAND),
            CfgValue(GRAPE),
            CfgValue(PRICE)
        ],
        [
            CfgValue(VINTAGE),
            CfgValue(BRAND),
            CfgValue(PRICE)
        ],
        [
            CfgValue(BRAND),
            CfgValue(TYPE),
            CfgValue(SWEETNESS),
            CfgValue(PRICE)
        ]
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
