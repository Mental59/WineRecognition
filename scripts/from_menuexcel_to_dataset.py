import pandas as pd
import numpy as np

OUTPUT_PATH = r'G:\PythonProjects\WineRecognition2\data\text\menu_txt_tagged.txt'
MENU_PATH = r'G:\PythonProjects\WineRecognition2\data\excel\Wine_test.xlsx'


def add_white_spaces_between_punctuation(s: str):
    chars = list(s)
    indices = np.array([i for i, c in enumerate(chars) if c in ',;'])
    for i in indices:
        left = next(iter(chars[i-1:i]), '')
        right = next(iter(chars[i + 1:i + 2]), '')
        if left != ' ':
            chars.insert(i, ' ')
            indices += 1
            i += 1
        if right != ' ':
            chars.insert(i + 1, ' ')
            indices += 1

    return ''.join(chars)


def find_columns_with_value(series, value):
    return list(series.where(series.str.contains(value)).dropna().index)


def main():
    df = pd.read_excel(MENU_PATH, dtype=str)

    source_strings = df['originFullName'].values
    columns = [
        'Add_TradeName',
        'Add_Brand',
        'Add_WineType',
        'Add_Sweetness',
        'Add_Vintage',
        'Add_GeoIndication',
        'Add_GrapeVarieties',
        'Add_BottleSize',
        'Add_ClosureType',
        'Add_Price',
        'Add_KeyWordTrue',
        'Other'
    ]

    df.rename(
        columns={
            'add_tradeName': 'Add_TradeName',
            'add_brand': 'Add_Brand',
            'add_wineType': 'Add_WineType',
            'add_wineColor': 'Add_WineColor',
            'add_sweetness': 'Add_Sweetness',
            'add_vintage': 'Add_Vintage',
            'add_geoIndication': 'Add_GeoIndication',
            'add_grapeVarieties': 'Add_GrapeVarieties',
            'add_bottleSize': 'Add_BottleSize',
            'add_closureType': 'Add_ClosureType',
            'add_price': 'Add_Price',
            'add_keywordTrue': 'Add_KeywordTrue',
            'other': 'Other'
        },
        inplace=True
    )

    df = df[columns]

    dataset = []
    for index, sentence in enumerate(source_strings):
        sentence = add_white_spaces_between_punctuation(sentence.strip())
        sent = []
        for word in sentence.split():
            possible_tags = find_columns_with_value(df.iloc[index], word)
            tag = 'UNKNOWN' if not possible_tags else possible_tags[0]
            sent.append(f'{word} {tag}')
        dataset.append('\n'.join(sent) + '\n')

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as file:
        file.write('\n'.join(dataset))


if __name__ == '__main__':
    #main()
    print(1)
