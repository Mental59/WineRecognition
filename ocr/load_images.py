import urllib.request
from tqdm import tqdm

WINE_LABEL_URLS = [
    'https://www.wine-searcher.com/images/labels/67/28/10626728.jpg',
    'https://www.wine-searcher.com/images/labels/74/46/11127446.jpg',
    'https://www.wine-searcher.com/images/labels/07/86/11560786.jpg',
    'https://www.wine-searcher.com/images/labels/79/94/11227994.jpg',
    'https://www.wine-searcher.com/images/labels/17/79/10441779.jpg',
    'https://www.wine-searcher.com/images/labels/43/06/10714306.jpg',
    'https://www.wine-searcher.com/images/labels/69/14/10666914.jpg',
    'https://www.wine-searcher.com/images/labels/03/83/10780383.jpg',
    'https://www.wine-searcher.com/images/labels/74/65/10307465.jpg',
    'https://www.wine-searcher.com/images/labels/35/47/11443547.jpg',
]

WINE_MENU_URLS = [
    'https://1220hslgc7hgrgolf.files.wordpress.com/2012/09/flinders-wine_cp.jpg',
    'https://www.aspleyhornets.com.au/wp-content/uploads/2018/05/Wine-List-Page-2-1024x725.jpg',
    'https://media-cdn.tripadvisor.com/media/photo-s/04/76/54/a4/picasso.jpg',
    'https://images.squarespace-cdn.com/content/v1/602087dd3a23c03b304fcc94/d8e28f42-4fd5-40f3-bab7-924b3e600527/Celebration+of+Australia+Degustation+Diner+-+Final+%281%29.jpg',
    'https://media-cdn.tripadvisor.com/media/photo-s/03/1d/f5/cb/bistro-821.jpg'
]


def main():
    for index, url in enumerate(tqdm(WINE_LABEL_URLS), 1):
        urllib.request.urlretrieve(url, f'data/labels/image_{index}.jpg')

    for index, url in enumerate(tqdm(WINE_MENU_URLS), 1):
        urllib.request.urlretrieve(url, f'data/menus/image_{index}.jpg')


if __name__ == '__main__':
    main()
