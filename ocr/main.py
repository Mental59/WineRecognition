import json
import os
from glob import glob
from tqdm import tqdm
from google.cloud import vision


def detect_text(client, path):
    with open(path, 'rb') as file:
        content = file.read()
    response = client.text_detection(image=vision.Image(content=content))
    texts = response.text_annotations
    return texts


def save_detected_text(text_annotations, path: str):
    texts = [text.description for text in text_annotations]
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(texts, file)


def detect_text_in(client, input_path, output_path):
    for image_path in tqdm(glob(os.path.join(input_path, '*')), desc=f'Detecting text in {input_path}'):
        filename = os.path.splitext(os.path.basename(image_path))[0]
        text_annotations = detect_text(client, image_path)
        save_detected_text(text_annotations, os.path.join(output_path, f'{filename}_results.json'))


def main():
    client = vision.ImageAnnotatorClient()
    detect_text_in(client, input_path='data/labels', output_path='data/results/labels')
    detect_text_in(client, input_path='data/menus', output_path='data/results/menus')


if __name__ == '__main__':
    main()
