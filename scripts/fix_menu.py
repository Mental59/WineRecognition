MENU_PATH = r'G:\PythonProjects\WineRecognition2\data\text\menu_txt_tagged.txt'
OUTPUT_PATH = r'G:\PythonProjects\WineRecognition2\data\text\menu_txt_tagged_fixed.txt'

with open(OUTPUT_PATH, 'w', encoding='utf-8') as file:
    for line in open(MENU_PATH, encoding='utf-8'):
        line = line.rstrip().replace('–', '-')
        print(line, file=file)
