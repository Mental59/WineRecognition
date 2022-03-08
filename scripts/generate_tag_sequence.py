from data_master import DataGenerator

data_path = r'G:\PythonProjects\WineRecognition2\data\text\menu_txt_tagged.txt'

with open(data_path, encoding='utf-8') as file:
    data = DataGenerator.generate_sents2(file.read().split('\n'))

with open(r'G:\PythonProjects\WineRecognition2\data\menus\WinesTagSequence.txt', 'w', encoding='utf-8') as file:
    for sentence, tags in data:
        if sentence[0] in ['NV', 'NSW', 'SA', 'VIC']:
            tags[0] = 'NV_NSW_SA_VIC'
        elif '/' in sentence[0]:
            tags[0] = 'FractionalPrice'
        elif sentence[0] in ['Glass', 'Bottle']:
            tags[0] = 'GlassBottle'
        elif sentence[0] in ['mL']:
            tags[0] = 'mL'
        current_tag = tags[0]
        tag_sequence = [current_tag]
        for tag, word in zip(tags[1:], sentence[1:]):
            if word in ['NV', 'NSW', 'SA', 'VIC']:
                tag = 'NV_NSW_SA_VIC'
            elif '/' in word:
                tag = 'FractionalPrice'
            elif word in ['Glass', 'Bottle']:
                tag = 'GlassBottle'
            elif word in ['mL']:
                tag = 'mL'
            prev_tag = current_tag
            current_tag = tag
            if current_tag == prev_tag:
                continue
            tag_sequence.append(current_tag)
        file.write(' '.join(tag_sequence) + '\n')
