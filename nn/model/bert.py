PROJECT_DIR = r'G:\PythonProjects\WineRecognition2'
TRAIN_DATASET_PATH = r'G:\PythonProjects\WineRecognition2\data\text\data_and_menu_gen_samples\Halliday_WineSearcher_Bruxelles_MenuGenSamples_v5_BottleSize_fixed.txt'
TEST_DATASET_PATH = r'G:\PythonProjects\WineRecognition2\data\text\menu_txt_tagged_fixed_bottlesize.txt'
DATA_INFO_PATH = r'G:\PythonProjects\WineRecognition2\data_info.json'
VOCAB_PATH = r'G:\PythonProjects\WineRecognition2\data\vocabs\Words_Halliday_WineSearcher_Bruxelles.json'

TASK = "ner"
NUM_EPOCHS=10
MODEL_CHECKPOINT = "distilbert-base-uncased"
BATCH_SIZE = 64
TEST_SIZE = 0.2
LABEL_ALL_TOKENS = True

import os
from datetime import datetime
import sys
import json
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
from nn.utils import CustomDataset, train, plot_losses, generate_tag_to_ix, get_model_confidence
from data_master import DataGenerator, count_unk_foreach_tag

with open(DATA_INFO_PATH) as file:
    label_list = json.load(file)['keys']['all']
    tag_to_ix = generate_tag_to_ix(label_list)


def get_token_dataset(train_dataset_path, test_dataset_path, columns=('tokens', f'{TASK}_tags')):
    with open(train_dataset_path, encoding='utf-8') as file:
        train_sents = DataGenerator.generate_sents2(file.read().split('\n'))

    with open(test_dataset_path, encoding='utf-8') as file:
        test_sents = DataGenerator.generate_sents2(file.read().split('\n'))

    train_df = pd.DataFrame(train_sents, columns=columns)
    test_df = pd.DataFrame(test_sents, columns=columns)

    train_df['whole_string'] = train_df['tokens'].apply(' '.join)
    test_df['whole_string'] = test_df['tokens'].apply(' '.join)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, test_dataset

old_tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
train_dataset, test_dataset = get_token_dataset(TRAIN_DATASET_PATH, TEST_DATASET_PATH)

def get_training_corpus():
    for start_idx in range(0, len(train_dataset), 1000):
        samples = train_dataset[start_idx : start_idx + 1000]
        yield samples["whole_string"]

tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), vocab_size=52000)
tokenizer.save_pretrained(f'./tokenizers/{MODEL_CHECKPOINT}-{datetime.today().strftime("%Y-%m-%d")}')

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{TASK}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(tag_to_ix[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(tag_to_ix[label[word_idx]] if LABEL_ALL_TOKENS else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

MODEL_CHECKPOINT = "distilbert-base-uncased"
model = AutoModelForTokenClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=len(tag_to_ix))

model_name = MODEL_CHECKPOINT.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-{TASK}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    push_to_hub=False,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

tokenized_train_eval_dataset = tokenized_train_dataset.train_test_split(test_size=TEST_SIZE)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train_eval_dataset["train"],
    eval_dataset=tokenized_train_eval_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()

trainer.save_model(f'{MODEL_CHECKPOINT}-wine')
trainer.evaluate(tokenized_test_dataset)
