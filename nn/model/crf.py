import sys
sys.path.insert(0, r'G:\PythonProjects\WineRecognition2')
# import sklearn_crfsuite
import eli5
from sklearn.model_selection import train_test_split
from features import features
from data_master import DataGenerator, DataLoader
from mlflow_utils import log_mlflow_on_train

TRAIN_DATASET_PATH = r"G:\PythonProjects\WineRecognition2\data\text\data_and_menu_gen_samples\Halliday_WineSearcher_Bruxelles_MenuGenSamples_v5_BottleSize_fixed.txt"
DICTIONARY_PATH = r"G:\PythonProjects\WineRecognition2\data\dictionaries\Dict-byword_Halliday_Winesearcher_Wine_AU-only_completed_rows"
MODEL_NAME = "CRF_test"
ALGORITHM = 'lbfgs'
C1 = 0.1
C2 = 0.1
MAX_ITERATIONS = 5
ALL_POSSIBLE_TRANSITIONS = True
TEST_SIZE = 0.2
RUN_NAME = 'test_run_train_model'
OUTPUT_DIR = r"G:\PythonProjects\WineRecognition2\artifacts\train_test_run"
START_TIME = ''

with open(TRAIN_DATASET_PATH, encoding='utf-8') as file:
    sents = DataGenerator.generate_sents(
        file.read().split('\n')
    )

train_sents, val_sents = train_test_split(sents, test_size=TEST_SIZE)

freq_dict = DataLoader.load_frequency_dictionary(DICTIONARY_PATH, to_lowercase=True)

X_train = [features.sent2features(s, freq_dict) for s in train_sents]
y_train = [features.sent2labels(s) for s in train_sents]

X_val = [features.sent2features(s, freq_dict) for s in val_sents]
y_val = [features.sent2labels(s) for s in val_sents]

feature_list = list(X_train[0][1].keys())

import sklearn_crfsuite
model = sklearn_crfsuite.CRF(
    algorithm=ALGORITHM,
    c1=C1,
    c2=C2,
    max_iterations=MAX_ITERATIONS,
    all_possible_transitions=ALL_POSSIBLE_TRANSITIONS
)
model.fit(X_train, y_train)
eli5.show_weights(model, top=30)

y_pred = model.predict(X_val)

test_eval = val_sents.copy()
for i, wine in enumerate(test_eval):
    for j, word in enumerate(wine):
        test_eval[i][j] += (y_pred[i][j],)

run_params = {
    'dataset_path': TRAIN_DATASET_PATH,
    'dictionary_path': DICTIONARY_PATH,
    'algorithm': ALGORITHM,
    'model_name': MODEL_NAME,
    'c1': C1,
    'c2': C2 ,
    'max_iterations': MAX_ITERATIONS,
    'all_possible_transitions': ALL_POSSIBLE_TRANSITIONS,
    'test_size': TEST_SIZE,
    'runname': RUN_NAME,
    'start_time': START_TIME,
    'output_dir': OUTPUT_DIR,
    'features': feature_list
}

log_mlflow_on_train(
    run_params=run_params,
    model=model,
    y_true=y_val,
    y_pred=y_pred,
    test_eval=test_eval
)
