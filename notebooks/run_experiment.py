from data_master import run_experiment

EXP_SETTINGS_PATH = r'G:\PythonProjects\WineRecognition2\nn\experiment_settings.json'
LOG_OUTPUT = True

# Train BiLSTM-CRF with dict features
# experiment = run_experiment(
#     experiment={
#         'MODEL_NAME': 'BiLSTM-CRF_dict_features',
#         'RUN_NAME': 'train-bilstm-crf-custom-features',
#         'DATASET_PATH': r'G:\PythonProjects\WineRecognition2\data\text\data_and_menu_gen_samples\Halliday_WineSearcher_Bruxelles_MenuGenSamples_v5_BottleSize_fixed.txt',
#         'VOCAB_PATH': r'G:\PythonProjects\WineRecognition2\data\vocabs\Words_Halliday_WineSearcher_Bruxelles_WORD_NUMS.json',
#         'DATAINFO_PATH': r'G:/PythonProjects/WineRecognition2/data_info.json',
#         'DICTIONARY_PATH': r'G:\PythonProjects\WineRecognition2\data\dictionaries\Dict-byword_Halliday_Winesearcher_Bruxelles',
#         'DEVICE': 'cuda',
#         'BATCH_SIZE': 2048,
#         'EMBEDDING_DIM': 64,
#         'HIDDEN_DIM': 64,
#         'NUM_EPOCHS': 100,
#         'LEARNING_RATE': 0.01,
#         'SCHEDULER_FACTOR': 0.1,
#         'SCHEDULER_PATIENCE': 10,
#         'CASE_SENSITIVE_VOCAB': False,
#         'WEIGHT_DECAY': 1e-4,
#         'USE_NUM2WORDS': True,
#         'TEST_SIZE': 0.2
#     },
#     exp_settings_path=EXP_SETTINGS_PATH,
#     notebook_path='train_bilstm_crf_our_features.ipynb',
#     train=True,
#     log_output=LOG_OUTPUT
# )

# Test BiLSTM-CRF on menu with dict features
experiment = run_experiment(
    experiment={
        'MODEL_NAME': 'BiLSTM-CRF_dict_features',
        'MODEL_PATH': r'G:/PythonProjects/WineRecognition2/artifacts/train/BiLSTM-CRF_dict_features_23042023_005004/model/data/model.pth',
        'RUN_NAME': 'test-menu-blocks-bilstm-crf-custom-features',
        'DATA_PATH': r'G:\PythonProjects\WineRecognition2\ocr\data\results2\wine_menus_blocks2_with_tags.txt',
        'VOCAB_PATH': r'G:\PythonProjects\WineRecognition2\data\vocabs\Words_Halliday_WineSearcher_Bruxelles_WORD_NUMS.json',
        'DATAINFO_PATH': r'G:/PythonProjects/WineRecognition2/data_info.json',
        'DICTIONARY_PATH': r'G:\PythonProjects\WineRecognition2\data\dictionaries\Dict-byword_Halliday_Winesearcher_Bruxelles',
        'COMPUTE_METRICS': True,
        'CASE_SENSITIVE_VOCAB': False,
        'USE_NUM2WORDS': True,
        'DEVICE': 'cpu',
    },
    exp_settings_path=EXP_SETTINGS_PATH,
    notebook_path='test_bilstm_crf_our_features.ipynb',
    train=False,
    log_output=LOG_OUTPUT
)

print(experiment)

