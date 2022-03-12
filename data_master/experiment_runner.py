import os
import json
import papermill as pm
import datetime


def run_experiment(experiment: dict, exp_settings_path: str, notebook_path: str, train=True) -> dict:
    experiment_settings = json.load(open(exp_settings_path))
    mode = 'train' if train else 'test'

    experiment['START_TIME'] = '{:%d%m%Y_%H%M%S}'.format(datetime.datetime.now())
    experiment['OUTPUT_DIR'] = (
        f"{experiment_settings['artifacts_path']}/{mode}/{experiment['MODEL_NAME'] + '_' + experiment['START_TIME']}"
    )

    if not os.path.exists(experiment['OUTPUT_DIR']):
        os.mkdir(experiment['OUTPUT_DIR'])

    pm.execute_notebook(
        input_path=notebook_path,
        output_path=os.path.join(experiment['OUTPUT_DIR'], os.path.basename(notebook_path)),
        parameters=experiment
    )

    return experiment
