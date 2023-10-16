import os
import json
import sys
import uuid

import papermill as pm
import datetime


def run_experiment(
        experiment: dict,
        exp_settings_path: str,
        notebook_path: str,
        train=True,
        log_output=False) -> dict:
    experiment_settings = json.load(open(exp_settings_path))
    mode = 'train' if train else 'test'
    run_id = str(uuid.uuid4())

    experiment['START_TIME'] = get_current_time()
    experiment['OUTPUT_DIR'] = get_output_dir(experiment_settings['artifacts_path'], mode, run_id)

    if not os.path.exists(experiment['OUTPUT_DIR']):
        os.mkdir(experiment['OUTPUT_DIR'])

    save_metadata(experiment['OUTPUT_DIR'], start_time=experiment['START_TIME'], run_id=run_id)

    pm.execute_notebook(
        input_path=notebook_path,
        output_path=os.path.join(experiment['OUTPUT_DIR'], os.path.basename(notebook_path)),
        parameters=experiment,
        log_output=log_output,
        stdout_file=sys.stdout,
        stderr_file=sys.stderr,
    )

    return experiment


def get_current_time():
    return '{:%d-%m-%Y_%H%M%S}'.format(datetime.datetime.now())


def get_output_dir(artifacts_path, mode, run_id):
    return f'{artifacts_path}/{mode}/experiment_{run_id}'


def save_metadata(output_dir, **kwargs):
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as file:
        json.dump(dict(kwargs), file)
