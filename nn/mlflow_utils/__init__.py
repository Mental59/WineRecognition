import os
import json
import mlflow
import pandas as pd
from sklearn_crfsuite import metrics

from data_master import DataAnalyzer
from ..utils import plot_losses


def log_mlflow_on_train(run_params, model, classes, losses, y_true, y_pred, test_eval=None):
    exp_settings = json.load(
        open(r"G:\PythonProjects\WineRecognition2\nn\experiment_settings.json")
    )

    mlflow.set_experiment(exp_settings['experiment'])

    with mlflow.start_run(run_name=run_params['run_name']):

        mlflow.log_metrics({
            'f1-score': metrics.flat_f1_score(y_true, y_pred, average='weighted', labels=classes),
            'precision': metrics.flat_precision_score(y_true, y_pred, average='weighted', labels=classes),
            'recall': metrics.flat_recall_score(y_true, y_pred, average='weighted', labels=classes),
            'accuracy': metrics.flat_accuracy_score(y_true, y_pred)
        })

        mlflow.log_text(
            metrics.flat_classification_report(y_true, y_pred, labels=classes, digits=3),
            f"{run_params['output_dir']}/flat-classification-report.txt"
        )

        unk_foreach_tag = run_params.pop('unk_foreach_tag', None)
        if unk_foreach_tag is not None:
            mlflow.log_text(
                unk_foreach_tag,
                f"{run_params['output_dir']}/unk_foreach_tag.txt"
            )

        if test_eval is not None:
            DataAnalyzer.analyze(
                test_eval,
                keys=classes,
                table_save_path=os.path.join(run_params['output_dir'], 'colored-table.xlsx'),
                diagram_save_path=os.path.join(run_params['output_dir'], 'diagram.png')
            )

        plot_losses(losses, figsize=(12, 8), show=False, savepath=os.path.join(run_params['output_dir'], 'losses.png'))

        mlflow.log_artifacts(run_params['output_dir'])

        mlflow.pytorch.log_model(model, f"{run_params['output_dir']}/model")

        mlflow.log_params(run_params)


def log_mlflow_on_test(run_params, classes, x_test, y_pred, y_true):
    exp_settings = json.load(
        open(r"G:\PythonProjects\WineRecognition2\nn\experiment_settings.json")
    )

    with open('{}/results.txt'.format(run_params['output_dir']), 'w', encoding='utf-8') as file:
        for sentence, tags in zip(x_test, y_pred):
            f = ['%-20s'] * len(sentence)
            file.write(' '.join(f) % tuple(tags) + '\n')
            file.write(' '.join(f) % tuple(sentence) + '\n')

    mlflow.set_experiment(exp_settings['experiment'])

    with mlflow.start_run(run_name=run_params['run_name']):
        if run_params['compute_metrics']:
            mlflow.log_metrics({
                'f1-score': metrics.flat_f1_score(y_true, y_pred, average='weighted', labels=classes),
                'precision': metrics.flat_precision_score(y_true, y_pred, average='weighted', labels=classes),
                'recall': metrics.flat_recall_score(y_true, y_pred, average='weighted', labels=classes),
                'accuracy': metrics.flat_accuracy_score(y_true, y_pred)
            })

            mlflow.log_text(
                metrics.flat_classification_report(y_true, y_pred, labels=classes, digits=3),
                f"{run_params['output_dir']}/flat-classification-report.txt"
            )

            test_eval = []
            for sentence, true_tags, pred_tags in zip(x_test, y_true, y_pred):
                test_eval.append(list(zip(sentence, true_tags, pred_tags)))

            DataAnalyzer.analyze(
                test_eval,
                keys=classes,
                table_save_path=os.path.join(run_params['output_dir'], 'colored-table.xlsx'),
                diagram_save_path=os.path.join(run_params['output_dir'], 'diagram.png')
            )
        else:
            df = []
            for sentence, tags in zip(x_test, y_pred):
                output = [' '.join(word for word, tag in zip(sentence, tags) if tag == cls) for cls in classes]
                df.append(output)

            with pd.ExcelWriter('{}/results.xlsx'.format(run_params['output_dir']), engine='xlsxwriter') as writer:
                pd.DataFrame(df, columns=classes).to_excel(writer, sheet_name='results')

        unk_foreach_tag = run_params.pop('unk_foreach_tag', None)
        if unk_foreach_tag is not None:
            mlflow.log_text(
                unk_foreach_tag,
                f"{run_params['output_dir']}/unk_foreach_tag.txt"
            )

        mlflow.set_experiment(exp_settings['experiment'])
        mlflow.log_artifacts(run_params['output_dir'])
        mlflow.log_params(run_params)
