import os
import json
import mlflow
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
