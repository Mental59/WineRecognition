import os
import json
import mlflow
import eli5
from sklearn_crfsuite import metrics
from data_master import DataAnalyzer
import pandas as pd


def log_mlflow_on_train(run_params, model, y_true, y_pred, test_eval=None):
    exp_settings = json.load(
        open(r'G:\PythonProjects\WineRecognition2\experiment_settings.json')
    )

    mlflow.set_experiment(exp_settings['experiment'])

    with mlflow.start_run(run_name=run_params['runname']):

        mlflow.log_metrics({
            'f1-score': metrics.flat_f1_score(y_true, y_pred, average='weighted', labels=model.classes_),
            'precision': metrics.flat_precision_score(y_true, y_pred, average='weighted', labels=model.classes_),
            'recall': metrics.flat_recall_score(y_true, y_pred, average='weighted', labels=model.classes_),
            'accuracy': metrics.flat_accuracy_score(y_true, y_pred)
        })

        mlflow.log_text(
            metrics.flat_classification_report(y_true, y_pred, labels=model.classes_, digits=3),
            f"{run_params['output_dir']}/flat-classification-report.txt"
        )

        if test_eval is not None:
            DataAnalyzer.analyze(
                test_eval,
                keys=model.classes_,
                table_save_path=os.path.join(run_params['output_dir'], 'colored-table.xlsx'),
                diagram_save_path=os.path.join(run_params['output_dir'], 'diagram.png')
            )

        mlflow.log_text(
            eli5.format_as_html(eli5.explain_weights(model)),
            f"{run_params['output_dir']}/explained_weights.html"
        )

        mlflow.log_artifacts(run_params['output_dir'])

        mlflow.sklearn.log_model(model, f"{run_params['output_dir']}/model")

        mlflow.log_params(run_params)


def log_mlflow_on_test(run_params, model, y_true, y_pred, test_eval=None):
    exp_settings = json.load(
        open(r"G:\PythonProjects\WineRecognition2\experiment_settings.json")
    )

    mlflow.set_experiment(exp_settings['experiment'])

    with mlflow.start_run(run_name=run_params['runname']):

        if run_params['compute_metrics']:

            mlflow.log_metrics({
                'f1-score': metrics.flat_f1_score(y_true, y_pred, average='weighted', labels=model.classes_),
                'precision': metrics.flat_precision_score(y_true, y_pred, average='weighted', labels=model.classes_),
                'recall': metrics.flat_recall_score(y_true, y_pred, average='weighted', labels=model.classes_),
                'accuracy': metrics.flat_accuracy_score(y_true, y_pred)
            })

            mlflow.log_text(
                metrics.flat_classification_report(y_true, y_pred, labels=model.classes_, digits=3),
                f"{run_params['output_dir']}/flat-classification-report.txt"
            )

            if test_eval is not None:
                DataAnalyzer.analyze(
                    test_eval,
                    keys=model.classes_,
                    table_save_path=os.path.join(run_params['output_dir'], 'colored-table.xlsx'),
                    diagram_save_path=os.path.join(run_params['output_dir'], 'diagram.png')
                )
        else:
            with pd.ExcelWriter(os.path.join(run_params['output_dir'], 'results.xlsx'), engine='xlsxwriter') as writer:
                test_eval.to_excel(writer, sheet_name='values')

        mlflow.log_artifacts(run_params['output_dir'])
        mlflow.log_params(run_params)
