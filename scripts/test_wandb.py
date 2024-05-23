import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="test-project",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
        "run_info": {
            "name": "some name",
            "output_directory": "dir1/dir2",
            "start_time": "2023-06-06"
        }
    },
)

obj = wandb.Object3D(r"G:\dev\texturing-3d-shapes\TEXTurePaper\shapes\barrel.obj")
wandb.log({'objects': {'barrel': obj}})

# artifact = wandb.Artifact('dataset', type='dataset')
# artifact.add_file(r"G:\PythonProjects\WineRecognition2\data\text\Bruxelles_Wine_ES-all_keys.txt", 'dataset.txt')
# wandb.log_artifact(artifact)

# simulate training
# epochs = 10
# offset = random.random() / 5
# for epoch in range(2, epochs):
#     acc = 1 - 2 ** -epoch - random.random() / epoch - offset
#     loss = 2 ** -epoch + random.random() / epoch + offset
#
#     # log metrics to wandb
#     wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
