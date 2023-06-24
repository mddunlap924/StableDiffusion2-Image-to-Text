# Shell script to execute customized model training.
# YAML configurations files are used to direct training factors such as datasets, models, hyperparameters, etc.

python3 ./src/train_ViT.py --dir './cfgs' --name 'cfg0.yaml' &&
