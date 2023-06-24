# Shell script to execute customized model training.
# YAML configurations files are used to direct training factors such as datasets, models, hyperparameters, etc.

python3 ./src/train_HF.py --dir './cfgs' --name 'cfg0_HF.yaml' &&
wait
sleep 10

python3 ./src/train_HF.py --dir './cfgs' --name 'cfg0_HF1.yaml' &&
wait
sleep 10

python3 ./src/train_HF.py --dir './cfgs' --name 'cfg0_HF2.yaml' &&
wait
sleep 10

python3 ./src/train_HF.py --dir './cfgs' --name 'cfg0_HF3.yaml' &&
wait
sleep 10

python3 ./src/train_HF.py --dir './cfgs' --name 'cfg0_HF4.yaml' &&
