# Libraries
import sys
import os
import argparse
import socket
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import wandb
from sklearn.model_selection import train_test_split

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get parent folder for the project locally
BASE_DIR = []
for i in Path.cwd().parts:
    if i != '/':
        BASE_DIR.append(i)
    if i == 'StableDiff':
        break
BASE_DIR = Path('/' + '/'.join(BASE_DIR))

# Add local path to libraries/modules (useful if no internet available)
for path_append in [BASE_DIR,
                    BASE_DIR / 'kaggle/input/sentence-transformers-222/sentence-transformers',
                    ]:
    sys.path.append(str(path_append))
    print(str(path_append))


from sentence_transformers import SentenceTransformer
from utils.misc import seed_everything, load_cfg, debugger_is_active
from utils.dataloaders.diffusion_data import get_dataloaders
from utils.training import vit
from utils.metrics import cosine_similarity
from utils.logging import WandB

# Seed Everything
SEED = 42
seed_everything(seed=SEED)

# Get Device type for processing
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":

    # Determine if running in debug mode
    # If in debug manually point to CFG file
    is_debugger = debugger_is_active()

    # Construct the argument parser and parse the arguments
    if is_debugger:
        args = argparse.Namespace()
        args.dir = './cfgs'
        args.name = 'cfg_coca0.yaml'
    else:
        arg_desc = '''This program points to input parameters for model training'''
        parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                         description= arg_desc)
        parser.add_argument("-cfg_basepath", 
                            "--dir",
                            required=True, 
                            help = "Base Dir. for the YAML config. file")
        parser.add_argument("-cfg_filename", 
                            "--name",
                            required=True, 
                            help="File name of YAML config. file")
        args = parser.parse_args()
        print(args)
    
    # Load configuration file
    CFG = load_cfg(base_dir=args.dir, filename=args.name)

    # Weights and Biases Logger
    wandb_group_id = wandb.util.generate_id()
    wb_logger, run_name = WandB(cfg_basepath=args.dir,
                                cfg_filename=args.name,
                                group_id=wandb_group_id).get_logger()
    
    # Path to save/load the model
    SAVE_MODEL_PATH = {'dir': BASE_DIR / 'output/exp' / f'{run_name}',
                       'filename': f'{CFG.model.name}_{run_name}.pth'}
    SAVE_MODEL_PATH['path'] = SAVE_MODEL_PATH['dir'] / SAVE_MODEL_PATH['filename']
    Path(SAVE_MODEL_PATH['dir']).mkdir(parents=True, exist_ok=True)

    
    # Make os command call to train COCa model
    code_to_execute = (
                        f'python -m training.main '
                        f'--dataset-type "csv" '
                        f'--train-data {CFG.datasets.train} '
                        f'--val-data {CFG.datasets.val} '
                        f'--warmup {CFG.train_params.warmup} '
                        f'--batch-size {CFG.train_params.batch_size} '
                        f'--lr {CFG.train_params.lr} '
                        f'--wd {CFG.train_params.wd} '
                        f'--epochs {CFG.train_params.epochs} '
                        f'--workers 8 '
                        f'--model {CFG.model.name} '
                        f'--pretrained {CFG.model.checkpoint} '
                        f'--report-to "wandb" '
                        f'--wandb-project-name "{CFG.wandb.project}" '
                        f'--coca-contrastive-loss-weight 0 '
                        f'--coca-caption-loss-weight 1 '
                        f'--log-every-n-steps {CFG.wandb.log_every_n_steps}'
                        )


    # Log a Leaderboard score placeholder
    wandb.log({'lb': np.nan})
    
    # Close logger
    wandb.finish()

print('End of Script - Completed')
