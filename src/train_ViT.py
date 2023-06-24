# Libraries
import sys
import argparse
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
        args.name = 'cfg0_HF.yaml'
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

    # Load samples from comp. host
    samples = pd.read_csv(Path(CFG.datasets.sample.base_dir) / CFG.datasets.sample.csv_name)
    samples['image_path'] = Path(CFG.datasets.sample.base_dir) / 'images' / samples['imgId']
    samples['image_path'] = samples['image_path'].astype(str) + '.png'
    
    # Load training and validation data
    train = pd.read_csv(CFG.datasets.train)
    val = pd.read_csv(CFG.datasets.val)
    # Reduce datasize for quicker testing
    if CFG.debug:
        train = train.iloc[0:500].reset_index(drop=True)
        val = val.iloc[0:500].reset_index(drop=True)
    
    # Train a ViT model
    vit.train(trn_df=train,
              val_df=val,
              model_name=CFG.model.name,
              params=CFG.train_params,
              model_save_path=SAVE_MODEL_PATH['path'],
              )
    # Inference with trained ViT model
    vit.inference(df=samples,
                  model_name=CFG.model.name,
                  model_load_path=SAVE_MODEL_PATH['path'],
                  batch_size=CFG.train_params.batch_size,
                  )
    
    # Log a Leaderboard score placeholder
    wandb.log({'lb': np.nan})
    
    # Close logger
    wandb.finish()

print('End of Script - Completed')
