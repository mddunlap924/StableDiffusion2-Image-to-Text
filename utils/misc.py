"""
Miscellaneous and helper code for various tasks will be used in this script.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import yaml
import torch
import random
import numpy as np
import pandas as pd
import gc
import torch
from sentence_transformers import SentenceTransformer


def seed_everything(*, seed: int=42):
    """
    Seed everything

    Args:
        seed (_type_): Seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class RecursiveNamespace(SimpleNamespace):
    """
    Extending SimpleNamespace for Nested Dictionaries
    # https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8
    Args:
        SimpleNamespace (_type_): Base class is SimpleNamespace
    Returns:
        _type_: A simple class for nested dictionaries
    """
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))


def load_cfg(base_dir: Path, filename: str, *,
             as_namespace: bool=True) -> SimpleNamespace:
    """
    Load YAML configuration files saved uding the "cfgs" directory
    Args:
        base_dir (Path): Directory to YAML config. file
        filename (str): Name of YAML configuration file to load
    Returns:
        SimpleNamespace: A simple class for calling configuration parameters
    """
    cfg_path = Path(base_dir) / filename
    with open(cfg_path, 'r') as file:
        cfg_dict = yaml.safe_load(file)
    file.close()
    if as_namespace:
        cfg = RecursiveNamespace(**cfg_dict)
    else:
        cfg = cfg_dict
    return cfg


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def prompt_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    
    # Load encoder model
    st_model = SentenceTransformer('./kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2',
                                device='cuda',
                                )

    # Get prompt embeddings
    prompts = df['prompt'].values
    prompt_embeddings = st_model.encode(prompts,
                                        show_progress_bar=False,
                                        convert_to_tensor=True,
                                        )

    # Get embeddings in another dataframe
    prompt_embed_numpy = prompt_embeddings.detach().cpu().numpy()
    embed_cols = [f'embed_{i + 1}' for i in range(prompt_embed_numpy.shape[1])]
    embed = pd.DataFrame(prompt_embed_numpy, columns=embed_cols)

    # Put embeddings with df
    df = pd.concat([df, embed], axis=1)
    
    del st_model, prompts, prompt_embed_numpy, prompt_embeddings, embed
    _ = gc.collect()
    torch.cuda.empty_cache()
    
    return df
