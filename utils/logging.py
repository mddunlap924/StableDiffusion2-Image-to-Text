import os
import wandb
from utils.misc import load_cfg


class WandB:
    """ Log results for Weights & Biases """

    def __init__(self, cfg_basepath: str, cfg_filename: str, group_id: str):
        self.cfg = load_cfg(base_dir=cfg_basepath, filename=cfg_filename, as_namespace=False)
        self.group_id = group_id
        self.job_type = wandb.util.generate_id()
        self.run_name = f'{self.group_id}_{self.job_type}'
        
        # WandB Key in the environment variables
        if 'WANDB_KEY' in os.environ:
            self.key = os.environ['WANDB_KEY']
        else:
            self.key = None
        
        # Absolute path to where WandB data will be stored
        self.wandb_save_dir = os.environ['DIR_SAVE_WANDB']
        
        # # Turn off all code tracking
        # os.environ['WANDB_DISABLE_CODE'] = True
        
        # # If debugging then only log results locally
        # if self.cfg['debug']:
        #     self.mode = self.cfg['mode']
        # else:
        #     self.mode = 'online'


    def login_to_wandb(self):
        # Store WandB key in the environment variables
        if self.key is not None:
            wandb.login(key=self.key)
        else:
            print('Not logging info. in WandB')

    def get_logger(self):
        self.login_to_wandb()
        wb_logger = wandb.init(project=self.cfg['wandb']['project'],
                               dir=self.wandb_save_dir,
                               group=self.group_id,
                               job_type=self.job_type,
                               name=self.run_name,
                               config=self.cfg,
                               mode=self.cfg['wandb']['mode'],
                            #    interval=self.cfg['wandb']['interval_delta'],
                               )

        return wb_logger, self.run_name
