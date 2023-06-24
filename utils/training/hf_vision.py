# from tqdm.notebook import tqdm
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import wandb

import timm
from timm.utils import AverageMeter

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sentence_transformers import SentenceTransformer

from utils.dataloaders.diffusion_data import IMGDataset
from utils.metrics import cosine_similarity

from torch_lr_finder import LRFinder
from transformers import AutoModel, AutoProcessor
from utils.metrics import cosine_similarity_loss


class Net(nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        clip = AutoModel.from_pretrained(model_name)
        self.vision = clip.vision_model
        self.fc = nn.Linear(1024, 384)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        out = self.vision(x)['pooler_output']
        return self.fc(out)


def load_pretrained_model(model_name, device, unfreeze_count):
    model = Net(model_name=model_name)

    trainable_model_weights = False
    for name, child in model.named_children():
        if name == 'vision':
            for pn, p in child.named_parameters():
                if str(unfreeze_count) in pn:
                    """start unfreezing layer , the weights are trainable"""
                    trainable_model_weights = True
                p.requires_grad = trainable_model_weights
                if p.requires_grad:
                    print(f"{pn} is set to be trainable.")

    return model


def train(trn_df,
          val_df,
          model_name,
          params,
          model_save_path,
          device,
):
    
    # Unpack training parameters
    num_epochs = params.epochs
    batch_size = params.batch_size
    lr = params.lr
    eta_min = params.eta_min
    EMBED_COLS = [f'embed_{i + 1}' for i in range(384)]
    
    # # Clip processor
    # clip_processor = AutoProcessor.from_pretrained(model_name)
    
    # Dataloaders
    train_dl = DataLoader(dataset=IMGDataset(image_paths=trn_df['image_path'].values.tolist(),
                                             targets=trn_df[EMBED_COLS].values,
                                             model_name=model_name),
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=8,
                          )
    val_dl = DataLoader(dataset=IMGDataset(image_paths=val_df['image_path'].values,
                                           targets=val_df[EMBED_COLS].values,
                                           model_name=model_name),
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=8,
                          )

    # Load Model
    model = load_pretrained_model(model_name=model_name,
                                  device=device,
                                  unfreeze_count=params.unfreeze.unfreeze_start)    
    model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr)
    optimizer.zero_grad()
    
    # Number of iterations / steps
    ttl_iters = num_epochs * len(train_dl)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=eta_min)
  
    # Training loop over epochs
    start_training_time = time.time()
    step_count = 0
    best_score = 0.0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'\nStart Epoch {epoch + 1}')
        train_meters = {
            'loss': AverageMeter(),
            'cos': AverageMeter(),
        }
        model.train()
        # TRAINING
        epoch_iters = int(len(train_dl))
        tk0 = tqdm(train_dl,
                   total=epoch_iters,
                #    miniters=epoch_iters / 100,
                   mininterval=10)
        for idx, (X, y) in enumerate(tk0):
        # for idx, (X, y) in enumerate(train_dl):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = cosine_similarity_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            trn_cos = cosine_similarity(y_pred.detach().cpu().numpy(),
                                        y.detach().cpu().numpy(),
                                        )

            # Store loss and cos
            train_meters['loss'].update(loss, n=X.size(0))
            train_meters['cos'].update(trn_cos, n=X.size(0))  
            
            # Progress bar info.
            tk0.set_postfix(train_loss=float(train_meters['loss'].avg.detach().cpu().numpy()),
                            train_cos=train_meters['cos'].avg,
                            lr=scheduler.get_last_lr()[0])       
            
            # Log Iterations results in WandB
            if ((step_count + 1) % 500) == 0:
                wandb.log({'step': step_count,
                           'train_loss': float(train_meters['loss'].avg.detach().cpu().numpy()),
                           'train_cos': train_meters['cos'].avg,
                           'lr': scheduler.get_last_lr()[0],
                           })
            step_count += 1
        print('Epoch {:d} / trn/loss={:.4f}, trn/cos={:.4f}'.format(
            epoch + 1,
            train_meters['loss'].avg,
            train_meters['cos'].avg))
        print(f'Epoch {epoch + 1} Training Time: '
              f'{(((time.time() - epoch_start_time) / 60) / 60):.1f} hrs.')

        # VALIDATION
        val_meters = {
            'loss': AverageMeter(),
            'cos': AverageMeter(),
            }
        model.eval()
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = cosine_similarity_loss(y_pred, y)
                
                val_cos = cosine_similarity(y_pred.detach().cpu().numpy(),
                                            y.detach().cpu().numpy(),
                                            )
                
                val_meters['loss'].update(loss, n=X.size(0))
                val_meters['cos'].update(val_cos, n=X.size(0))

        print('Epoch {:d} / val/loss={:.4f}, val/cos={:.4f}'.format(
            epoch + 1,
            val_meters['loss'].avg,
            val_meters['cos'].avg))
        
        # Save best model to disk
        if val_meters['cos'].avg > best_score:
            print(f"Epoch {epoch + 1}; Saved best model at: {model_save_path}")
            best_score = val_meters['cos'].avg
            torch.save(model.state_dict(), model_save_path)
            
        # Log Epoch results in WandB
        wandb.log({'epoch': epoch,
                   'train_loss': train_meters['loss'].avg,
                   'train_cos': train_meters['cos'].avg,
                   'val_loss': val_meters['loss'].avg,
                   'val_cos': val_meters['cos'].avg,
                   })
    
    # Best cosine similarity score during training
    wandb.log({'best_val_cos': best_score})
    
    # Total training time
    total_training_time = (((time.time() - start_training_time) / 60) / 60)
    print(f'Total Training Time: {total_training_time:.1f} hrs')
    wandb.log({'total_train_time': total_training_time})
    
    return


def inference(df,
    model_name,
    batch_size,
    model_load_path,
    device,
    *,
    flatten: bool=False):
    
    # Embedding column names
    EMBED_COLS = [f'embed_{i + 1}' for i in range(384)]
    
    # Load Model
    model = Net(model_name=model_name)
    
    # Load model weights and assign to device
    # model_load_path ='/home/mdunlap/Projects/StableDiff/output/exp/1g1gqrsy_bl3ah3bz/openai__clip-vit-large-patch14_1g1gqrsy_bl3ah3bz.pth'
    state_dict = torch.load(model_load_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Dataloaders
    test_dl = DataLoader(dataset=IMGDataset(image_paths=df['image_path'].values.tolist(),
                                            targets=df[EMBED_COLS].values,
                                            model_name=model_name),
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=8,
                          )
    
    # Predict on sample data points
    y_preds = []
    for X, _ in test_dl:
        X = X.to(device)
        with torch.no_grad():
            y_pred = model(X)
            y_preds.append(y_pred.cpu().numpy())
    y_preds = y_preds[0]
    
    # Cosine Similarity for 7-samples
    y_true = df[EMBED_COLS].values

    sample_cos = cosine_similarity(y_trues=y_true,
                                   y_preds=y_preds)
    print(f'7-Examples Cos. Score: {sample_cos:.5f}')
    
    # Log 7-example images 
    wandb.log({'7-example-imgs cos.': sample_cos})

    return
    