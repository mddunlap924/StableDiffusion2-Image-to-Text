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

from sentence_transformers import SentenceTransformer

from utils.dataloaders.diffusion_data import get_dataloaders, DiffusionTestDataset
from utils.metrics import cosine_similarity

from torch_lr_finder import LRFinder


def train(
    trn_df,
    val_df,
    model_name,
    params,
    model_save_path,
):
    
    # Unpack training parameters
    num_epochs = params.epochs
    batch_size = params.batch_size
    lr = params.lr
    eta_min = params.eta_min
    
    # CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = timm.create_model(model_name,
                              pretrained=True,
                              num_classes=384,
                              )
    
    # Create model compose transform
    data_cfg = timm.data.resolve_data_config(model.default_cfg)
    
    # Create different transforms for training and validation
    val_transform = timm.data.create_transform(**data_cfg)
    train_transform = timm.data.create_transform(**data_cfg)
    (train_transform.transforms
     .insert(1, transforms.RandomHorizontalFlip(p=params.img_aug.horizontal_flip)))
    
    # Get dataloaders
    dataloaders = get_dataloaders(
        trn_df,
        val_df,
        transform={'train': train_transform, 'val': val_transform},
        batch_size=batch_size,
    )
    
    # Freeze/Unfreeze Certain Layers
    if params.unfreeze.apply:
        trainable_model_weights = False
        for name, child in model.named_children():
            if name == 'vision':
                for pn, p in child.named_parameters():
                    if str(params.unfreeze.unfreeze_start) in pn:
                        """start unfreezing layer , the weights are trainable"""
                        trainable_model_weights = True
                    p.requires_grad = trainable_model_weights
                    if p.requires_grad:
                        print(f"{pn} is set to be trainable.")
    
    # model.set_grad_checkpointing()
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    ttl_iters = num_epochs * len(dataloaders['train'])
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=eta_min)
    criterion = nn.CosineEmbeddingLoss()
    best_score = -1.0
    
    # # Learning Rate Finder
    # lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    # lr_finder.range_test(dataloaders['train'],
    #                     #  val_loader=dataloaders['val'],
    #                      end_lr=0.1,
    #                      num_iter=100,
    #                      step_mode="linear")
    # lr_finder.plot(log_lr=False)
    # lr_finder.reset()
    
    start_training_time = time.time()
    step_count = 0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'\nStart Epoch {epoch + 1}')
        train_meters = {
            'loss': AverageMeter(),
            'cos': AverageMeter(),
        }
        model.train()
        # epoch_iters = int(len(dataloaders['train']))
        # tk0 = tqdm(dataloaders['train'],
        #            total=epoch_iters,
        #            miniters=epoch_iters / 1_000)
        for idx, (X, y) in enumerate(dataloaders['train']):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            X_out = model(X)
            target = torch.ones(X.size(0)).to(device)
            loss = criterion(X_out, y, target)
            loss.backward()

            optimizer.step()
            scheduler.step()

            trn_loss = loss.item()
            trn_cos = cosine_similarity(
                X_out.detach().cpu().numpy(), 
                y.detach().cpu().numpy()
            )

            train_meters['loss'].update(trn_loss, n=X.size(0))
            train_meters['cos'].update(trn_cos, n=X.size(0))
            
            # # Progress bar info.
            # tk0.set_postfix(train_loss=train_meters['loss'].avg,
            #                 train_cos=train_meters['cos'].avg)
            
            
            # Log Iterations results in WandB
            if ((step_count + 1) % 2_000) == 0:
                wandb.log({'step': step_count,
                           'train_loss': train_meters['loss'].avg,
                           'train_cos': train_meters['cos'].avg,
                           'lr': scheduler.get_last_lr(),
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
        for X, y in dataloaders['val']:
            X, y = X.to(device), y.to(device)

            with torch.no_grad():
                X_out = model(X)
                target = torch.ones(X.size(0)).to(device)
                loss = criterion(X_out, y, target)

                val_loss = loss.item()
                val_cos = cosine_similarity(
                    X_out.detach().cpu().numpy(), 
                    y.detach().cpu().numpy()
                )

            val_meters['loss'].update(val_loss, n=X.size(0))
            val_meters['cos'].update(val_cos, n=X.size(0))

        print('Epoch {:d} / val/loss={:.4f}, val/cos={:.4f}'.format(
            epoch + 1,
            val_meters['loss'].avg,
            val_meters['cos'].avg))
        

        # Save best model to disk
        if val_meters['cos'].avg > best_score:
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
    *,
    flatten: bool=False):
    
    # CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = timm.create_model(model_name,
                              pretrained=False,
                              num_classes=384,
                              )
    
    # Create model compose transform
    data_cfg = timm.data.resolve_data_config(model.default_cfg)
    
    # Create different transforms for training and validation
    transform = timm.data.create_transform(**data_cfg)

    # Datasets and Loaders
    dataset = DiffusionTestDataset(df['image_path'].to_list(), transform)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )

    # Load model weights and assign to device
    state_dict = torch.load(model_load_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Predict on sample data points
    y_preds = []
    for X in dataloader:
        X = X.to(device)

        with torch.no_grad():
            X_out = model(X)
            y_preds.append(X_out.cpu().numpy())
    if flatten:
        preds = np.vstack(y_preds).flatten()
    
    # Inference on 7-sample examples
    st_model = SentenceTransformer(
        './kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2',
        device='cpu',
        )
    y_true = []
    for prompt in df['prompt'].to_list():
        y = st_model.encode(
                prompt, 
                show_progress_bar=False, 
                convert_to_tensor=True
            )
        y_true.append(y.numpy())

    sample_cos = cosine_similarity(y_trues=y_true,
                                   y_preds=y_preds[0])
    print(f'7-Examples Cos. Score: {sample_cos:.5f}')
    
    # Log 7-example images 
    wandb.log({'7-example-imgs cos.': sample_cos})

    return
    