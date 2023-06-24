from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor


class DiffusionDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path'])
        if len(image.getbands()) < 3:
            image = image.convert('RGB')
        image = self.transform(image)
        prompt = row['prompt']
        return image, prompt


class DiffusionCollator:
    def __init__(self):
        self.st_model = SentenceTransformer(
            './kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2',
            device='cpu'
        )

    def __call__(self, batch):
        images, prompts = zip(*batch)
        images = torch.stack(images)
        prompt_embeddings = self.st_model.encode(prompts,
                                                 show_progress_bar=False,
                                                 convert_to_tensor=True,
                                                 )
        return images, prompt_embeddings


def get_dataloaders(
    trn_df,
    val_df,
    transform,
    # input_size,
    batch_size,
):

    trn_dataset = DiffusionDataset(trn_df, transform['train'])
    val_dataset = DiffusionDataset(val_df, transform['val'])
    collator = DiffusionCollator()
    
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        dataset=trn_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        collate_fn=collator
    )
    dataloaders['val'] = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=8,
        drop_last=False,
        collate_fn=collator
    )
    return dataloaders


class DiffusionTestDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.transform(image)
        return image


class IMGDataset:
    def __init__(self, image_paths, targets, model_name):
        self.images = image_paths
        self.labels = targets
        # Clip processor
        self.input_processor = AutoProcessor.from_pretrained(model_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        if len(image.getbands()) < 3:
            image = image.convert('RGB')
        image = self.input_processor(images=image)
        image = image.pixel_values[0]
        target = self.labels[item]
        return image, target
