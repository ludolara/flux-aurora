import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

def ensure_three_channels(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

class HuggingFaceImageDataset(Dataset):
    def __init__(self, dataset_name, split, img_size=512):
        self.dataset = load_dataset(dataset_name, split=split, cache_dir='/home/mila/l/luis.lara/scratch/.cache')
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            input_img = self.dataset[idx].get('input')
            output_img = self.dataset[idx].get('output')
            instruction = self.dataset[idx].get('instruction')

            if input_img is None or output_img is None or instruction is None:
                raise ValueError(f"Missing data at index {idx}")

            input_img = ensure_three_channels(c_crop(input_img).resize((self.img_size, self.img_size)))
            output_img = ensure_three_channels(c_crop(output_img).resize((self.img_size, self.img_size)))
            
            input_img = torch.from_numpy((np.array(input_img) / 127.5) - 1).permute(2, 0, 1)
            output_img = torch.from_numpy((np.array(output_img) / 127.5) - 1).permute(2, 0, 1)

            return input_img, output_img, instruction
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return None, None, None

def custom_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    return torch.utils.data.dataloader.default_collate(batch)

def loader(train_batch_size, num_workers, **args):
    dataset = HuggingFaceImageDataset(**args)
    return DataLoader(
        dataset, 
        batch_size=train_batch_size, 
        num_workers=num_workers, 
        shuffle=True,
        collate_fn=custom_collate_fn
    )
