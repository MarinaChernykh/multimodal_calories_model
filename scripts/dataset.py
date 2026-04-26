from functools import partial

import pandas as pd
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class FoodCaloriesDataset(Dataset):
    """Датасет блюд: текст ингредиентов, изображение, масса и калории."""
    def __init__(self, dish_df, ingredients_df, transforms=None):
        self.dish_df = dish_df
        self.ingredients_df = ingredients_df
        self.ingredients_dict = {f'ingr_{row.id:010d}': row.ingr for _, row in ingredients_df.iterrows()}        
        self.transforms = transforms

    def __len__(self):
        return len(self.dish_df)
    
    def _ingredients_to_text(self, ingredient_ids):
        """Преобразует список id ингредиентов в текстовое описание."""
        if pd.isna(ingredient_ids):
            return 'unknown'
        ingredients_names = [
            self.ingredients_dict[ingr_id]
            for ingr_id in ingredient_ids.split(';')
            if ingr_id in self.ingredients_dict
        ]
        return ', '.join(ingredients_names)

    def __getitem__(self, idx):
        row = self.dish_df.iloc[idx]
        text = self._ingredients_to_text(row['ingredients'])
        mass = float(row['total_mass'])
        calories = float(row['total_calories'])
        image = Image.open(f'data/images/{row["dish_id"]}/rgb.png').convert('RGB')
        if self.transforms:
            image = self.transforms(image=np.array(image))['image']

        return {
            'calories': calories,
            'text': text,
            'mass': mass,
            'image': image
        }


def collate_fn(batch, tokenizer):
    """Собирает батч и токенизирует текст для модели."""
    texts = [item['text'] for item in batch]
    masses = torch.tensor([item['mass'] for item in batch])
    calories = torch.tensor([item['calories'] for item in batch])
    images = torch.stack([item['image'] for item in batch])

    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )

    return {
        'image': images,
        'calories': calories,
        'mass': masses,
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask']
    }


def get_transforms(image_model_cfg, cfg, mode='train'):
    """Возвращает аугментации для train/val/test изображений."""
    if mode == 'train':
        return A.Compose(
            [
				A.SmallestMaxSize(max_size=max(image_model_cfg.input_size[1], image_model_cfg.input_size[2]), p=1.0),
    			A.CenterCrop(height=image_model_cfg.input_size[1], width=image_model_cfg.input_size[2], p=1.0),
				A.HorizontalFlip(p=0.5),
    			A.Rotate(limit=10, p=0.5),
       			A.RandomBrightnessContrast(p=0.3),
				A.ColorJitter(
					brightness=0.2,
					contrast=0.2,
					saturation=0.2,
					hue=0.05,
					p=0.5
				),
				A.OneOf([
					A.GaussianBlur(blur_limit=(3, 5)),
					A.GaussNoise(var_limit=(10.0, 50.0)),
				], p=0.3),
				A.Normalize(mean=image_model_cfg.mean, std=image_model_cfg.std),
				ToTensorV2()
        ], seed=cfg.RANDOM_SEED)
    else:
        return A.Compose([
            A.SmallestMaxSize(max_size=max(image_model_cfg.input_size[1], image_model_cfg.input_size[2]), p=1.0),
            A.CenterCrop(height=image_model_cfg.input_size[1], width=image_model_cfg.input_size[2], p=1.0),
			A.Normalize(mean=image_model_cfg.mean, std=image_model_cfg.std),
            ToTensorV2()
        ])


def create_dataloader(df, ingredients_df, tokenizer, image_model_cfg, cfg, mode):
    """Создаёт DataLoader для указанного режима (train/val/test)."""
    transforms = get_transforms(image_model_cfg, cfg, mode)
    dataset = FoodCaloriesDataset(df, ingredients_df, transforms)
    shuffle = (mode == 'train')
    dataloader = DataLoader(
		dataset,
		batch_size=cfg.BATCH_SIZE,
		shuffle=shuffle,
		collate_fn=partial(collate_fn, tokenizer=tokenizer)
	)
    return dataloader
