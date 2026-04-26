import os

import numpy as np
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel
from tqdm import tqdm


class MultimodalCaloriesModel(nn.Module):
    """
    Мультимодальная регрессионная модель для предсказания калорийности
    по изображению блюда, списку ингредиентов и весу.
    """
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(cfg.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            cfg.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )
        
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, cfg.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, cfg.HIDDEN_DIM)
        self.mass_proj = nn.Linear(1, cfg.HIDDEN_DIM // 4)

        self.fusion = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM * 2 + cfg.HIDDEN_DIM // 4, cfg.HIDDEN_DIM),
            nn.LayerNorm(cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT)
        )

        self.regressor = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM // 2, 1)
        )

    def forward(self, text_input, image_input, mass):
        """Прямой проход модели: возвращает предсказание калорийности."""
        text_features = self.text_model(**text_input).last_hidden_state[:, 0, :]
        text_emb = self.text_proj(text_features)

        image_features = self.image_model(image_input)
        image_emb = self.image_proj(image_features)

        mass = mass.unsqueeze(1).float()
        mass_emb = self.mass_proj(mass)

        fused = torch.cat([text_emb, image_emb, mass_emb], dim=1)
        fused = self.fusion(fused)

        calories = self.regressor(fused)
        return calories.squeeze(1)


def set_requires_grad(module, unfreeze_pattern='', verbose=False):
    """Замораживает или размораживает слои модели по шаблону."""
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split('|')

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f'Разморожен слой: {name}')
        else:
            param.requires_grad = False


def train_one_epoch(train_dataloader, model, device, optimizer, criterion):
    """Обучение модели на одной эпохе."""
    model.train()
    total_loss = 0
        
    for batch in tqdm(train_dataloader, desc='train'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        mass = batch['mass'].to(device)
        targets = batch['calories'].to(device)
            
        optimizer.zero_grad()
        preds = model(
                text_input={
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                },
                image_input=images,
                mass=mass
        )
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
            
        total_loss += loss.item()
        
    avg_train_loss = total_loss / len(train_dataloader)
    return avg_train_loss


def evaluate(model, dataloader, device, return_details=False):
    """Оценка модели."""
    mode = 'test' if return_details else 'val'       
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=mode):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            mass = batch['mass'].to(device)
            targets = batch['calories'].to(device)
            
            preds = model(
                text_input={
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                },
                image_input=images,
                mass=mass
            )
            preds_all.append(preds.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    errors = np.abs(preds_all - targets_all)
    mae = np.mean(errors)
    
    if return_details:
        return mae, preds_all, targets_all, errors
    return mae


def train(train_dataloader, val_dataloader, cfg):
    """Полный цикл обучения модели с сохранением лучшей версии."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Обучение модели запущено на {device}.')
    model = MultimodalCaloriesModel(cfg).to(device)
    set_requires_grad(model.text_model, cfg.TEXT_MODEL_UNFREEZE)
    set_requires_grad(model.image_model, cfg.IMAGE_MODEL_UNFREEZE)

    optimizer = AdamW([
        {'params': model.text_model.parameters(), 'lr': cfg.TEXT_LR},
        {'params': model.image_model.parameters(), 'lr': cfg.IMAGE_LR},
        {'params': (
			list(model.text_proj.parameters()) +
			list(model.image_proj.parameters()) +
			list(model.mass_proj.parameters()) +
			list(model.fusion.parameters()) +
			list(model.regressor.parameters())), 'lr': cfg.CLASSIFIER_LR}
    ])
    criterion = nn.L1Loss()  # MAE loss
    best_val_mae = float('inf')
    train_losses = []
    val_maes = []
    
    for epoch in range(cfg.EPOCHS):
        print(f'\nЭпоха {epoch+1}')
        
		# Обучение
        train_loss = train_one_epoch(train_dataloader, model, device, optimizer, criterion)
        train_losses.append(train_loss)
        
        # Валидация
        val_mae = evaluate(model, val_dataloader, device)
        val_maes.append(val_mae)
        
        print(f'Epoch {epoch+1}/{cfg.EPOCHS} | train Loss: {train_loss:.4f} | val MAE: {val_mae:.4f}')
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            os.makedirs(cfg.SAVE_MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg.SAVE_MODEL_DIR, cfg.SAVE_MODEL_NAME))
            print(f'Модель эпохи {epoch+1} c MAE {val_mae:.4f} сохранена.')
    
    # График
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train MAE')
    plt.plot(val_maes, label='Val MAE')
    plt.legend()
    plt.title('Динамика обучения модели')
    plt.xlabel('Эпохи')
    plt.ylabel('MAE')
    plt.show()

    return model
