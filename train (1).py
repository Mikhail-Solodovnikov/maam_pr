import torch
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from model import LanguageModel
import numpy as np


sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})


def plot_losses(train_losses: List[float], val_losses: List[float]):
    """
    График лосса и perplexity для трейна и валидационных выборок
    :param train_losses: список потерь на тренировочной выборке для каждой эпохи
    :param val_losses: список потерь на валидационной выборке для каждой эпохи
    """
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    """
    Подсчет метрики для трейна и валидации с учетом списка лоссов
    """
    train_perplexities, val_perplexities = np.exp(train_losses), np.exp(val_losses)

    axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    axs[1].set_ylabel('perplexity')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    """
    Обрабатываем одну эпоху обучения модели
    :param model: языковая модель для обучения
    :param optimizer: оптимизатор для обновления параметров модели
    :param criterion: функция потерь
    :param loader: даталоадер для тренировочной выборки
    :param tqdm_desc: описание для прогресс-бара
    :return: running train loss
    """
    device = next(model.parameters()).device
    train_loss = 0.0

    model.train()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        """
        Проходим по батчам данных, 
        вычисляем потери, делаем шаг оптимизации и 
        накапливаем потери для вычисления среднего значения
        """
        optimizer.zero_grad()
        indices = indices[:, :lengths.max()].to(device)
        target_indices = indices[:, 1:]
        logits = model(indices[:, :-1], lengths - 1)
        loss = criterion(logits.transpose(1, 2), target_indices)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * indices.shape[0]

    train_loss /= len(loader.dataset)
    return train_loss


@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Обрабатывает одну эпоху валидации модели
    :param model: языковая модель для валидации
    :param criterion: функция потерь
    :param loader: даталоадер для валидационной выборки
    :param tqdm_desc: описание для прогресс-бара
    :return: лосс валидации
    """
    device = next(model.parameters()).device
    val_loss = 0.0

    model.eval()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        """
        Проходим по батчам данных, 
        вычисляем потери и накапливаем
        их для вычисления среднего значения
        """
#        with torch.no_grad():
        indices = indices[:, :lengths.max()].to(device)
        target_indices = indices[:, 1:]
        logits = model(indices[:, :-1], lengths - 1)
        loss = criterion(logits.transpose(1, 2), target_indices)
        val_loss += loss.item() * indices.shape[0]


    val_loss /= len(loader.dataset)
    return val_loss


def train(model: LanguageModel, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, num_examples=5, model_name='model_number_'):
    """
    Обучаем языковую модель в течение нескольких эпох
    :param model: языковая модлель
    :param optimizer: оптимизатор для обновления параметров модели
    :param scheduler: планировщик для изменения скорости обучения
    :param train_loader: даталоадер для тренировочной выборки
    :param val_loader: даталоадер для валидационной выборки
    :param num_epochs: количество эпох обучения
    :param num_examples: количество примеров генерации текста, выводимых после каждой эпохи
    """
    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]

        if val_loss < 1e6:
            best_loss = val_loss
            torch.save(model.state_dict(), f'{model_name}_{val_loss}.pth')
            print(f'Saved the better model with loss={val_loss}')

        plot_losses(train_losses, val_losses)

        print('Generation examples:')
        for _ in range(num_examples):
            print(model.inference())
