import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Модель для генерации текста
        :param dataset: объект типа TextDataset, который предоставляет словарь (vocab_size) и параметр максимальной длины текста (max_length).
        :param embed_size: размерность эмбеддингов (векторного представления слов)
        :param hidden_size: размерность скрытого состояния RNN
        :param rnn_type: тип рекуррентного слоя (по умолчанию nn.RNN, можно использовать nn.LSTM)
        :param rnn_layers: количество слоев RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # требуется для декодирования во время вывода
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        # cоздаем необходимые слои
        self.embedding = nn.Embedding(self.vocab_size, embed_size, padding_idx=dataset.pad_id)
        self.rnn = rnn_type(embed_size, hidden_size, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели и возвращение логитов для вероятностей следующего токена
        :param indices: тензор с индексами токенов (размерность: размерность батча, длина)
        :param lengths: длины последовательностей в батче
        :return: FloatTensor логитов (batch_size, length, vocab_size)
        """
        logits = torch.randn(
            indices.shape[0], indices.shape[1], self.vocab_size,
            device=indices.device
        )
        """
        Преобразовываем индексы в эмбеддинги
        пропускаем через рекуррентные слои
        и применяем выходной линейный слой для получения логитов
        """
        '''
        B - размер батча
        L - длина последовательности
        E - размерность эмбеддинга
        H - размерность hidden
        V - размер словаря
        '''
        # tokens: (B, L)
        embeds = self.embedding(indices)
        # embeds: (B, L, E)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed_embeds)
        # output: (B, L, H), hidden: (B, H)
        outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
        logits = self.linear(outputs)
        # logits: (B, L, V)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        генерация текста с заданным префиксом
        :param prefix: начальный текст, с которого начинается генерация
        :param temp: параметр температуры для управления случайностью генерации
        :return: сгенерированный текст
        """
        self.eval()
        """
        Кодируем префикс, прогоняем его через модель
        для получения скрытых состояний, затем пошагово 
        генерируем новые токены до достижения 
        максимальной длины или символа конца последовательности (<eos>)
        """
        tokens = [self.dataset.bos_id] + self.dataset.text2ids(prefix)  # [bos, prefix]
        tokens = torch.tensor(tokens).unsqueeze(0).to(next(self.parameters()).device)

        # создание hidden для префикса
        embeds = self.embedding(tokens)
        output, hidden = self.rnn(embeds)
        logits = self.linear(output) / temp

        # образец нового токена из logits
        new_tokens = Categorical(logits=logits[:, -1:]).sample()
        tokens = torch.cat([tokens, new_tokens], dim=1)

        # критерии остановы
        while tokens.shape[1] < self.max_length:
            if new_tokens.item() == self.dataset.eos_id:
                break

            # обработаем вновь полученный токен
            embeds = self.embedding(new_tokens)
            output, hidden = self.rnn(embeds, hidden)
            logits = self.linear(output) / temp
            # выбор следующего токена из логитов
            new_tokens = Categorical(logits=logits[:, -1:]).sample()
            tokens = torch.cat([tokens, new_tokens], dim=1)

        generated = self.dataset.ids2text(tokens)[0]
        return generated
