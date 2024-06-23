import os
import torch
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class TextDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42
    VAL_RATIO = 0.05  # доля данных для валидации

    def __init__(self, data_file: str, train: bool = True, sp_model_prefix: str = None,
                 vocab_size: int = 2000, normalization_rule_name: str = 'nmt_nfkc_cf',
                 model_type: str = 'bpe', max_length: int = 128):
        """
        Dataset с текстами, поддерживающий токенизатор BPE
        :param data_file: путь к файлу с текстовыми данными формата txt
        :param train: использовать ли тренировочную выборку
        :param sp_model_prefix: префикс для сохранения модели токенизации
        :param vocab_size: размер словаря токенизатора
        :param normalization_rule_name: правило нормализации для токенизатора
        :param model_type: тип модели токенизатора
        :param max_length: максимальная длина текста в токенах
        """
        if not os.path.isfile(
                sp_model_prefix + '.model'):  # Если модель токенизатора не существует, она тренируется и сохраняется, затем загружается
            SentencePieceTrainer.train(
                input=data_file, vocab_size=vocab_size,
                model_type=model_type, model_prefix=sp_model_prefix,
                normalization_rule_name=normalization_rule_name,
                pad_id=3
            )
        # загрузка токенизатора из файла
        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix + '.model')

        # Загрузка текстов и их разбиение
        with open(data_file) as file:
            texts = file.readlines()

        train_texts, val_texts = train_test_split(texts, test_size=self.VAL_RATIO,
                                                  random_state=self.TRAIN_VAL_RANDOM_SEED)
        self.texts = train_texts if train else val_texts
        self.indices = self.sp_model.encode(self.texts)

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.sp_model.pad_id(), self.sp_model.unk_id(), \
                self.sp_model.bos_id(), self.sp_model.eos_id()
        self.max_length = max_length
        self.vocab_size = self.sp_model.vocab_size()

    def text2ids(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """
        Кодировка текста или списка текстов как токенизированные индексы
        :param texts: текст или список текстов для токенизации
        :return: закодированные индексы
        """
        return self.sp_model.encode(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        Декодировать индексы как текст или список токенов
        :param ids: 1D или 2D список (или torch.Tensor) индексов для декодирования
        :return: декодированные тексты
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.sp_model.decode(ids)

    def __len__(self):
        """
        Размер датасета
        :return: количество текстов в датасете
        """
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        """
        Добавить специальные предложения в массив индексов и дополнить их до максимальной длины
        :param item: id текста
        :return: индексы закодированного текста и его фактическая длина (включая специальные предложения BOS и EOS)
        """
        """
        Берем соответствующий массив индексов из self.indices,
        добавляем специальные токены (self.bos_id и self.eos_id) и 
        дополняем до self.max_length, используя паддинги self.pad_id.
        Возвращаем дополненные индексы размера (max_length, ) и его фактической длины.
        """
        encoded = [self.bos_id] + self.indices[item] + [self.eos_id]

        if len(encoded) <= self.max_length:
            padded_encoded = encoded + [self.pad_id] * (self.max_length - len(encoded))
        else:
            padded_encoded = encoded[:self.max_length]

        indices = torch.tensor(padded_encoded)
        length = min(len(encoded), self.max_length)

        return indices, length
