import os
import re
from collections import Counter

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class SimpleVocab(object):
    def __init__(self, token_to_idx):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token for token, idx in token_to_idx.items()}

    def __contains__(self, token):
        return token in self.token_to_idx

    def __getitem__(self, token):
        return self.token_to_idx[token]

    def __len__(self):
        return len(self.token_to_idx)


def basic_english_tokenizer(text):
    # 將英文轉小寫後切成單字、數字與標點
    return re.findall(r"[a-z0-9]+|[^\s\w]", text.lower())


class DatasetProcessor(object):
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = basic_english_tokenizer
        self.vocab = None

        # 載入 AG News 訓練與測試資料
        train_iter, test_iter = self.load_ag_news()
        self.train_iter = train_iter
        self.test_iter = test_iter
        # tqdm 需要每個 epoch 的 batch 數；這裡用整除對齊 DataLoader 的主要批次數
        self.train_len = len(train_iter) // batch_size
        self.test_len = len(test_iter) // batch_size

        # <unk> 處理未出現在訓練集的字，<pad> 用來補齊同一批次內不同長度的句子
        self.specials = ['<unk>', '<pad>']
        self.build_vocab(train_iter, self.specials)
        self.train_dataloader = self.get_dataloader(train_iter, shuffle=True)
        self.test_dataloader = self.get_dataloader(test_iter)

    def load_ag_news(self):
        try:
            dataset = load_dataset('ag_news')
        except ImportError as exc:
            raise ImportError("請先安裝 Hugging Face datasets：pip install datasets") from exc

        # load_dataset 讀到的 label 是 0~3；這裡先轉成 1~4，collate_fn 再統一轉回 0~3
        train_iter = [(item['label'] + 1, item['text']) for item in dataset['train']]
        test_iter = [(item['label'] + 1, item['text']) for item in dataset['test']]
        return train_iter, test_iter

    def yield_tokens(self, data_iter):
        # build_vocab 只需要 token 序列，不需要 label
        for _, text in data_iter:
            yield self.tokenizer(text)

    def build_vocab(self, data_iter, special_tokens, cache_path='cache/hf_vocab_cache.pt', use_cache=True):
        # 詞彙表只根據訓練集建立，避免測試集資訊洩漏到訓練流程
        if use_cache and os.path.exists(cache_path):
            token_to_idx = torch.load(cache_path)
        else:
            counter = Counter()
            for tokens in self.yield_tokens(data_iter):
                counter.update(tokens)

            # special tokens 固定放在最前面，讓 padding 與 mask 使用穩定的 index
            token_to_idx = {token: idx for idx, token in enumerate(special_tokens)}
            for token, _ in counter.most_common():
                if token not in token_to_idx:
                    token_to_idx[token] = len(token_to_idx)

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(token_to_idx, cache_path)

        self.vocab = SimpleVocab(token_to_idx)

    def t2i(self, token):
        # 未登入詞彙表的 token 會映射到 <unk>，避免推論時遇到新字造成 KeyError
        return self.vocab[token] if token in self.vocab else self.vocab['<unk>']

    def get_dataloader(self, data_iter, shuffle=False):
        def process_data(batch):
            # collate_fn 負責把原始文字批次轉成模型可吃的 tensor
            # text_list 最後形狀為 [seq_len, batch]，符合 network.py 中 batch_first=False 的設定
            label_list, text_list, offsets = [], [], [0]
            for (_label, _text) in batch:
                # CrossEntropyLoss 需要 class index 從 0 開始，因此將 1~4 轉成 0~3
                label_list.append(_label - 1)
                processed_text = torch.tensor([self.t2i(token) for token in self.tokenizer(_text)], dtype=torch.long)
                text_list.append(processed_text)
                offsets.append(processed_text.size(0))

            label_list = torch.tensor(label_list, dtype=torch.long)
            # offsets 記錄每筆文字在攤平成一維序列時的起點
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            # 不同長度的句子用 <pad> 補到同長度，模型端再透過 pad index 找出有效文字範圍
            text_list = pad_sequence(text_list, padding_value=self.vocab['<pad>'])
            return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)

        return DataLoader(data_iter, batch_size=self.batch_size, shuffle=shuffle, collate_fn=process_data)
