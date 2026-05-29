import torch
from torch import nn
import math


def get_pad_output(src, src_feat, pad_token=1):
    """取得每筆文字最後一個有效 token 對應的 hidden state。"""
    # src 形狀為 [seq_len, batch]，True 代表該位置是 <pad>
    pad_idx = src == pad_token
    # argmax 找到第一個 <pad>，往前一格才是最後一個有效 token
    pad_idx = torch.argmax(pad_idx.int(), dim=0) - 1
    # 若整句都沒有 <pad>，argmax 原本會回傳 0；減 1 後剛好代表取序列最後一格
    _seq, b, _c = src_feat.size()
    # 為每筆 batch 建立自己的索引，才能一次取出不同長度句子的代表 hidden state
    batch_range = torch.arange(b, device=src_feat.device)
    return src_feat[pad_idx, batch_range, :]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Transformer 本身沒有順序概念，因此要替每個 token 位置加入固定的位置編碼
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term 控制不同 embedding 維度的波長，讓模型同時看到短距離與長距離位置差異
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # buffer 會跟著模型搬到 GPU，但不會被 optimizer 更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 形狀為 [seq_len, batch, emb_size]，只取目前序列長度需要的位置編碼
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, nhead, nhid, nlayers, nclass, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(emb_size, dropout)
        # EncoderLayer 內含 multi-head self-attention 與 feed-forward network
        encoder_layers = nn.TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        # 將 vocab index 轉成連續向量，輸出形狀為 [seq_len, batch, emb_size]
        self.embed = nn.Embedding(vocab_size, emb_size)
        # CLS token 是可學習的句子代表向量，最後用它做整篇新聞分類
        self.cls_token = nn.Parameter(torch.randn(emb_size))
        self.decoder = nn.Linear(emb_size, nclass)

    def forward(self, src, src_mask):
        src = self.embed(src)
        # 在每筆新聞最前面接上同一個 CLS token，讓 self-attention 聚合整句資訊
        cls_tokens = self.cls_token.repeat(1, src.shape[1], 1)
        src = torch.cat([cls_tokens, src], dim=0)
        src = self.pos_encoder(src)
        # src_key_padding_mask 形狀為 [batch, seq_len]，True 的位置不參與 attention
        src_mask = src_mask.T
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        # output[0] 對應 CLS token 的輸出，線性層轉成 4 類新聞 logits
        output = self.decoder(output[0])
        return output


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, nclass, nlayers=1):
        super(RNNClassifier, self).__init__()
        # 三種循環模型都使用相同輸入格式，方便比較 RNN/LSTM/GRU 的序列建模能力
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers=nlayers, batch_first=False)
        self.fc = nn.Linear(hidden_size, nclass)

    def forward(self, x):
        # x 形狀為 [seq_len, batch]，embedding 後為 [seq_len, batch, emb_size]
        xe = self.embedding(x)
        out, _ = self.rnn(xe)
        # 取每筆文字最後的有效 hidden state，再輸出分類 logits
        out = self.fc(get_pad_output(x, out))
        return out


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, nclass, nlayers=1):
        super(LSTMClassifier, self).__init__()
        # LSTM 多了 cell state，適合保留較長距離的文字資訊
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=nlayers, batch_first=False)
        self.fc = nn.Linear(hidden_size, nclass)

    def forward(self, x):
        xe = self.embedding(x)
        out, _ = self.lstm(xe)
        # 使用最後有效 token 的 hidden state 代表整篇新聞
        out = self.fc(get_pad_output(x, out))
        return out


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, nclass, nlayers=1):
        super(GRUClassifier, self).__init__()
        # GRU 以較少 gate 近似 LSTM，參數量通常比 LSTM 少
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=nlayers, batch_first=False)
        self.fc = nn.Linear(hidden_size, nclass)

    def forward(self, x):
        xe = self.embedding(x)
        out, _ = self.gru(xe)
        # 與 RNN/LSTM 相同，取最後有效 hidden state 做分類
        out = self.fc(get_pad_output(x, out))
        return out


if __name__ == '__main__':
    from torchinfo import summary
    from torch.utils.tensorboard import SummaryWriter
    from data_loader import DatasetProcessor
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_process = DatasetProcessor(batch_size, device)
    # 建立四種模型，輸入與輸出設定相同，方便比較不同序列模型
    vocab_size = len(data_process.vocab)
    emb_size = 32
    hidden_size = 128
    nclass = 4
    nlayers = 5

    transformer_model = TransformerClassifier(vocab_size, emb_size, nhead=4, nhid=hidden_size, nlayers=nlayers, nclass=nclass)
    rnn_model = RNNClassifier(vocab_size, emb_size, hidden_size, nclass, nlayers=nlayers)
    lstm_model = LSTMClassifier(vocab_size, emb_size, hidden_size, nclass, nlayers=nlayers)
    gru_model = GRUClassifier(vocab_size, emb_size, hidden_size, nclass, nlayers=nlayers)

    # 將模型圖與摘要寫入 TensorBoard，方便檢查各層輸入輸出形狀
    model_list = [transformer_model, rnn_model, lstm_model, gru_model]
    model_names = ['transformer', 'rnn', 'lstm', 'gru']

    for model, model_name in zip(model_list, model_names):
        model = model.to(device)
        writer = SummaryWriter(log_dir=f'./runs/{model_name}')
        # dummy_input 形狀維持 [seq_len, batch]，與 batch_first=False 的模型設定一致
        dummy_input = torch.zeros(64, 50).long()

        print(f'Generating graph for {model_name}...')
        if model_name == 'transformer':
            # Transformer 的 mask 需包含前方 CLS 位置；False 代表該位置可參與 attention
            cls_mask = torch.zeros((1, dummy_input.size(1)), dtype=torch.bool)
            dummy_mask = torch.cat([cls_mask, dummy_input == 1], dim=0).to(device)
            writer.add_graph(model, [dummy_input.to(device), dummy_mask])
            print(summary(model, input_data=[dummy_input.to(device), dummy_mask], device='cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            writer.add_graph(model, dummy_input.to(device))
            print(summary(model, input_data=dummy_input, device='cuda' if torch.cuda.is_available() else 'cpu'))
        print('\n\n')
        writer.close()

    print("Network graphs saved successfully!")
