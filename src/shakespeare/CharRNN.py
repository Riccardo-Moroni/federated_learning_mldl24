# %%
import torch 
import torch.nn as nn
import torch.optim as optim

# %%
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units, sequence_length):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm1 = nn.LSTM(embed_dim, lstm_units, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True, bias=False)
        self.fc = nn.Linear(lstm_units, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
# %%
# model infos

# vocab_size = 90 # 86 characters + 4 special tokens (padding, out-of-vocabulary, beginning of line and end of line)
# embedding_dim = 8
# hidden_size = 256
# num_layers = 2

# model = CharRNN(vocab_size, embedding_dim, hidden_size, num_layers)

# # print trainable parameters per layer
# def print_trainable_parameters(model):
#     print(f"{'Layer':<30} {'Parameters':<10}")
#     print("="*40)
#     total_params = 0
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             num_params = param.numel()
#             total_params += num_params
#             print(f"{name:<30} {num_params:<10}")
#     print("="*40)
#     print(f"Total trainable parameters: {total_params}")

# print_trainable_parameters(model)
# %%
