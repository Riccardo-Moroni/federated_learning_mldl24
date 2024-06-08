# %% 
import torch.nn as nn

# %%
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.stacked_lstm = nn.LSTM(embed_dim, lstm_units, num_layers=2, batch_first=True)
        self.fc = nn.Linear(lstm_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        outputs, _ = self.stacked_lstm(x) # outputs shape is (batch_size, sequence_length, lstm_units)
        pred = self.fc(outputs[:, -1, :]) # pred shape is (batch_size, vocab_size)
        return pred
    
# %%
# model infos

# vocab_size = 70 # circa
# embedding_dim = 8
# lstm_units = 256

# model = CharRNN(vocab_size, embedding_dim, lstm_units)

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