# %%
import torch 
import torch.nn as nn
import torch.optim as optim


VOCAB_SIZE = 90  # 86 characters + 4 special tokens (padding, out-of-vocabulary, beginning of line and end of line)
EMBED_DIM = 8
LSTM_UNITS = 256
SEQUENCE_LENGTH = 80

# %%
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units, sequence_length):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm1 = nn.LSTM(embed_dim, lstm_units, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.fc = nn.Linear(lstm_units, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

model = CharRNN(VOCAB_SIZE, EMBED_DIM, LSTM_UNITS, SEQUENCE_LENGTH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)
# %%
