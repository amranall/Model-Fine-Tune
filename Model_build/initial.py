import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, Dataset

# Download NLTK data files
nltk.download('punkt')

# Sample initial text data
initial_text = "This is a sample text for generating random text using a Transformer model."

# Tokenize the text
tokens = word_tokenize(initial_text)
vocab = set(tokens)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# Convert tokens to indices
token_indices = [word2idx[token] for token in tokens]

# Define the Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_len, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        src_pos = torch.arange(0, src_seq_len).unsqueeze(0).expand(src.shape[0], src_seq_len).to(src.device)
        tgt_pos = torch.arange(0, tgt_seq_len).unsqueeze(0).expand(tgt.shape[0], tgt_seq_len).to(tgt.device)

        src = self.embedding(src) + self.pos_encoder(src_pos)
        tgt = self.embedding(tgt) + self.pos_encoder(tgt_pos)

        transformer_out = self.transformer(src, tgt)
        output = self.fc_out(transformer_out)
        return output

# Hyperparameters
vocab_size = len(vocab)
d_model = 128
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 512
max_len = 50
learning_rate = 0.001
num_epochs = 10

# Initialize the model, loss function, and optimizer
model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    src = torch.tensor(token_indices[:-1]).unsqueeze(0)
    tgt = torch.tensor(token_indices[1:]).unsqueeze(0)

    output = model(src, tgt[:, :-1])
    loss = criterion(output.view(-1, vocab_size), tgt[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'initial_model.pth')
