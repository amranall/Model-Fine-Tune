import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, Dataset

# Download NLTK data files
nltk.download('punkt')

# Additional text data for fine-tuning
additional_text = "Here is some more text for fine-tuning the Transformer model. This should help the model learn better."

# Tokenize the additional text
new_tokens = word_tokenize(additional_text)
vocab = set(new_tokens)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# Convert tokens to indices
new_token_indices = [word2idx[token] for token in new_tokens]

# Update vocab size
vocab_size = len(vocab)

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
d_model = 128
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 512
max_len = 50
learning_rate = 0.001
num_epochs = 10

# Initialize the model with the new vocab size
model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len)

# Load the initial model state dictionary with weights_only=True
initial_state_dict = torch.load('initial_model.pth', map_location=torch.device('cpu'), weights_only=True)

# Update the model with the initial state dictionary, excluding the embedding and fc_out layers
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in initial_state_dict.items() if k in model_dict and k not in ['embedding.weight', 'fc_out.weight', 'fc_out.bias']}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# Initialize the embedding and fc_out layers with the new vocab size
model.embedding = nn.Embedding(vocab_size, d_model)
model.fc_out = nn.Linear(d_model, vocab_size)

# Initialize the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create dataset and data loader
class TextDataset(Dataset):
    def __init__(self, token_indices, seq_len):
        self.token_indices = token_indices
        self.seq_len = seq_len

    def __len__(self):
        return max(len(self.token_indices) - self.seq_len, 0)

    def __getitem__(self, idx):
        src = self.token_indices[idx:idx+self.seq_len]
        tgt = self.token_indices[idx+1:idx+self.seq_len+1]
        return torch.tensor(src), torch.tensor(tgt)

# Ensure max_len is not greater than the length of the token indices
max_len = min(max_len, len(new_token_indices) - 1)

dataset = TextDataset(new_token_indices, max_len)

# Check if the dataset is empty
if len(dataset) == 0:
    raise ValueError("The dataset is empty. Please ensure that the sequence length (max_len) is less than or equal to the length of the token indices.")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Fine-tuning loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.view(-1, vocab_size), tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')

# Save the fine-tuned model
torch.save(model.state_dict(), 'fine_tuned_model.pth')
