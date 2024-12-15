import random
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Download NLTK data files
nltk.download('punkt')

# Generate random movie reviews and user preferences
movies = ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"]
users = ["User 1", "User 2", "User 3", "User 4", "User 5"]
reviews = [
    "I loved Movie A, it was fantastic!",
    "Movie B was okay, but not great.",
    "Movie C was amazing, I highly recommend it.",
    "Movie D was terrible, I did not enjoy it.",
    "Movie E was good, but not as good as Movie A."
]

# Tokenize the reviews
tokens = [word_tokenize(review) for review in reviews]
vocab = set(word for review in tokens for word in review)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# Convert tokens to indices
token_indices = [[word2idx[word] for word in review] for review in tokens]

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

# Create dataset and data loader
class TextDataset(Dataset):
    def __init__(self, token_indices, seq_len):
        self.token_indices = token_indices
        self.seq_len = seq_len

    def __len__(self):
        return max(len(self.token_indices) - self.seq_len, 0)

    def __getitem__(self, idx):
        src = self.token_indices[idx][:self.seq_len]
        tgt = self.token_indices[idx][1:self.seq_len+1]
        return torch.tensor(src), torch.tensor(tgt)

# Ensure max_len is not greater than the length of the token indices
max_len = min(max_len, len(token_indices) - 1)

dataset = TextDataset(token_indices, max_len)

# Check if the dataset is empty
if len(dataset) == 0:
    raise ValueError("The dataset is empty. Please ensure that the sequence length (max_len) is less than or equal to the length of the token indices.")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
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
torch.save(model.state_dict(), 'movie_recommendation_model.pth')
