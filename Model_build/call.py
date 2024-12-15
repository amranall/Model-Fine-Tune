import torch
import torch.nn as nn
import nltk
from nltk.tokenize import word_tokenize
import pickle

# Download NLTK data files
nltk.download('punkt')

# Load the vocabulary
with open('vocab.pkl', 'rb') as f:
    word2idx, idx2word = pickle.load(f)

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
vocab_size = len(word2idx)
d_model = 128
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 512
max_len = 50

# Initialize the model with the new vocab size
model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len)

# Load the fine-tuned model state dictionary with weights_only=True
model.load_state_dict(torch.load('movie_recommendation_model.pth', map_location=torch.device('cpu'), weights_only=True))

# Generate text function
def generate_text(model, start_token, max_len=50):
    model.eval()
    input_seq = torch.tensor([[start_token]])
    generated_tokens = [start_token]

    for _ in range(max_len):
        output = model(input_seq, input_seq)
        next_token = torch.argmax(output[0, -1, :]).item()
        generated_tokens.append(next_token)
        input_seq = torch.tensor([generated_tokens])

    return ' '.join([idx2word[token] for token in generated_tokens])

# Generate movie recommendations based on user input
def recommend_movies(user_input):
    # Tokenize the user input
    tokens = word_tokenize(user_input)

    # Update the vocabulary with the tokens from the user input
    for token in tokens:
        if token not in word2idx:
            word2idx[token] = len(word2idx)
            idx2word[len(idx2word)] = token

    # Update the vocabulary size
    vocab_size = len(word2idx)

    # Update the embedding and fc_out layers with the new vocab size
    model.embedding = nn.Embedding(vocab_size, d_model)
    model.fc_out = nn.Linear(d_model, vocab_size)

    # Convert tokens to indices
    token_indices = [word2idx[token] for token in tokens]
    start_token = token_indices[0]

    # Generate text
    generated_text = generate_text(model, start_token)
    return generated_text

# Example user input
user_input = "I want to watch a movie similar to Movie A."
recommended_movies = recommend_movies(user_input)
print(recommended_movies)





# import torch
# import torch.nn as nn
# import nltk
# from nltk.tokenize import word_tokenize

# # Download NLTK data files
# nltk.download('punkt')

# # Additional text data for fine-tuning
# additional_text = "Here is some more text for fine-tuning the Transformer model. This should help the model learn better."

# # Tokenize the additional text
# new_tokens = word_tokenize(additional_text)
# vocab = set(new_tokens)
# word2idx = {word: idx for idx, word in enumerate(vocab)}
# idx2word = {idx: word for word, idx in word2idx.items()}

# # Define the Transformer Model
# class TransformerModel(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len):
#         super(TransformerModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.pos_encoder = nn.Embedding(max_len, d_model)
#         self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, batch_first=True)
#         self.fc_out = nn.Linear(d_model, vocab_size)
#         self.d_model = d_model
#         self.max_len = max_len

#     def forward(self, src, tgt):
#         src_seq_len = src.shape[1]
#         tgt_seq_len = tgt.shape[1]

#         src_pos = torch.arange(0, src_seq_len).unsqueeze(0).expand(src.shape[0], src_seq_len).to(src.device)
#         tgt_pos = torch.arange(0, tgt_seq_len).unsqueeze(0).expand(tgt.shape[0], tgt_seq_len).to(tgt.device)

#         src = self.embedding(src) + self.pos_encoder(src_pos)
#         tgt = self.embedding(tgt) + self.pos_encoder(tgt_pos)

#         transformer_out = self.transformer(src, tgt)
#         output = self.fc_out(transformer_out)
#         return output

# # Hyperparameters
# vocab_size = len(vocab)
# d_model = 128
# nhead = 8
# num_encoder_layers = 3
# num_decoder_layers = 3
# dim_feedforward = 512
# max_len = 50

# # Initialize the model with the new vocab size
# model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len)

# # Load the fine-tuned model state dictionary
# model.load_state_dict(torch.load('fine_tuned_model.pth', map_location=torch.device('cpu')))

# # Generate text function
# def generate_text(model, start_token, max_len=50):
#     model.eval()
#     input_seq = torch.tensor([[start_token]])
#     generated_tokens = [start_token]

#     for _ in range(max_len):
#         output = model(input_seq, input_seq)
#         next_token = torch.argmax(output[0, -1, :]).item()
#         generated_tokens.append(next_token)
#         input_seq = torch.tensor([generated_tokens])

#     return ' '.join([idx2word[token] for token in generated_tokens])

# # Generate random text
# start_token = word2idx['This']
# generated_text = generate_text(model, start_token)
# print(generated_text)






