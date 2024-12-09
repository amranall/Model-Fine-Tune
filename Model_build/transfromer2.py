import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Sample dataset for conversations
conversations = [
    "User: Recommend a movie. Bot: I recommend 'Inception'.",
    "User: Suggest a book. Bot: You might enjoy '1984' by George Orwell.",
    "User: What's the weather like? Bot: It looks sunny today.",
    "User: Recommend a song. Bot: I recommend 'Bohemian Rhapsody' by Queen.",
    "User: Suggest a series. Bot: Try watching 'Stranger Things'.",
]

# Tokenize the text and create vocabulary
tokens = []
for convo in conversations:
    tokens.extend(word_tokenize(convo))

vocab = set(tokens)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# Encode the conversations into token indices
encoded_conversations = []
for convo in conversations:
    encoded_conversations.append([word2idx[word] for word in word_tokenize(convo)])

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_len, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
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
num_epochs = 20

# Initialize the model, loss function, and optimizer
model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Prepare training data
src_sequences = []
tgt_sequences = []

for convo in encoded_conversations:
    src = convo[:-1]  # Input is everything except the last token
    tgt = convo[1:]   # Target is everything except the first token
    src_sequences.append(src)
    tgt_sequences.append(tgt)

# Pad sequences to max_len
def pad_sequence(seq, max_len, pad_idx=0):
    return seq + [pad_idx] * (max_len - len(seq))

src_sequences = [pad_sequence(seq, max_len) for seq in src_sequences]
tgt_sequences = [pad_sequence(seq, max_len) for seq in tgt_sequences]

src_tensor = torch.tensor(src_sequences)
tgt_tensor = torch.tensor(tgt_sequences)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    src = src_tensor
    tgt = tgt_tensor

    output = model(src, tgt)
    loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Generate text
def generate_response(model, user_input, max_len=50):
    model.eval()
    tokens = word_tokenize(user_input)
    input_seq = torch.tensor([[word2idx.get(token, 0) for token in tokens]])
    generated_tokens = []

    for _ in range(max_len):
        output = model(input_seq, input_seq)
        next_token = torch.argmax(output[0, -1, :]).item()
        generated_tokens.append(next_token)
        if idx2word[next_token] == "<end>":  # Use an end token to stop generation
            break
        input_seq = torch.tensor([list(input_seq[0]) + [next_token]])
    
    response = ' '.join([idx2word[token] for token in generated_tokens])
    return response

# Add a recommendation system
recommendations = {
    "movie": ["Inception", "Interstellar", "The Matrix"],
    "book": ["1984", "To Kill a Mockingbird", "Sapiens"],
    "series": ["Stranger Things", "Breaking Bad", "Dark"],
}

def get_recommendation(user_input):
    for keyword, items in recommendations.items():
        if keyword in user_input.lower():
            return f"I recommend: {', '.join(items)}."
    return "I'm not sure, but I can help you find something!"

# Chat function
def chatbot(user_input):
    if any(keyword in user_input.lower() for keyword in recommendations.keys()):
        return get_recommendation(user_input)
    else:
        return generate_response(model, user_input)

# Example interaction
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    response = chatbot(user_input)
    print(f"Bot: {response}")
