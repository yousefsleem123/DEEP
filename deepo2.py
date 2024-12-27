# Libraries
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import tkinter as tk
from tkinter import messagebox, scrolledtext

# Reading data
filepath = 'shakespeare.txt'
url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
urllib.request.urlretrieve(url, filepath)

# Read and preprocess text
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# Debug text content
print(f"Start of text: {text[:100]}")  # Print the first 100 characters to verify the content

# Convert the text into numerical values
text = text[300000:800000]  # Use a slice of the text
characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

# Debug character mappings
print(f"Unique characters: {characters}")
print(f"Character to index mapping: {char_to_index}")

SEQ_LENGTH = 40  # How many letters to use for prediction
STEP_SIZE = 3    # Step size for shifting

# Preparing the features (sentences) and targets (next characters)
sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

# Debug prepared sentences and next characters
print(f"Sample sentences: {sentences[:3]}")
print(f"Sample next characters: {next_characters[:3]}")

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.float32)
y = np.zeros((len(sentences), len(characters)), dtype=np.float32)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

# Convert to PyTorch tensors
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define the PyTorch model
class TextGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TextGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.softmax(out)

# Model parameters
input_size = len(characters)
hidden_size = 128
output_size = len(characters)

# Initialize model, loss function, and optimizer
model = TextGenerator(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# Training the model
epochs = 4
batch_size = 256

for epoch in range(epochs):
    model.train()
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, torch.argmax(y_batch, dim=1))
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Sampling function
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Text generation function
def generate_text(length, temperature):
    model.eval()
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index:start_index + SEQ_LENGTH]
    generated += sentence

    for _ in range(length):
        x_pred = torch.zeros((1, SEQ_LENGTH, len(characters)), dtype=torch.float32)
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1.0

        with torch.no_grad():
            predictions = model(x_pred).squeeze(0).numpy()

        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]
        generated += next_character
        sentence = sentence[1:] + next_character

    return generated

# GUI functions and application
def on_generate():
    try:
        length = int(length_entry.get())
        temperature = float(temp_entry.get())
        if length <= 0:
            raise ValueError("Length should be a positive integer.")
        if not (0 < temperature <= 2):
            raise ValueError("Temperature should be between 0 and 2.")
        result = generate_text(length, temperature)
        output_area.delete(1.0, tk.END)
        output_area.insert(tk.END, result)
    except ValueError as ve:
        messagebox.showerror("Input Error", f"Invalid input: {ve}")

root = tk.Tk()
root.title("Text Generator")
root.geometry("500x400")

length_label = tk.Label(root, text="Text Length:")
length_label.pack(pady=5)
length_entry = tk.Entry(root)
length_entry.pack(pady=5)

temp_label = tk.Label(root, text="Temperature:")
temp_label.pack(pady=5)
temp_entry = tk.Entry(root)
temp_entry.pack(pady=5)

generate_button = tk.Button(root, text="Generate Text", command=on_generate)
generate_button.pack(pady=10)

output_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=10)
output_area.pack(pady=10)

root.mainloop()
