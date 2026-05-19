import torch
import os
from model.tokenizer import GPT2SubwordTokenizer
from model.architecture import MiniGPT


# --- Parameters ---
BATCH_SIZE = 4       # Text snippets to process at once
BLOCK_SIZE = 128     # Must match architecture.py
MAX_ITERS = 1000     # Training steps to run
EVAL_INTERVAL = 100  # How often to print the loss
LEARNING_RATE = 3e-4 # How big of a step the optimizer takes

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Training Abdel ALI on: {device}")

data_path = "data/input.txt"


# dataset fallback
if not os.path.exists(data_path):
    print("WARNING: data/input.txt not found. Creating a tiny dummy file to test.")
    os.makedirs("data", exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as f:
        f.write("Abdel ALI is a very smart AI. " * 1000)

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()


tokenizer = GPT2SubwordTokenizer()
# Convert text file into massive 1D tensor of token IDs
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)


# 90% train, 10% validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # Grabs random chunk of text for the model to read
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    
    # x is the input context, y is the target (same text, shifted by 1 token)
    x = torch.stack([d[i : i+BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i+1 : i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)


# --- Initialize Model ---
model = MiniGPT()
model = model.to(device)


# Weight update
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# ----- The Training Loop -----

print("Starting training...")
for iter in range(MAX_ITERS):
    
    # Every few steps, check how well we are doing on the validation set
    if iter % EVAL_INTERVAL == 0:
        model.eval() # Turn off training mode (stops Dropout)
        with torch.no_grad():
            x_val, y_val = get_batch('val')
            _, val_loss = model(x_val, y_val)
        print(f"Step {iter}: Validation Loss = {val_loss.item():.4f}")
        model.train() # Turn training back on

    # 1. Grab a batch of data
    xb, yb = get_batch('train')

    # 2. Forward Pass: Guess next words and calculate the error
    logits, loss = model(xb, yb)

    # 3. Backward Pass: Calculate gradients
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # 4. Update weights
    optimizer.step()



# --- Save the Model ---
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/abdel_ali.pt")
print("Training complete! Brain saved to checkpoints/abdel_ali.pt")














