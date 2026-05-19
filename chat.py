import torch
from torch.nn import functional as F
from model.tokenizer import GPT2SubwordTokenizer
from model.architecture import MiniGPT

device = 'cpu'
tokenizer = GPT2SubwordTokenizer()
BLOCK_SIZE = 128

# --- AI Personality Settings ---
TEMPERATURE = 0.8  # 1.0 is default. Lower = stricter, Higher = crazier.
TOP_K = 10         # Only pick from the top 10 most likely next words.

print("Loading brain...")
model = MiniGPT()
model.load_state_dict(torch.load("checkpoints/abdel_ali.pt", map_location=device, weights_only=True))
model = model.to(device)
model.eval()

def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Takes the current context (idx) and generates the next words one by one.
    This replaces the simple argmax we used before with a 'weighted random roll'.
    """
    for _ in range(max_new_tokens):
        # crop context to max block size
        idx_cond = idx[:, -BLOCK_SIZE:]
        
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        # focus only on the last time step
        logits = logits[:, -1, :] 
        
        # Apply temperature (divide logits by temperature to flatten/sharpen probabilities)
        logits = logits / temperature
        
        # Optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        
        # sample from the distribution (rolling the weighted die)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Yield the new token so we can print it immediately in the terminal
        yield idx_next[0].tolist()

print("\nAbdel ALI is awake! (Type 'quit' to exit)")
print("=" * 50)

while True:
    try:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Abdel ALI: Goodbye!")
            break
            
        if not user_input.strip():
            print("Abdel ALI: (Please type something!)")
            continue
            
        context = torch.tensor([tokenizer.encode(user_input)], dtype=torch.long, device=device)
        print("Abdel ALI: ", end="", flush=True)
        
        # Use our new generate function to get 50 words
        for token_id in generate(model, context, max_new_tokens=50, temperature=TEMPERATURE, top_k=TOP_K):
            word = tokenizer.decode(token_id)
            print(word, end="", flush=True)
            
        print("\n" + "=" * 50)
        
    # Catch Ctrl+C cleanly so it doesn't print a massive error block
    except KeyboardInterrupt:
        print("\nAbdel ALI: Shutting down via KeyboardInterrupt. Goodbye!")
        break




    