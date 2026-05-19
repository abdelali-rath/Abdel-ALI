import gradio as gr
import torch
from torch.nn import functional as F
from model.tokenizer import GPT2SubwordTokenizer
from model.architecture import MiniGPT


# --- Setup & Load Model ---
device = 'cpu'
tokenizer = GPT2SubwordTokenizer()
BLOCK_SIZE = 128
TEMPERATURE = 0.8
TOP_K = 10

print("Loading Abdel ALI's brain for the web...")
model = MiniGPT()
model.load_state_dict(torch.load("checkpoints/abdel_ali.pt", map_location=device, weights_only=True))
model = model.to(device)
model.eval()


# --- The Brain Interface ---
def generate_response(message, history):

    if not message.strip():
        yield "(Please type something!)"
        return

    # Translate human message to numbers
    context = torch.tensor([tokenizer.encode(message)], dtype=torch.long, device=device)
    
    response_string = ""
    
    # Generate 50 words
    for _ in range(50):
        idx_cond = context[:, -BLOCK_SIZE:]
        
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        logits = logits[:, -1, :] / TEMPERATURE
        
        if TOP_K is not None:
            v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, idx_next), dim=1)
        
        # Translate new token back to a word
        word = tokenizer.decode(idx_next[0].tolist())
        response_string += word
        
        # yield pushes the updated string to the UI instantly, creating the typing effect
        yield response_string


# Web UI
# Gradio ChatInterface handles HTML, CSS, and chat history
demo = gr.ChatInterface(
    fn=generate_response,
    title="Abdel ALI",
    description="I aint finna lie this AI is still horribly bad",
    examples=["In Dusseldorf", "A large", "Morocco is a"],
)



if __name__ == "__main__":
    demo.launch()