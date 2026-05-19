# Abdel-ALI

<img width="2000" height="2000" alt="Abdel ALI (2)" src="https://github.com/user-attachments/assets/1af9877d-f596-4e46-a3cb-2c489874f223" />

### Abdel ALI is a miniaturized Generative Pre-trained Transformer (GPT) built entirely from scratch using PyTorch. This project demonstrates the fundamental mathematics and software engineering behind modern Large Language Models (LLMs). 

### It features a custom-built Transformer architecture, a backpropagation training loop, and both Command Line and Web-based chat interfaces.

---

## 🚀 Features
* **Transformer Architecture from Scratch:** Implements Token/Positional Embeddings, Multi-Head Self-Attention, and Feed-Forward neural networks directly in PyTorch.
* **Subword Tokenization:** Utilizes OpenAI's `tiktoken` (cl100k_base/gpt2) for efficient subword encoding, mirroring the tokenization strategies of GPT-3 and GPT-4.
* **Custom Training Loop:** Includes a complete `train.py` script featuring cross-entropy loss calculation, AdamW optimization, and model checkpointing.
* **Creative Generation:** The chat inference script utilizes Temperature and Top-K sampling for natural and dynamic text generation.
* **Dual Interfaces:** Interact with Abdel ALI via the terminal (`chat.py`) or a local web GUI (`ui.py`) powered by Gradio.

---

## 📂 Repository Structure
```
Abdel-ALI/
│
├── checkpoints/            # Directory for saved model weights (.pt files)
├── data/                   # Directory for training datasets (.txt)
│
├── model/                  # The core AI architecture
│   ├── __init__.py
│   ├── architecture.py     # PyTorch Transformer network (Attention, Blocks, etc.)
│   └── tokenizer.py        # Subword tokenizer wrapper using tiktoken
│
├── train.py                # Script to train the model on your custom dataset
├── chat.py                 # Command Line Interface (CLI) chat script
├── ui.py                   # Graphical Web Interface using Gradio
│
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules for large datasets and weights
└── README.md               # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/abdelali-rath/Abdel-ALI.git
cd Abdel-ALI
```

### 2. Install the required dependencies:
It is recommended to use a virtual environment.
```
pip install -r requirements.txt
# still empty tho
```
(Requires `torch`, `tiktoken`, and `gradio`)

### 3. Prepare the Data:

Create a `data` folder in the root directory. Place your raw text dataset (e.g., Wikipedia articles, movie scripts, or book texts) inside and name it `input.txt`.

(Note: The `data/` folder is ignored by Git due to GitHub's file size limits).

---

## 🧠 Training the Model

To train Abdel ALI on your dataset, run the training script:
```
python train.py
```
• Hardware Note: The script automatically detects if you have a CUDA-enabled GPU, an Apple Silicon MPS, or a CPU. If you are on an AMD GPU without DirectML configured, it will safely default to your CPU.

• Checkpoints: Once training completes (or is manually stopped), the model's weights will be automatically saved to `checkpoints/abdel_ali.pt`.

---

## 💬 Chatting with Abdel ALI
Once the model is trained, you can talk to it using one of two interfaces:

### Option 1: Command Line Interface (CLI)
Run the chat script directly in your terminal:
```
python chat.py
```

### Option 2: Web User Interface (GUI)
Launch the Gradio web server for a clean, ChatGPT-style interface:
```
python ui.py
```
After running the command, open the provided local URL (usually `http://127.0.0.1:7860`) in your web browser.

---

## ⚙️ Hyperparameters

The model's brain size and learning rates can be easily adjusted.

  • To change the model's depth or context window, edit the constants at the top of `model/architecture.py`.

  • To adjust the learning rate or batch size, edit the constants in `train.py`.

  • To adjust the AI's creativity, edit the `TEMPERATURE` and `TOP_K` variables in `chat.py` or `ui.py`.


***Disclaimer: Abdel ALI's intelligence is directly correlated to the size of the dataset and the length of the training time provided by the user. Hallucinations and nonsensical grammar are expected during early training epochs!***



