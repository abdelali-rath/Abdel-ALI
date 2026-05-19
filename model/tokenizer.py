# Used OpenAIs open source tokenizer 

import tiktoken

class GPT2SubwordTokenizer:
    def __init__(self):
        # We use the 'gpt2' encoding, which has a vocabulary size of 50,257.
        # This keeps our model's final layer much lighter than GPT-4's 100k vocabulary.
        self.encoder = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.encoder.n_vocab

    def encode(self, string_input):
        """Translates text into a list of subword token IDs."""
        return self.encoder.encode(string_input)

    def decode(self, list_of_ints):
        """Translates a list of token IDs back into readable text."""
        return self.encoder.decode(list_of_ints)

# --- Quick Test ---
if __name__ == "__main__":
    tokenizer = GPT2SubwordTokenizer()
    
    print(f"Vocabulary size: {tokenizer.vocab_size} unique tokens")
    
    sample_text = "Abdelali Oumachi was here!"
    
    # Test Encoding
    encoded = tokenizer.encode(sample_text)
    print(f"\nOriginal text: '{sample_text}'")
    print(f"Encoded into token IDs: {encoded}")
    print(f"Number of tokens: {len(encoded)}")
    
    # Let's see how it broke down the words
    token_strings = [tokenizer.decode([token_id]) for token_id in encoded]
    print(f"How the AI sees it: {token_strings}")