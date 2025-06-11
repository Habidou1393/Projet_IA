import os
from tokenizers import ByteLevelBPETokenizer

def train_tokenizer():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=["data/mon_corpus.txt"], vocab_size=5000, min_frequency=2)

    save_dir = "tokenizer"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tokenizer.save_model(save_dir)
    print("Tokenizer entraîné et sauvegardé dans 'tokenizer/'")

if __name__ == "__main__":
    train_tokenizer()
