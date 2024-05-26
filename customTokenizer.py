class CustomTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab if vocab is not None else {}

    def tokenize(self, text):
        tokens = text.split()
        return tokens

    def build_vocab(self, texts):
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab[token] for token in tokens if token in self.vocab]

    def decode(self, token_ids):
        id_to_token = {id: token for token, id in self.vocab.items()}
        return [id_to_token[id] for id in token_ids]

tokenizer = CustomTokenizer()