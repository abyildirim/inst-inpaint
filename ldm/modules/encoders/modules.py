import torch
import torch.nn as nn
from ldm.modules.x_transformer import Encoder, TransformerWrapper


class BERTTokenizer(nn.Module):
    def __init__(self, vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text, return_batch_encoding=False):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"]
        if return_batch_encoding:
            return tokens, batch_encoding
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text
    
class BERTEmbedder(nn.Module):
    """Uses the BERT tokenizer model and adds some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77, use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, cond, text):
        assert cond is None # Not supported for now (LDM conditioning key == "concat")
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)
            if next(self.transformer.parameters()).is_cuda:
                tokens = tokens.cuda()
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True) # Size: [batch_size, max_seq_len, n_embed]
        return z