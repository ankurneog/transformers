import torch 

class GPT2Config:
    def __init__(self,
                 model_name ="gpt2-124M",
                 vocab_size = 50257,
                 context_length = 1024,
                 embeding_dim = 768,
                 num_sdpa_layers = 12,
                 num_att_heads = 12,
                 ffn_inner_dim = 4*768,
                 bos_tokenid = 50256,
                 eos_tokenid =50256,
                 ):
        self.model_name = model_name
    
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embeding_dim
        self.num_sdpa_layers =num_sdpa_layers
        self.num_att_heads = num_att_heads
        self.ffn_inner_dim = ffn_inner_dim
