"""
GPT2 Model
With inputs from : 
Hugging Face
https://huggingface.co/docs/transformers/en/model_doc/gpt2
A Karpathy : 
https://github.com/karpathy/nanoGPT/
Origin OpenAI paper :
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

Explanation and basic idea is documented here : 
transformers/gpt2/toy_gpt2.ipynb
Recap of GPT2 Model pipline 

Parameters Layers dmodel
------------------------
117M        12      768 
345M        24      1024 
762M        36      1280 
1542M       48       1600
-------------------------
vocab_size : 50257
context_size : 1024

[Tokenizer]

[Word Embeddings]

[Postional Embedding]

[TrasformerBlock1: [[Layer Norm]->[Self Attention Block x12]->[Drop Out]->[Layer Norm]->[FC layers]->[Drop Out]]

.....

[TransformerBlock12: [[Layer Norm]->[Self Attention Block x12]->[Drop Out]->[Layer Norm]->[FC layers]->[Drop Out] ] 

[Layer Normalization]->[GPT2 Workload Head/ Linear Layer]

"""
import torch
from utils.device import Device
import utils.logging
from utils.train_config import TrainConfig
from gpt2_config import GPT2Config

torch.manual_seed(12131306)

"""
Breaking down the transformers implementation 
[[Layer Norm]->[Self Attention Block x12]->[Drop Out]->[Layer Norm]->[FC layers]->[Drop Out]]
"""

class FFNetwork(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer_1    = torch.nn.Linear(config.n_embd, config.ffn_inner_dim )
        self.activation    = torch.nn.GELU(approximate='tanh')
        self.layer_2  = torch.nn.Linear(config.ffn_inner_dim, config.n_embd)

    def forward(self, x):
        out = self.layer_2(self.activation(self.layer_1(x)))
        return out 
    
class SelfAttentionBlock(torch.nn.Module):
    """ we will borrow a neat trick to create the QKV vectors which was shared by Andrej Karpathy in one of his youtube videos
        Basically for creating the qkv vectors , which has learnable parameter (weights,biases)
        we would first need to create a linear layer and then feed x through that layer , in essence
        doing a matmul of input with the weights , x @ Qw, x @ Kw and x @ Vw
        further since would create a multi headed attention block , we would also need to split the 
        vectors into chucks of head size , for GPT2 124M model the number of heads is 12 and hence
        these QKV vectors should be split into vectors with dimension 64 (768/12)
    """

    def __init__(self, config):
        super().__init__()
        """combined weights of QKV:
           in_features : embedding_dim==768, thats because x is (batch_size,context_length,embedding_dimension)
           out_features : the combined dimension of QKV, which is embedding_dim * 3
        """
        self.combined_qkv_weights = torch.nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        
        # output layer

        self.output_layer = torch.nn.Linear(config.embedding_dim, config.embedding_dim)
        self.config = config


    def forward(self, x): # x is our input tokens of shape (batch_size,context_length,embedding_dimension)

        """Now the fun part :
           x.shape ==[config.batch_size,config.context_length,config.embedding_dimension]
           combined_qkv_weights(x) will essentially do a malmul with the linear layer weights
           weights.shape of combined_qkv_weights == [config.embedding_dimension,3 * config.embedding_dimension] 
           the resulting tensor of the matmul (including projection of bias, bias.shape==out_features==3*config.embedding_dimension)
           is
           [batch_size,context_length,3*embedding_dimension]
        """
        qkv = self.combined_qkv_weights(x) 

        """
            split along the last dimension which is 3*embedding_dimension, the resultant is our q,k,v tensors
            shape [batch_size,context_length, embedding_dimension]
        """
        q, k, v = qkv.split(self.config.embedding_dimension, dim=2)
        
        """
            Now create the multi-headed attention (remember for multi-dimensional tensors , the last two dimensions are considered for matmul):
            so from batch_size, context_length, embedding_dimension ---> batch_size, context_length, num_heads, embedding_dimension//num_heads
            From : https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            Q shape : (N....L,E)
            K shape : (N....S,E)
            V shape : (N....S,Ev)
            output shape : (N,...L,Ev)
            where , 
            N : batch_size
            S : source sequence length
            L : Target sequence length
            E : Embedding dimension of Q and K
            Ev : Embedding dimension of V
            if we observe , the last 2 dimension, which is important for matmul needs to be L, E which is context_length, embedding_dimension (or Head size for
            multi-headed attention)
            Further we have to reshape our original q,k,v vectors to meet the pytorch API requirement, hence we need to do transpose(1,2)


        """
        heads = self.config.num_att_heads
        individual_head_dim = self.config.embedding_dimension // heads # 768/12 ==64
            
        q = q.view(self.config.batch_size, self.config.context_size, heads, individual_head_dim).transpose(1, 2) 

        k = k.view(self.config.batch_size, self.config.context_size, heads, individual_head_dim).transpose(1, 2) 
       
        v = v.view(self.config.batch_size, self.config.context_size, heads, individual_head_dim).transpose(1, 2)
       
       # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True) # is_causal (bool) – If true, assumes upper left causal attention masking
       
        """
        The ouput of the API as stated above is of dimension (batch_size, head, context_length,individual_head_dim)
        with transpose we bring it back to (batch_size,context_length, heads,individual_head_dim)
        """
        y = y.transpose(1, 2).contiguous().view(self.config.batch_size, self.config.context_length, self.config.embedding_dimension) # combine the heads
        
        """output linear layer : after concatenation and before projection , y.shape :(batch_size,context_length, embedding_dimension)
           after linear layer y.shape :(batch_size,context_length, embedding_dimension), since the out feature of the linear layer is embedding_dimension
        """

        y = self.output_layer(y)
       
        return y   

"""
    Stiching it all together, we now create the complete transformer pipeline for GPT2
"""

class GPT2TransformerBlock(torch.nn.Module):
        
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = torch.nn.LayerNorm(config.embedding_dim)
        self.self_attention_block = SelfAttentionBlock(config)
        self.layer_norm_2 = torch.nn.LayerNorm(config.embedding_dim)
        self.fully_connected_network = FFNetwork(config)

    def forward(self, x):
        x = x + self.self_attention_block(self.layer_norm_1(x))
        x = x + self.fully_connected_network(self.layer_norm_2(x))
        return x


