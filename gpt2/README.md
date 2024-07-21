## Overview 
Hugging Face provides a very  comprehensive overview of openAI GPT2 
https://huggingface.co/docs/transformers/en/model_doc/gpt2

GPT2 uses the decoder part of the orginal transformers with some modification 
Details in the original paper : https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.
GPT2 was one of the first models to utilize unsupervised fine-tuning.
The crux being the self attention mechanism , where by we calculate the attention score of each token w.r.t 
all other tokens in the sample

Prediction is auto-regressive in nature 
eg: nth token is predicted based on tn-1, tn-2....t1 tokens 

The Attention score is computed based on the Scaled Dot Product Attention Formula

A(Q,K,V) = softmax(Q@K.transpose(-2,-1)/(K.shape(-1))**0.5)@V

Where Q,K,V are the vectors computed during the traing process.
The Dimension of the Q,K ,V is embedding_dimension x (embedding_dimension/num_attention_head)
If there was only one self attention head it would be embedding_dimension x embedding_dimension 
GPT2's embedding_dimension is 764 and number of attention heads is 12 , hence the dimension is 
764 x 64

All these numbers are usually based on Hieuristics 

## GPT2 Blocks
Components of the GPT2 blocks can be described as Below:

[Tokenizer]->[Word Embeddings]->[Postional Embedding]->[TrasformerBlock1: [[Layer Norm]->[Self Attention Block x12]->[Drop Out]->[Layer Norm]->[FC layers]->[Drop Out]].....[TransformerBlock12: [[Layer Norm]->[Self Attention Block x12]->[Drop Out]->[Layer Norm]->[FC layers]->[Drop Out] ] ->[Layer Normalization]->[Linear Layer]--->[GPT2 Workload Head]

## GPT2 Configuration Attributes

The vocabulary Dimension : vocab_size
The Embedding Dimension : embedding_size
Number of transformer Layers :  num_self_attention_blocks
Number of heads per attention block : num_heads
Dimension of the Weights of the Q,K,V vectors : embedding_size x (embedding_size/num_heads)
Dimension of the Feed Forward (FF) layer : 4 * embedding_size
Dimension of the Linear Layer : vocab_size

Others :
Drop Out Ratio of DropOut Layers : dropout ( usually 0.2 to 0.1)
Epsilon of the Layer Norm : ep_layerNorm (usually 1e-5)

## GPT2 Model heads 
The workload head can be plugged in to do various tasks, some of which can be as follows : 

1. Classification Head
This head is used for tasks where the model needs to classify the input into predefined categories. It typically consists of one or more fully connected layers followed by a softmax activation function to output probabilities for each class.
2. Regression Head
Used for tasks that require predicting continuous values, such as predicting prices or coordinates. This head usually consists of fully connected layers without an activation function at the output layer (or with a linear activation).
3. Sequence-to-Sequence Head
This head is suitable for tasks like translation or summarization, where the model generates a sequence of outputs based on the input sequence. It often involves a decoder structure that can process the output from the GPT-2 block.
4. Token Classification Head
Used for tasks like Named Entity Recognition (NER) or Part-of-Speech (POS) tagging, where each token in the input sequence needs to be classified. This head typically outputs a classification for each token.
5. Multi-Task Head
This head can handle multiple tasks simultaneously, combining different types of outputs (e.g., classification and regression) from a single model. It may involve multiple branches in the architecture.
6. Attention Heads
Additional attention heads can be added for specialized attention mechanisms, allowing the model to focus on different aspects of the input data. This is particularly useful in multi-head attention setups.
7. Mixture of Experts Head
This head leverages multiple expert models, activating only a subset of them for each input, which can improve efficiency and performance on diverse tasks.
8. Contrastive Learning Head
Used in self-supervised learning scenarios, this head is designed to maximize the similarity between positive pairs and minimize it between negative pairs, often implemented using a projection head followed by a contrastive loss.
9. Generative Head
For tasks involving text generation, this head can be designed to produce sequences based on the learned representations, often using techniques like beam search or sampling.
