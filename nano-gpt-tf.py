"""Tensorflow implementation of nano GPT

@author: Alejandro Velez Arce | amva13@alum.mit.edu | GH: amva13
@info: assumes cpu

"""

#imports - minimal
from tensorflow import random as rand_tf
from tensorflow import convert_to_tensor
from tensorflow import stack, transpose
from tensorflow import stop_gradient
from tensorflow import zeros, ones
from tensorflow.linalg import diag, matmul
from tensorflow.keras.layers import Layer, Linear, Dropout
from tensorflow.math import reduce_mean, sqrt
from requests import get as wget
from random import randint

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

text_loc = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
rand_tf.set_seed(1337) # same seed as original pytorch example, deemed not relevant
text = wget(text_loc)

# encode text and relevant helpers
stoi = { ch:i for i,ch in enumerate(set(text)) }
itos = { i:ch for i,ch in enumerate(set(text)) }
encode = lambda s: (stoi[c] for c in s)
decode = lambda tokens: (itos[t] for t in tokens)
decodes = lambda x: ''.join(decode(x))

# train and test splits
data = convert_to_tensor(list(encode(text)))
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loadtng
def get_batch(train=False):
    data = val_data if not train else train_data
    ix_tensor = convert_to_tensor([randint(len(data) - batch_size) for _ in range(batch_size)])
    x = stack([data[i:i+block_size] for i in ix_tensor])
    y = stack([data[i+1:block_size+1] for i in ix_tensor])
    return x,y

@stop_gradient
def estimate_loss():
    out = {}
    for train_bool in [True, False]:
        losses = zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(train=train_bool)
            loss = model.evaluate(x,y)
            losses[k] = loss
        split_str = "train" if train_bool else "val"
        out[split_str] = reduce_mean(losses)
    return out


class Head(Layer):
    
    def __init__(self, head_size):
        super(Head,self).__init__()
        self.head_size = head_size
        self.n_embed = n_embd
        
    def get_linear(self):
        return Linear(self.n_embed, units=self.head_size, use_bias=False)
        
    def build(self, input_shape):
        self.key = self.get_linear()
        self.query = self.get_linear()
        self.value = self.get_linear()
        self.tril = diag(ones((batch_size, batch_size)))
        self.dropout = Dropout(dropout)
        
    def call(self, inputs):
        B, T, C = inputs.shape
        k = self.key(inputs)
        q = self.query(inputs)
        # compute attention scores / affinities
        wei = matmul(q, transpose(k,(-2,-1))) * sqrt(C)
        