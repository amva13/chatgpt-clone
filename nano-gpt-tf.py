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
from tensorflow import where
from tensorflow import concat
from tensorflow import reshape
from tensorflow.linalg import diag, matmul
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Layer, Dropout, ReLU, LayerNormalization, Embedding, concatenate, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.math import reduce_mean, sqrt
from tensorflow.nn import softmax
from tensorflow_lattice.layers import Linear
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

# important constants
vocab_size = max(itos.keys()) + 1

# train and test splits
data = convert_to_tensor(list(encode(text)))
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loadtng
def get_batch(train=False):
    data = val_data if not train else train_data
    ix_tensor = convert_to_tensor([randint(0, len(data) - block_size) for _ in range(batch_size)])
    x = stack([data[i:i+block_size] for i in ix_tensor])
    y = stack([data[i+1:i+block_size+1] for i in ix_tensor])
    return x,y

class Head(Layer):
    """One self-attention unit

    Args:
        Layer (head_size): number of outputs / units
    """
    
    def __init__(self, head_size):
        super(Head,self).__init__()
        self.head_size = head_size
        self.n_embed = n_embd
        
    def get_linear(self):
        return Dense(self.n_embed)
        
    def build(self, input_shape):
        self.key = self.get_linear()
        self.query = self.get_linear()
        self.value = self.get_linear()
        self.tril = diag(ones((batch_size, batch_size)))
        self.dropout = Dropout(dropout)
        self.built = True
        
    def call(self, inputs):
        # TODO: dimensionality error found here
        B, T, C = inputs.shape
        k = self.key(inputs)
        q = self.query(inputs)
        # compute attention scores / affinities
        wei = q @ transpose(k, perm=[0,2,1]) * C**0.5
        wei = softmax(wei, -1)
        wei = self.dropout(wei)
        aggregator = self.value(inputs)
        out = matmul(wei, aggregator)
        return out
    
    
class MultiHeadAttention(Layer):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = Dense(n_embd)
        
    def build(self, inpit_shape):
        self.dropout = Dropout(dropout)
        self.built = True
        
    def call(self, inputs):
        out = concatenate([h(inputs) for h in self.heads])
        out = self.dropout(self.proj(out))
        return out
        
class FeedForward(Layer):
    
    def __init__(self, num_embed):
        super().__init__()
        self.n_embd = num_embed
    
    def build(self, input_shape):
        self.net = Sequential([
            Dense(4 * self.n_embd),
            ReLU(),
            Dense(self.n_embd),
            Dropout(dropout)
        ])
    
    def call(self, inputs):
        return self.net(inputs)
    
class Block(Layer):
    """Transformer block

    Args:
        Layer (n_embd, n_heads): n_embd: embedding dimension, n_head: the number of heads we'd like
    """
    
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.self_attention = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        
    def build(self, input_size):
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.built = True
        
    def call(self, inputs):
        inputs += self.self_attention(self.ln1(inputs)) + self.ffwd(self.ln2(inputs))
        return inputs
    

class GPTLanguageModel(Layer):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = Embedding(vocab_size, n_embd)
        self.position_embedding_table = Embedding(vocab_size, n_embd)
        self.blocks = Sequential([Block(n_embd, n_heads=n_head) for _ in range(n_layer)])
        self.ln_f = LayerNormalization()
        self.lm_head = Dense(vocab_size)
        
    def build(self, input_size):
        self.built = True
        
    def call(self, inputs):
        B, T = inputs.shape
        token_embed = self.token_embedding_table(inputs)
        pos_emb = self.position_embedding_table(convert_to_tensor(range(T)))
        x = token_embed + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

        
model = Sequential([GPTLanguageModel()])
model.compile(
    Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"]
)
x_train, y_train = get_batch(train=True)
x_test, y_test = get_batch(train=False)
epochs = 20 # start small

# checkpoint path
checkpoint_path = "tmp/checkpoint"

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1)


model.fit(x=x_train, y=y_train, epochs=epochs, validation_data=(x_test, y_test), callbacks=[cp_callback], batch_size=batch_size)
