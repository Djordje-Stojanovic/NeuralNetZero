"""
NeuralNetZero: The most minimal GPT you can build.
~100 parameters. Pure Python. Zero dependencies beyond stdlib.

Based on Karpathy's microgpt.py — same algorithm, shrunk to ~100 params
so you can watch every single parameter learn.

Architecture: 1-layer transformer with:
  - Token embeddings (vocab_size x n_embd)
  - Position embeddings (block_size x n_embd)
  - 1 attention head (Q, K, V, O projections)
  - 1 MLP (fc1: n_embd -> 4*n_embd, fc2: 4*n_embd -> n_embd)
  - Output head (vocab_size x n_embd)

The dataset: simple 4-character patterns like "abab", "abba", "aabb"
so we can see the model learn with a tiny vocab (a, b + BOS = 3 tokens).
"""

import math
import random
random.seed(42)

# ============================================================================
# DATASET — tiny patterns so we can use a 3-token vocab
# ============================================================================
docs = [
    "abab", "baba", "aabb", "bbaa", "abba", "baab",
    "aaab", "bbba", "aaba", "bbab", "abaa", "babb",
    "abbb", "baaa", "aabb", "bbaa", "abab", "baba",
]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ============================================================================
# TOKENIZER — maps characters to integers and back
# ============================================================================
uchars = sorted(set(''.join(docs)))  # ['a', 'b']
BOS = len(uchars)                     # token id 2 = Beginning of Sequence
vocab_size = len(uchars) + 1          # 3 tokens: a=0, b=1, BOS=2
print(f"vocab: {uchars} + [BOS]")
print(f"vocab size: {vocab_size}")

# ============================================================================
# AUTOGRAD ENGINE — the heart of backpropagation
#
# Every Value stores:
#   data       = the actual number (computed in forward pass)
#   grad       = how much the final loss changes if this value changes
#                (computed in backward pass via chain rule)
#   _children  = which Values were used to compute this one
#   _local_grads = the derivative of this Value w.r.t. each child
#
# Example: if c = a * b, then:
#   c.data = a.data * b.data
#   dc/da = b.data  (local gradient w.r.t. a)
#   dc/db = a.data  (local gradient w.r.t. b)
#
# Chain rule: if loss depends on c, and c depends on a:
#   dloss/da = dloss/dc * dc/da = c.grad * b.data
# ============================================================================
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads', 'label')

    def __init__(self, data, children=(), local_grads=(), label=''):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
        self.label = label

    def __add__(self, other):
        # d(a+b)/da = 1, d(a+b)/db = 1
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        # d(a*b)/da = b, d(a*b)/db = a
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # d(a^n)/da = n * a^(n-1)
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        # d(log(a))/da = 1/a
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        # d(exp(a))/da = exp(a)
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        # d(relu(a))/da = 1 if a > 0, else 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        """Backpropagation: compute gradients for ALL Values in the graph.

        1. Topological sort: order nodes so children come before parents
        2. Set this node's gradient to 1 (dloss/dloss = 1)
        3. Walk backwards: for each node, push gradients to its children
           using the chain rule: child.grad += local_grad * parent.grad
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

# ============================================================================
# MODEL HYPERPARAMETERS — tuned to give ~100 parameters
# ============================================================================
n_layer = 1       # 1 transformer layer
n_embd = 2        # embedding dimension of 2 (so we can almost visualize it!)
block_size = 6    # max sequence length (longest doc is 4 chars + 2 BOS)
n_head = 1        # 1 attention head
head_dim = n_embd // n_head  # = 2

# ============================================================================
# PARAMETER INITIALIZATION — small random numbers from Gaussian distribution
#
# Why random? If all params were the same, all neurons would compute the
# same thing and learn the same gradients — they'd never differentiate.
# Small values prevent exploding activations at the start.
# ============================================================================
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),      # token embeddings: 3 x 2 = 6 params
    'wpe': matrix(block_size, n_embd),       # position embeddings: 6 x 2 = 12 params
    'lm_head': matrix(vocab_size, n_embd),   # output projection: 3 x 2 = 6 params
}
# Attention: Q, K, V, O projections — each 2x2 = 4 params, total 16
state_dict['layer0.attn_wq'] = matrix(n_embd, n_embd)
state_dict['layer0.attn_wk'] = matrix(n_embd, n_embd)
state_dict['layer0.attn_wv'] = matrix(n_embd, n_embd)
state_dict['layer0.attn_wo'] = matrix(n_embd, n_embd)
# MLP: fc1 is 2->8 (16 params), fc2 is 8->2 (16 params)
state_dict['layer0.mlp_fc1'] = matrix(4 * n_embd, n_embd)
state_dict['layer0.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# Show what each parameter group contributes
print("\n--- Parameter Breakdown ---")
for name, mat in state_dict.items():
    count = sum(len(row) for row in mat)
    print(f"  {name:25s}: {count:3d} params  (shape {len(mat)}x{len(mat[0])})")
print()

# ============================================================================
# MODEL ARCHITECTURE — the transformer forward pass
# ============================================================================
def linear(x, w):
    """Matrix-vector multiply: y = W @ x. Each row of W dots with x."""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """Convert raw scores to probabilities (0-1, sum to 1).
    Subtract max for numerical stability (doesn't change result)."""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """Root Mean Square normalization — stabilizes training.
    Scales x so its RMS (root mean square) is approximately 1."""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    """Process ONE token at position pos_id through the entire transformer.

    Returns logits: raw scores for what the next token should be.
    Higher logit = model thinks that token is more likely to come next.
    """
    # Step 1: Look up embeddings
    # Token embedding: what does this token "mean"?
    tok_emb = state_dict['wte'][token_id]
    # Position embedding: where is this token in the sequence?
    pos_emb = state_dict['wpe'][pos_id]
    # Combine: the token's meaning + its position
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    # Step 2: Self-Attention — let this token look at all previous tokens
    # and decide which ones are relevant
    x_residual = x
    x = rmsnorm(x)

    # Q = "what am I looking for?"  K = "what do I contain?"  V = "what do I offer?"
    q = linear(x, state_dict['layer0.attn_wq'])
    k = linear(x, state_dict['layer0.attn_wk'])
    v = linear(x, state_dict['layer0.attn_wv'])

    # Store K and V for future tokens to attend to
    keys[0].append(k)
    values[0].append(v)

    # Attention scores: how much does this token care about each previous token?
    attn_logits = [
        sum(q[j] * keys[0][t][j] for j in range(head_dim)) / head_dim**0.5
        for t in range(len(keys[0]))
    ]
    attn_weights = softmax(attn_logits)  # normalize to probabilities

    # Weighted sum of values: blend previous tokens' info based on attention
    x_attn = [
        sum(attn_weights[t] * values[0][t][j] for t in range(len(values[0])))
        for j in range(head_dim)
    ]
    x = linear(x_attn, state_dict['layer0.attn_wo'])
    # Residual connection: add the original input back (prevents vanishing gradients)
    x = [a + b for a, b in zip(x, x_residual)]

    # Step 3: MLP — a small neural network that processes each position independently
    x_residual = x
    x = rmsnorm(x)
    x = linear(x, state_dict['layer0.mlp_fc1'])  # expand: 2 -> 8
    x = [xi.relu() for xi in x]                   # non-linearity: kill negatives
    x = linear(x, state_dict['layer0.mlp_fc2'])   # compress: 8 -> 2
    x = [a + b for a, b in zip(x, x_residual)]    # residual connection

    # Step 4: Project to vocabulary — one score per possible next token
    logits = linear(x, state_dict['lm_head'])
    return logits

# ============================================================================
# OPTIMIZER — Adam, the most popular optimizer
#
# Instead of just subtracting the gradient (SGD), Adam:
# 1. Tracks a moving average of gradients (momentum — like a ball rolling)
# 2. Tracks a moving average of squared gradients (adapts learning rate per param)
# This makes training much more stable and faster to converge.
# ============================================================================
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)  # momentum buffer (first moment)
v = [0.0] * len(params)  # velocity buffer (second moment)

# ============================================================================
# TRAINING LOOP — the actual learning
# ============================================================================
num_steps = 200

print("=" * 60)
print("TRAINING — watch the loss go down!")
print("=" * 60)

for step in range(num_steps):
    # Pick a training document
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward pass: compute predictions and loss
    keys, values = [[]], [[]]  # KV cache for attention
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        # Loss = -log(probability of correct answer)
        # If model assigns prob 1.0 to correct token: loss = -log(1) = 0 (perfect!)
        # If model assigns prob 0.01 to correct token: loss = -log(0.01) = 4.6 (terrible!)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    # Backward pass: compute gradients via chain rule (backpropagation)
    loss.backward()

    # Show detailed info on first few steps and then periodically
    if step < 3 or step % 50 == 0 or step == num_steps - 1:
        print(f"\nstep {step+1:4d}/{num_steps} | loss {loss.data:.4f} | doc: '{doc}'")
        # Show some parameter gradients so you can see learning happening
        print(f"  token_emb[a] = [{state_dict['wte'][0][0].data:.4f}, {state_dict['wte'][0][1].data:.4f}]"
              f"  grads: [{state_dict['wte'][0][0].grad:.4f}, {state_dict['wte'][0][1].grad:.4f}]")
        print(f"  token_emb[b] = [{state_dict['wte'][1][0].data:.4f}, {state_dict['wte'][1][1].data:.4f}]"
              f"  grads: [{state_dict['wte'][1][0].grad:.4f}, {state_dict['wte'][1][1].grad:.4f}]")

        # Show what the model predicts after BOS
        test_keys, test_values = [[]], [[]]
        test_logits = gpt(BOS, 0, test_keys, test_values)
        test_probs = softmax(test_logits)
        prob_strs = [f"'{c}':{p.data:.3f}" for c, p in zip(uchars + ['BOS'], test_probs)]
        print(f"  P(next|BOS) = {{{', '.join(prob_strs)}}}")
    else:
        print(f"step {step+1:4d}/{num_steps} | loss {loss.data:.4f}", end='\r')

    # Adam optimizer update
    lr_t = learning_rate * (1 - step / num_steps)  # linear LR decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0  # reset gradient for next step

# ============================================================================
# INFERENCE — generate new sequences!
# ============================================================================
temperature = 0.5
print("\n" + "=" * 60)
print("INFERENCE — the model generates new sequences")
print("=" * 60)
for sample_idx in range(10):
    keys, values = [[]], [[]]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"  sample {sample_idx+1:2d}: {''.join(sample)}")

# ============================================================================
# FINAL ANALYSIS — show what the model learned
# ============================================================================
print("\n" + "=" * 60)
print("WHAT THE MODEL LEARNED")
print("=" * 60)

print(f"\nToken embeddings (2D — you could literally plot these!):")
for i, ch in enumerate(uchars + ['BOS']):
    emb = state_dict['wte'][i]
    print(f"  '{ch}' -> ({emb[0].data:.4f}, {emb[1].data:.4f})")

print(f"\nPosition embeddings:")
for pos in range(min(5, block_size)):
    emb = state_dict['wpe'][pos]
    print(f"  pos {pos} -> ({emb[0].data:.4f}, {emb[1].data:.4f})")

print(f"\nTransition probabilities (what follows what?):")
for token_name, token_id in [('BOS', BOS), ('a', 0), ('b', 1)]:
    keys, values = [[]], [[]]
    logits = gpt(token_id, 0, keys, values)
    probs = softmax(logits)
    prob_strs = [f"'{c}':{p.data:.3f}" for c, p in zip(uchars + ['BOS'], probs)]
    print(f"  After '{token_name}': {{{', '.join(prob_strs)}}}")
