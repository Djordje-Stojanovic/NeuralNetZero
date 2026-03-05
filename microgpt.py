"""
NeuralNetZero — ~1000 parameter GPT trained on first-principles physics.
Pure Python. Zero dependencies. Every line explained.

=============================================================================
HOW A NEURAL NETWORK WORKS (the 60-second version)
=============================================================================

A neural net is a function: text_in -> probability_of_next_character

It has thousands of adjustable numbers called "parameters" (or "weights").
Training = finding the right values for these parameters so the function
makes good predictions.

THE TRAINING LOOP (repeat 1000x):
  1. FORWARD PASS:  Feed text through the network, get predictions
  2. LOSS:          Measure how wrong the predictions are
  3. BACKWARD PASS: Calculate how to nudge each parameter to reduce the loss
                    (this is "backpropagation" — applying the chain rule of calculus)
  4. UPDATE:        Nudge each parameter a tiny bit in the right direction

That's it. Everything else is details about the architecture (transformer),
the math (attention, matrix multiplies), and the optimization (Adam).

=============================================================================
WHAT ARE "WEIGHTS" / "PARAMETERS"?
=============================================================================

Every parameter is just a floating-point number (like 0.0342 or -0.1587).
Together, they define the function. Think of them as knobs on a mixing board:
- Token embeddings:    "what does each character mean?" (numbers encoding identity)
- Position embeddings: "what does position 0 vs position 5 mean?" (numbers encoding order)
- Attention weights:   "which past characters should I pay attention to?" (numbers encoding relevance)
- MLP weights:         "what patterns should I detect?" (numbers encoding features)
- Output weights:      "given everything, what character comes next?" (numbers encoding predictions)

We initialize them to small random numbers, then the training loop adjusts them.

=============================================================================
ARCHITECTURE: 1-layer transformer with sinusoidal positions
=============================================================================

  Input character
       |
  [Token Embedding]     — look up a vector for this character (LEARNED)
       +
  [Position Encoding]   — add a vector for this position (FIXED, not learned)
       |
  [RMSNorm]             — normalize the scale (no params at this size)
       |
  ┌────┴────┐
  | Attention|           — let this character look at all previous characters
  |  Q, K, V |           — Q="what am I looking for?", K="what do I have?", V="what do I offer?"
  |  softmax  |          — convert similarity scores to probabilities
  └────┬────┘
       + (residual)      — add back the input (skip connection)
       |
  ┌────┴────┐
  |   MLP    |           — 2-layer neural network: expand -> ReLU -> compress
  |  fc1,fc2 |           — detects patterns in the attention output
  └────┬────┘
       + (residual)      — add back the input again
       |
  [Output Head]          — project to vocabulary size: one score per character
       |
  [Softmax]              — convert scores to probabilities
       |
  Probability of each next character

=============================================================================
WHY PYTHON AND NOT C/CUDA?
=============================================================================

At 1000 parameters, the entire model is 4 KB. Your RTX 5070 Super would
finish each training step in nanoseconds — you couldn't even see it happen.
Pure Python lets us inspect EVERY value, EVERY gradient, in real time.

The stack for scaling up:
  100-10K params:   Pure Python (this file) — for understanding
  10K-1M params:    PyTorch (Python API calling CUDA) — for speed
  1M-1B params:     PyTorch + FP16/BF16 on GPU — memory matters
  1B+ params:       PyTorch + FP8/MXFP4 + multi-GPU — every bit counts

PyTorch IS the industry standard for GPU training. The Python is just the
interface — the actual matrix math runs as CUDA C++ kernels on your GPU.
FP8 and MXFP4 are precision formats that use fewer bits per number to save
memory. At 1000 params we don't need them. At 1 billion params they let
you fit the model in GPU VRAM.

=============================================================================
"""

import math
import random
import time
import sys
import os
random.seed(42)

# Fix Windows terminal encoding for Unicode characters
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ============================================================================
# PART 1: THE DATASET — First-Principles Physics
# ============================================================================
# We train on real physics text. The model will learn character-level patterns:
# common letter combinations, word structures, spacing patterns.
# At 1000 params it won't "understand" physics, but as we scale up,
# you'll see it progress from gibberish -> words -> phrases -> concepts.
#
# Examples range from 3 characters ("f=ma") to ~256 characters (full paragraphs).
# This teaches the model about varying sequence lengths.
# ============================================================================

docs = [
    # --- Tiny (1-8 chars) — formulas and constants ---
    "f=ma",
    "e=mc2",
    "pv=nrt",
    "v=ir",
    "f=kx",
    "s=vt",
    "p=mv",
    "w=fd",

    # --- Short (16-32 chars) — law statements ---
    "force equals mass times acceleration",
    "energy cannot be created or destroyed",
    "every action has an equal reaction",
    "entropy of a system always increases",
    "light speed is constant for all observers",
    "an object at rest stays at rest",
    "pressure times volume equals nrt",
    "energy equals mass times c squared",

    # --- Medium (32-64 chars) — principles explained ---
    "gravity pulls all objects toward each other with a force proportional to their masses",
    "the total momentum of a closed system is always conserved in all collisions",
    "heat flows from hot objects to cold objects and never the reverse direction",
    "electric current flows through a conductor proportional to the voltage applied",
    "a wave carries energy through space without transporting matter along with it",
    "the position and momentum of a particle cannot both be known with precision",
    "time passes slower for objects moving at speeds close to the speed of light",
    "the total energy of an isolated system remains constant through all changes",

    # --- Long (64-128 chars) — deeper explanations ---
    "newtons first law says that an object in motion will stay in motion at constant velocity unless acted on by an external net force",
    "the second law of thermodynamics states that the entropy of an isolated system will tend to increase over time toward a maximum",
    "maxwells equations describe how electric and magnetic fields are generated and altered by each other and by charges and currents",
    "the photoelectric effect shows that light behaves as particles called photons each carrying energy proportional to its frequency",
    "general relativity describes gravity not as a force but as the curvature of spacetime caused by the presence of mass and energy",
    "quantum mechanics reveals that at the smallest scales particles exist in superposition of multiple states until they are measured",
    "conservation of angular momentum means a spinning object will keep spinning at the same rate unless a torque acts upon it from outside",
    "the strong nuclear force binds protons and neutrons together inside atomic nuclei overcoming the electric repulsion between protons",

    # --- Very long (128-256 chars) — full explanations ---
    "the heisenberg uncertainty principle states that there is a fundamental limit to the precision with which certain pairs of physical properties such as position and momentum can be simultaneously known and this is not due to measurement error but is a property of nature itself",
    "einsteins special theory of relativity rests on two postulates first that the laws of physics are the same in all inertial reference frames and second that the speed of light in a vacuum is the same for all observers regardless of their motion or the motion of the source",
    "the principle of least action states that the path taken by a physical system between two states is the one for which the action integral is stationary and this single principle can be used to derive all of classical mechanics electromagnetism and even quantum field theory",
    "thermodynamic equilibrium is reached when a system has maximized its entropy and no further spontaneous changes can occur the system then has uniform temperature pressure and chemical potential throughout and all net flows of energy and matter have ceased completely",
    "the wave function in quantum mechanics contains all information about a quantum system and its square gives the probability of finding a particle at a given location but the wave function itself is not directly observable only its statistical predictions can be tested",
    "electromagnetic waves are oscillating electric and magnetic fields that propagate through space at the speed of light and they require no medium to travel through unlike sound waves they can cross the vacuum of space carrying energy from stars to planets across vast distances",
    "nuclear fission occurs when a heavy atomic nucleus splits into two lighter nuclei releasing enormous energy because the binding energy per nucleon is higher for the products than for the original heavy nucleus and this energy difference is what powers nuclear reactors and atomic bombs",
    "the doppler effect describes how the observed frequency of a wave changes when the source and observer are moving relative to each other approaching objects have higher frequency and receding objects have lower frequency and this applies to both sound waves and light waves in space",
    "in quantum field theory particles are excitations of underlying quantum fields that permeate all of space and every type of particle has its own field electrons arise from the electron field photons from the electromagnetic field and the interactions between fields give rise to forces",
    "the equivalence principle states that the effects of gravity are locally indistinguishable from the effects of acceleration in a small enough region of spacetime a person in a closed elevator cannot tell whether they are standing on earth or being accelerated through space at one g",
]

random.shuffle(docs)
print(f"num docs: {len(docs)}")
print(f"doc lengths: {sorted(len(d) for d in docs)} chars")
print(f"shortest: {min(len(d) for d in docs)} chars, longest: {max(len(d) for d in docs)} chars")


# ============================================================================
# PART 2: THE TOKENIZER — Character-Level
# ============================================================================
# At 1000 params, character-level is the right choice. Here's why:
#
# GPT-4/5 use BPE (Byte Pair Encoding) with ~100K vocabulary.
# BPE merges frequent character pairs into single tokens: "th" -> [th], "the" -> [the]
# This is more efficient but requires a HUGE embedding table (100K x embedding_dim).
# With 1000 total params and 100K vocab, you'd have 0.01 params per token — impossible.
#
# Character-level IS the foundation that BPE is built on. We start here.
# As we scale to millions of params, we'll add BPE on top.
#
# Our vocab: lowercase a-z, digits 0-9, space, =, plus BOS/EOS
# ============================================================================

uchars = sorted(set(''.join(docs)))  # all unique characters in our physics text
BOS = len(uchars)                     # special Beginning/End of Sequence token
vocab_size = len(uchars) + 1
print(f"\nvocab ({vocab_size} tokens): {uchars} + [BOS]")
print(f"  this is a CHARACTER-LEVEL tokenizer")
print(f"  'force' = {[uchars.index(c) for c in 'force']} (5 tokens)")
print(f"  GPT-4 would encode 'force' as 1 token from a 100K vocab")
print(f"  we can't afford 100K vocab at 1000 params — character-level is correct here")


# ============================================================================
# PART 3: AUTOGRAD — How Backpropagation Actually Works
# ============================================================================
# The neural network is a chain of math operations: add, multiply, exp, log...
# Each operation knows its own derivative (from calculus).
#
# FORWARD PASS: compute the result, building a graph of operations
# BACKWARD PASS: walk the graph backwards, multiplying derivatives (chain rule)
#
# THE CHAIN RULE (the single most important idea in deep learning):
#   If loss = f(g(h(x))), then:
#   dloss/dx = dloss/df * df/dg * dg/dh * dh/dx
#
#   In English: "how much does the loss change if I nudge x?"
#   = "how much does f change per unit g" * "how much does g change per unit h" * ...
#
# Each Value node stores:
#   .data         — the number (computed going forward)
#   .grad         — dloss/d(this_value) (computed going backward)
#   ._children    — the inputs that produced this value
#   ._local_grads — the derivative of this value w.r.t. each input
# ============================================================================

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data        # the actual number
        self.grad = 0           # dloss/d(self) — filled in during backward()
        self._children = children
        self._local_grads = local_grads

    # ADDITION: c = a + b
    # dc/da = 1 (if a goes up by 1, c goes up by 1)
    # dc/db = 1
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    # MULTIPLICATION: c = a * b
    # dc/da = b (if a goes up by 1, c goes up by b)
    # dc/db = a
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    # POWER: c = a^n
    # dc/da = n * a^(n-1)  (power rule from calculus)
    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    # LOG: c = log(a)
    # dc/da = 1/a
    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))

    # EXP: c = e^a
    # dc/da = e^a (exp is its own derivative!)
    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    # RELU: c = max(0, a)
    # dc/da = 1 if a > 0, else 0 (gradient flows through if positive, blocked if negative)
    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        """THE BACKWARD PASS — compute dloss/d(every_value) via chain rule.

        Step 1: Topological sort — order all operations so inputs come before outputs
        Step 2: Set dloss/dloss = 1 (the loss's gradient w.r.t. itself is 1)
        Step 3: Walk backward through every operation:
                For each child:  child.grad += local_grad * parent.grad
                This IS the chain rule: dloss/dchild = dloss/dparent * dparent/dchild
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


# ============================================================================
# PART 4: POSITION ENCODING — Sinusoidal (Fixed, No Learned Parameters)
# ============================================================================
# The transformer processes tokens in parallel, so it needs to know WHERE
# each token is in the sequence. Two approaches:
#
# LEARNED positions (Karpathy's microgpt): A lookup table. Each position gets
#   its own learned vector. Cost: block_size * n_embd parameters.
#   At block_size=256, n_embd=6: that's 1536 params — more than our entire budget!
#
# SINUSOIDAL positions (original Transformer paper, "Attention Is All You Need"):
#   Use sin/cos waves at different frequencies. FREE — zero parameters.
#   Position 0 gets one pattern, position 1 gets a slightly different pattern, etc.
#   The model can learn to use these fixed patterns through its other weights.
#
# Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d))
#          PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
# Each dimension oscillates at a different frequency, creating a unique "fingerprint"
# for each position.
# ============================================================================

def make_sinusoidal_positions(max_len, d_model):
    """Create fixed position encodings — no learnable parameters!"""
    pe = []
    for pos in range(max_len):
        row = []
        for i in range(d_model):
            angle = pos / (10000 ** (2 * (i // 2) / d_model))
            if i % 2 == 0:
                row.append(math.sin(angle))
            else:
                row.append(math.cos(angle))
        pe.append(row)
    return pe


# ============================================================================
# PART 5: MODEL CONFIGURATION — Targeting ~1000 Parameters
# ============================================================================
# Parameter budget breakdown:
#
#   Token embeddings (wte):   vocab_size * n_embd  (what each character "means")
#   Output head (lm_head):    vocab_size * n_embd  (predict next character)
#   Attention Q,K,V,O:        4 * n_embd * n_embd  (how tokens attend to each other)
#   MLP fc1:                  n_embd * 4*n_embd    (pattern detection, expand)
#   MLP fc2:                  4*n_embd * n_embd    (compress back)
#   Position encoding:        0 (sinusoidal = free!)
#
#   Total = 2 * V * E + 4*E^2 + 8*E^2 = 2*V*E + 12*E^2
#
# With vocab V and embedding dim E, solving 2*V*E + 12*E^2 ≈ 1000:
# ============================================================================

n_layer = 1
n_embd = 6          # each character becomes a 6-dimensional vector
block_size = 64      # max context window per training step (truncate longer docs)
                     # architecture SUPPORTS 256 via sinusoidal positions,
                     # but pure Python autograd is O(n^2) per token — 256 chars would
                     # create millions of Value nodes and take hours per step.
                     # When we move to PyTorch+GPU, we'll use the full 256.
n_head = 1           # 1 attention head (with n_embd=6, multi-head doesn't help)
head_dim = n_embd

# Position encodings — fixed sin/cos, supports up to 256 positions
POS_ENC = make_sinusoidal_positions(block_size, n_embd)


# ============================================================================
# PART 6: PARAMETER INITIALIZATION
# ============================================================================
# Every weight starts as a small random number drawn from a Gaussian (bell curve).
#
# Why random? If all weights were identical, every neuron would compute the same
# thing, get the same gradient, and update identically. They'd never specialize.
# Random initialization "breaks symmetry" so different neurons learn different things.
#
# Why SMALL? Large initial weights cause large activations, which cause large
# gradients, which cause huge parameter updates, which cause instability.
# std=0.08 keeps things in a safe range at the start.
# ============================================================================

matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)]
    for _ in range(nout)
]

state_dict = {
    # Token embeddings: each character gets a 6D vector
    # 'a' might be [0.03, -0.1, 0.05, 0.02, -0.08, 0.01]
    # 'e' might be [-0.02, 0.04, 0.09, -0.06, 0.03, -0.01]
    # After training, similar characters (vowels, consonants) cluster together
    'wte': matrix(vocab_size, n_embd),

    # Output head: maps the 6D hidden state back to vocab_size scores
    # High score for a character = model predicts that character is likely next
    'lm_head': matrix(vocab_size, n_embd),
}

# Attention weights: how tokens communicate with each other
# Q = "what am I looking for?" (query)
# K = "what do I contain?" (key)
# V = "what information do I offer?" (value)
# O = "how do I combine the attended information?" (output)
#
# Attention score = Q . K / sqrt(d)
# If Q of "=" and K of "f" have high dot product, "=" pays attention to "f"
# This is how the model might learn "after = comes m" in "f=ma"
state_dict['layer0.attn_wq'] = matrix(n_embd, n_embd)
state_dict['layer0.attn_wk'] = matrix(n_embd, n_embd)
state_dict['layer0.attn_wv'] = matrix(n_embd, n_embd)
state_dict['layer0.attn_wo'] = matrix(n_embd, n_embd)

# MLP (Multi-Layer Perceptron): a small 2-layer neural net inside the transformer
# fc1: expand from 6 -> 24 dimensions (detect 24 different features)
# ReLU: set negative values to 0 (non-linearity — without this, the network
#        would just be one big linear function and couldn't learn complex patterns)
# fc2: compress from 24 -> 6 dimensions (combine features back)
state_dict['layer0.mlp_fc1'] = matrix(4 * n_embd, n_embd)
state_dict['layer0.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"\n{'='*60}")
print(f"MODEL: {len(params)} parameters")
print(f"{'='*60}")
print(f"  (for context: GPT-2 = 1,500,000,000 parameters)")
print(f"  (for context: GPT-4 = ~1,800,000,000,000 parameters)")
print(f"  (we have {len(params)}. this is a baby learning the alphabet.)")
print()
for name, mat in state_dict.items():
    count = sum(len(row) for row in mat)
    shape = f"{len(mat)}x{len(mat[0])}"
    purpose = {
        'wte': 'token embeddings — what each character means',
        'lm_head': 'output head — predict next character',
        'layer0.attn_wq': 'attention Q — "what am I looking for?"',
        'layer0.attn_wk': 'attention K — "what do I contain?"',
        'layer0.attn_wv': 'attention V — "what info do I offer?"',
        'layer0.attn_wo': 'attention O — "combine attended info"',
        'layer0.mlp_fc1': 'MLP expand — detect 24 features',
        'layer0.mlp_fc2': 'MLP compress — combine features back',
    }.get(name, '')
    print(f"  {name:25s} {shape:>8s} = {count:4d} params  | {purpose}")
print(f"  {'TOTAL':25s} {'':>8s} = {len(params):4d} params")
print(f"\n  Position encoding: sinusoidal (FIXED — 0 learned params)")
print(f"  Max context length: {block_size} characters")
print(f"  Embedding dimension: {n_embd}")
print(f"  Attention heads: {n_head}")
print(f"  Transformer layers: {n_layer}")


# ============================================================================
# PART 7: THE TRANSFORMER — Forward Pass
# ============================================================================

def linear(x, w):
    """Matrix-vector multiply: y[i] = sum_j(W[i][j] * x[j])

    This is the fundamental operation. Almost all neural net computation
    is matrix multiplies. On a GPU, this runs on thousands of cores in parallel.
    On your RTX 5070 Super, a single matmul can process billions of numbers/second.
    Here in pure Python, we do it with nested loops — educational but slow.
    """
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """Convert raw scores (logits) to probabilities.

    softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Properties:
    - All outputs are between 0 and 1
    - All outputs sum to 1
    - Larger inputs get larger probabilities (exponentially)

    The max subtraction is for numerical stability: exp(1000) would overflow,
    but exp(1000-1000) = exp(0) = 1 is fine. Doesn't change the result.
    """
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """Root Mean Square Layer Normalization.

    Divides each element by the RMS of the vector.
    This keeps activations at a stable scale throughout the network.
    Without normalization, values can grow or shrink exponentially
    through many layers, causing training to diverge.
    """
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    """Process ONE token through the transformer.

    This function is called once per character in the sequence.
    It maintains a KV-cache so each new token can attend to all previous tokens.

    Returns: logits — one score per character in the vocabulary.
    The highest score = the model's best guess for the next character.
    """
    # --- EMBEDDING ---
    # Look up the token's embedding vector (6 numbers that represent this character)
    tok_emb = state_dict['wte'][token_id]
    # Add the position encoding (tells the model WHERE this character is)
    # These are fixed sin/cos values — not learned
    pos_enc = POS_ENC[pos_id]
    x = [t + Value(p) for t, p in zip(tok_emb, pos_enc)]
    x = rmsnorm(x)

    # --- SELF-ATTENTION ---
    # This is THE key innovation of the Transformer.
    # For each token, attention lets it "look at" all previous tokens
    # and decide which ones are relevant for predicting what comes next.
    #
    # Example: in "f=ma", when processing 'a', attention might learn to
    # look back at 'm' (because 'a' often follows 'm' in physics).
    x_residual = x
    x = rmsnorm(x)

    q = linear(x, state_dict['layer0.attn_wq'])  # query: "what am I looking for?"
    k = linear(x, state_dict['layer0.attn_wk'])  # key: "what do I contain?"
    v = linear(x, state_dict['layer0.attn_wv'])  # value: "what info do I provide?"

    keys[0].append(k)
    values[0].append(v)

    # Compute attention scores: dot product of query with all keys
    # High score = "I should pay attention to that token"
    attn_logits = [
        sum(q[j] * keys[0][t][j] for j in range(head_dim)) / head_dim**0.5
        for t in range(len(keys[0]))
    ]
    attn_weights = softmax(attn_logits)

    # Weighted sum: blend all value vectors based on attention weights
    x_attn = [
        sum(attn_weights[t] * values[0][t][j] for t in range(len(values[0])))
        for j in range(head_dim)
    ]
    x = linear(x_attn, state_dict['layer0.attn_wo'])

    # Residual connection: x = attention_output + original_input
    # This is crucial! Without it, gradients would have to flow through
    # every layer during backprop, getting smaller and smaller (vanishing gradient).
    # The skip connection gives gradients a "highway" straight to earlier layers.
    x = [a + b for a, b in zip(x, x_residual)]

    # --- MLP (Feed-Forward Network) ---
    # A 2-layer neural network that processes each position independently.
    # Attention mixes information BETWEEN positions.
    # MLP processes information WITHIN each position.
    x_residual = x
    x = rmsnorm(x)
    x = linear(x, state_dict['layer0.mlp_fc1'])   # expand: 6 -> 24
    x = [xi.relu() for xi in x]                    # ReLU: kill negatives
    x = linear(x, state_dict['layer0.mlp_fc2'])    # compress: 24 -> 6
    x = [a + b for a, b in zip(x, x_residual)]     # residual connection

    # --- OUTPUT ---
    # Project from 6D hidden state to vocab_size scores
    logits = linear(x, state_dict['lm_head'])
    return logits


# ============================================================================
# PART 8: THE OPTIMIZER — Adam
# ============================================================================
# SGD (Stochastic Gradient Descent) is the simplest optimizer:
#   param -= learning_rate * gradient
#
# Problem: all parameters use the same learning rate.
# Parameters with large gradients overshoot. Parameters with small gradients crawl.
#
# ADAM (Adaptive Moment Estimation) fixes this:
#   m = moving average of gradient        (momentum — remembers direction)
#   v = moving average of gradient^2      (adapts learning rate per parameter)
#   param -= lr * m / sqrt(v)
#
# Params with large gradients get smaller effective lr (stabilized by sqrt(v))
# Params with small gradients get larger effective lr (momentum in m accumulates)
#
# beta1=0.85: how much momentum to keep (0=none, 1=full memory)
# beta2=0.99: how quickly the adaptive lr adjusts (closer to 1 = more stable)
# ============================================================================

learning_rate = 0.01
beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)
v_buf = [0.0] * len(params)


# ============================================================================
# PART 9: TRAINING LOOP with CLI Visualization
# ============================================================================

num_steps = 300  # pure Python autograd is slow; 300 steps is enough to see learning
loss_history = []

# Use ASCII-safe progress bars (Windows cp1252 can't handle Unicode blocks)
def loss_bar(loss_val, width=30):
    """ASCII bar chart for loss value."""
    filled = min(width, max(0, int(loss_val / 4.0 * width)))
    return f"[{'#' * filled}{'.' * (width - filled)}]"

def generate_sample(max_len=50, temperature=0.5):
    """Generate a text sample from the model."""
    keys, values = [[]], [[]]
    token_id = BOS
    sample = []
    for pos_id in range(min(max_len, block_size)):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    return ''.join(sample)

print(f"\n{'='*60}")
print(f"TRAINING -- {num_steps} steps on {len(docs)} physics documents")
print(f"{'='*60}")
print(f"  The loss measures how bad the predictions are.")
print(f"  Random guessing among {vocab_size} tokens: loss = {math.log(vocab_size):.2f}")
print(f"  Perfect predictions: loss = 0.00")
print(f"  Watch the bar shrink as the model learns!")
print()

t_start = time.time()

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # --- FORWARD PASS: build computation graph, compute loss ---
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    # --- BACKWARD PASS: compute all gradients via chain rule ---
    loss.backward()

    loss_history.append(loss.data)

    # --- DISPLAY ---
    elapsed = time.time() - t_start
    if step < 5 or step % 50 == 0 or step == num_steps - 1:
        bar = loss_bar(loss.data)
        print(f"  step {step+1:4d}/{num_steps} | loss {loss.data:.4f} {bar} | {elapsed:5.1f}s | {len(doc):3d} chars")
        if step < 5:
            max_grad = max(abs(p.grad) for p in params)
            avg_grad = sum(abs(p.grad) for p in params) / len(params)
            print(f"       gradients: max={max_grad:.4f} avg={avg_grad:.4f} "
                  f"(these tell each param which direction to move)")
        if step % 100 == 0 and step > 0:
            sample = generate_sample(max_len=40)
            print(f"       sample: \"{sample}\"")
    else:
        if step % 10 == 0:
            bar = loss_bar(loss.data)
            sys.stdout.write(f"\r  step {step+1:4d}/{num_steps} | loss {loss.data:.4f} {bar} | {elapsed:5.1f}s")
            sys.stdout.flush()

    # --- ADAM OPTIMIZER UPDATE ---
    lr_t = learning_rate * (1 - step / num_steps)  # linear LR decay
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
        v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

total_time = time.time() - t_start
print(f"\n\n  Training complete in {total_time:.1f}s")
print(f"  Final loss: {loss_history[-1]:.4f} (started at {loss_history[0]:.4f})")
print(f"  Improvement: {loss_history[0] - loss_history[-1]:.4f}")


# ============================================================================
# PART 10: LOSS CURVE — ASCII Art
# ============================================================================

print(f"\n{'='*60}")
print(f"LOSS CURVE")
print(f"{'='*60}")
height = 15
width = 60
# Sample loss_history to fit width
if len(loss_history) > width:
    sampled = [loss_history[int(i * len(loss_history) / width)] for i in range(width)]
else:
    sampled = loss_history

max_loss = max(sampled)
min_loss = min(sampled)
loss_range = max_loss - min_loss if max_loss != min_loss else 1

for row in range(height):
    threshold = max_loss - (row / (height - 1)) * loss_range
    line = ""
    for val in sampled:
        if val >= threshold:
            line += "#"
        else:
            line += " "
    label = f"{threshold:.2f}" if row % 3 == 0 else "     "
    print(f"  {label:>5s} |{line}|")
print(f"        +{'-' * width}+")
print(f"         step 1{' ' * (width - 10)}step {num_steps}")


# ============================================================================
# PART 11: INFERENCE — Generate Physics Text
# ============================================================================

print(f"\n{'='*60}")
print(f"INFERENCE — Model generates text")
print(f"{'='*60}")
print(f"  With ~1000 params, expect character patterns, not physics knowledge.")
print(f"  At 1M+ params you'd start seeing real words form.")
print()

for temp_name, temp in [("low (0.3)", 0.3), ("medium (0.7)", 0.7), ("high (1.2)", 1.2)]:
    print(f"  Temperature {temp_name}:")
    for i in range(5):
        sample = generate_sample(max_len=60, temperature=temp)
        print(f"    {i+1}. \"{sample}\"")
    print()


# ============================================================================
# PART 12: WHAT THE MODEL LEARNED
# ============================================================================

print(f"{'='*60}")
print(f"ANALYSIS — What did {len(params)} parameters learn?")
print(f"{'='*60}")

# Character frequency analysis
print(f"\n  Character predictions after [BOS] (start of sequence):")
test_keys, test_values = [[]], [[]]
test_logits = gpt(BOS, 0, test_keys, test_values)
test_probs = softmax(test_logits)
# Sort by probability
char_probs = [(uchars[i] if i < len(uchars) else 'BOS', test_probs[i].data)
              for i in range(vocab_size)]
char_probs.sort(key=lambda x: -x[1])
for ch, p in char_probs[:10]:
    bar_len = int(p * 40)
    ch_display = repr(ch) if ch != ' ' else "' '"
    print(f"    {ch_display:>5s}: {p:.3f} {'#' * bar_len}")

# Show what the model predicts after common physics characters
print(f"\n  What follows common characters:")
for test_char in ['f', 'e', '=', ' ', 't']:
    if test_char in uchars:
        tid = uchars.index(test_char)
        test_keys, test_values = [[]], [[]]
        logits = gpt(tid, 1, test_keys, test_values)
        probs = softmax(logits)
        top3 = sorted(range(vocab_size), key=lambda i: -probs[i].data)[:3]
        preds = ", ".join(
            f"'{uchars[i] if i < len(uchars) else 'BOS'}'({probs[i].data:.2f})"
            for i in top3
        )
        ch_display = repr(test_char) if test_char != ' ' else "' '"
        print(f"    After {ch_display}: {preds}")

# Token embedding distances
print(f"\n  Token embedding distances (similar characters cluster together):")
vowels = [c for c in 'aeiou' if c in uchars]
consonants = [c for c in 'bcdfghjklmnpqrstvwxyz' if c in uchars]
if len(vowels) >= 2 and len(consonants) >= 2:
    def emb_dist(c1, c2):
        e1 = state_dict['wte'][uchars.index(c1)]
        e2 = state_dict['wte'][uchars.index(c2)]
        return math.sqrt(sum((a.data - b.data)**2 for a, b in zip(e1, e2)))

    print(f"    vowel-vowel (e vs a):     {emb_dist('e', 'a'):.4f}")
    print(f"    consonant-consonant (t vs s): {emb_dist('t', 's'):.4f}")
    print(f"    vowel-consonant (e vs t): {emb_dist('e', 't'):.4f}")


# ============================================================================
# PART 13: SCALING ROADMAP
# ============================================================================

print(f"\n{'='*60}")
print(f"SCALING ROADMAP — What Comes Next")
print(f"{'='*60}")
print(f"""
  CURRENT: {len(params)} params | Pure Python | Character-level | CPU
    - Learns character frequencies and simple patterns
    - Training: {total_time:.0f} seconds

  NEXT STEPS:
  +---------+----------+------------+-----------+---------------------+
  | Params  | Engine   | Tokenizer  | Hardware  | What it learns      |
  +---------+----------+------------+-----------+---------------------+
  | 10K     | Python   | Character  | CPU       | Common words        |
  | 100K    | PyTorch  | Character  | GPU FP32  | Sentence patterns   |
  | 1M      | PyTorch  | BPE        | GPU BF16  | Grammar, structure  |
  | 10M     | PyTorch  | BPE        | GPU FP8   | Facts, reasoning    |
  | 100M+   | PyTorch  | BPE 32K    | GPU FP8   | Knowledge, fluency  |
  +---------+----------+------------+-----------+---------------------+

  Your RTX 5070 Super: 16GB VRAM, Blackwell architecture
    - FP32: ~50 TFLOPS
    - FP16/BF16: ~100 TFLOPS (2x speedup, half memory)
    - FP8: ~200 TFLOPS (4x speedup, quarter memory)
    - At 10M params with FP8, training would take seconds on your GPU
""")
