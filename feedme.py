#!/usr/bin/env python3
"""
Autoregressive character-level model for Z80.

Instead of classifying into response categories, this model generates
responses character-by-character:

1. Input: query_trigrams[128] + context[128] = 256 dimensions
2. Output: next_char probabilities[64]
3. Loop: run inference, emit char, update context, repeat

The context encodes the last few output characters using the same
trigram hashing approach as the query.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import List, Tuple
from collections import Counter

from qat import OverflowAwareLinear


# Character set - built dynamically from training data
# EOS is always last character
EOS_CHAR = '\x00'

DEFAULT_TRAINING_SETS = [
    'training_conversation.jsonl',  # Generated from .lines files
    # 'training_conversational.jsonl',  # ELIZA-style responses
    # 'training_data.jsonl',  # CP/M commands
]

def build_charset_from_data(filenames: List[str]|None = None) -> str:
    """Scan training data and build minimal charset from responses."""
    if filenames is None:
        filenames = DEFAULT_TRAINING_SETS

    chars = set()
    for filename in filenames:
        try:
            with open(filename) as f:
                for line in f:
                    obj = json.loads(line)
                    response = obj['response'].upper()  # Normalize to uppercase
                    chars.update(response)
        except FileNotFoundError:
            pass

    # Sort for consistency: space first, then A-Z, then 0-9, then punctuation
    chars.discard(EOS_CHAR)  # Remove if present, we add it last

    letters = sorted(c for c in chars if c.isalpha())
    digits = sorted(c for c in chars if c.isdigit())
    space = [' '] if ' ' in chars else []
    punct = sorted(c for c in chars if not c.isalnum() and c != ' ')

    charset = ''.join(space + letters + digits + punct) + EOS_CHAR
    return charset


# Build charset on module load (will be rebuilt in train function with actual files)
CHARSET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.:=*?-/'" + EOS_CHAR  # Fallback
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARSET)}
EOS_IDX = len(CHARSET) - 1
NUM_CHARS = len(CHARSET)


def char_to_idx(c: str) -> int:
    """Convert character to index, defaulting to space for unknown."""
    c_upper = c.upper()
    if c_upper in CHAR_TO_IDX:
        return CHAR_TO_IDX[c_upper]
    elif c in CHAR_TO_IDX:
        return CHAR_TO_IDX[c]
    else:
        return 0  # space for unknown


def idx_to_char(i: int) -> str:
    """Convert index to character."""
    return IDX_TO_CHAR.get(i, ' ')


class TrigramEncoder:
    """Encode text into trigram hash buckets (integer-friendly, no normalization)."""

    def __init__(self, num_buckets: int = 128):
        self.num_buckets = num_buckets

    def _hash_trigram(self, trigram: str) -> int:
        """Hash a trigram to a bucket index."""
        h = 0
        for c in trigram:
            h = (h * 31 + ord(c)) & 0xFFFF
        return h % self.num_buckets

    def encode(self, text: str) -> np.ndarray:
        """Encode text into bucket counts (raw counts, Z80-compatible)."""
        vec = np.zeros(self.num_buckets, dtype=np.float32)
        text = text.lower()
        text = ' ' + text + ' '  # Pad for boundary trigrams

        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            bucket = self._hash_trigram(trigram)
            vec[bucket] += 1.0

        # No normalization - use raw counts for Z80 compatibility
        return vec


class ContextEncoder:
    """Encode recent output characters into hash buckets (integer-friendly)."""

    def __init__(self, num_buckets: int = 128, context_len: int = 8):
        self.num_buckets = num_buckets
        self.context_len = context_len

    def _hash_ngram(self, ngram: str, offset: int = 0) -> int:
        """Hash an n-gram with position offset."""
        h = offset * 7
        for c in ngram:
            h = (h * 31 + ord(c)) & 0xFFFF
        return h % self.num_buckets

    def encode(self, recent_chars: str) -> np.ndarray:
        """Encode recent output characters (raw counts, Z80-compatible)."""
        vec = np.zeros(self.num_buckets, dtype=np.float32)

        # Pad to context_len
        recent = recent_chars[-self.context_len:].lower()
        recent = recent.rjust(self.context_len)

        # Hash character n-grams with position info
        for n in [1, 2, 3]:  # Unigrams, bigrams, trigrams
            for i in range(len(recent) - n + 1):
                ngram = recent[i:i+n]
                bucket = self._hash_ngram(ngram, offset=i)
                vec[bucket] += 1.0

        # No normalization - use raw counts for Z80 compatibility
        return vec


def create_training_examples(query: str, response: str,
                            query_encoder: TrigramEncoder,
                            context_encoder: ContextEncoder) -> List[Tuple[np.ndarray, int]]:
    """
    Create training examples from a (query, response) pair.

    For response "hello", creates:
    - (query + context(""), 'h')
    - (query + context("h"), 'e')
    - (query + context("he"), 'l')
    - ...
    - (query + context("hello"), EOS)
    """
    examples = []
    query_vec = query_encoder.encode(query)

    # Add EOS to response
    response_with_eos = response + "\x00"

    output_so_far = ""
    for char in response_with_eos:
        # Encode current context
        context_vec = context_encoder.encode(output_so_far)

        # Combine query and context
        full_input = np.concatenate([query_vec, context_vec])

        # Target is next character (or EOS)
        target = char_to_idx(char) if char != "\x00" else EOS_IDX

        examples.append((full_input, target))
        output_so_far += char

    return examples


def load_training_pairs(filenames: List[str]|None = None) -> List[Tuple[str, str]]:
    """Load query-response pairs from training data files."""
    if filenames is None:
        filenames = DEFAULT_TRAINING_SETS

    pairs = []
    for filename in filenames:
        try:
            with open(filename) as f:
                for line in f:
                    obj = json.loads(line)
                    query = obj['query']
                    response = obj['response']
                    pairs.append((query, response))
            print(f"  Loaded {filename}")
        except FileNotFoundError:
            print(f"  Skipped {filename} (not found)")

    return pairs


class AutoregressiveModel(nn.Module):
    """Autoregressive character model with configurable depth."""

    def __init__(self, input_size: int = 256, hidden_sizes: list = [128, 128],
                 num_chars: int = 64):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_chars = num_chars

        # Build layers dynamically
        self.layers = nn.ModuleList()
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(OverflowAwareLinear(prev_size, hidden_size))
            prev_size = hidden_size
        self.layers.append(OverflowAwareLinear(prev_size, num_chars))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, use_int: bool = False, quant_temp: float = 1.0) -> torch.Tensor:
        if use_int:
            return self._forward_int(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, quant_temp=quant_temp)
            x = self.relu(x)
        x = self.layers[-1](x, quant_temp=quant_temp)
        return x

    def _forward_int(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass simulating Z80 integer inference (16-bit accumulator)."""
        # Scale input like Z80 does
        x = (x * 32).round()

        for i, layer in enumerate(self.layers):
            # Quantize weights to {-2, -1, 0, +1} (4 values for 2 bits)
            w = layer.weight
            scale = torch.quantile(w.abs().flatten(), 0.95).clamp(min=1e-6)
            w_quant = torch.clamp(torch.round(w / scale), -2, 1)

            # Quantize bias
            b_quant = torch.round(layer.bias * 32)

            # Integer matmul with 16-bit overflow simulation
            x = x @ w_quant.T + b_quant
            # Simulate 16-bit signed overflow (wrap around)
            x = ((x + 32768) % 65536) - 32768

            # Shift down (divide by 4, arithmetic right shift)
            x = torch.div(x, 4, rounding_mode='trunc')

            # ReLU (except last layer)
            if i < len(self.layers) - 1:
                x = torch.relu(x)

        return x

    def get_overflow_stats(self) -> dict:
        return {f'layer{i+1}': layer.get_overflow_risk()
                for i, layer in enumerate(self.layers)}

    def reset_overflow_stats(self):
        for layer in self.layers:
            layer.reset_overflow_stats()

    def compute_quantization_loss(self) -> torch.Tensor:
        return sum(layer.get_quantization_loss() for layer in self.layers)

    def compute_total_overflow_penalty(self, x: torch.Tensor) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=x.device)
        for i, layer in enumerate(self.layers[:-1]):
            penalty = penalty + layer.compute_overflow_penalty(x)
            x = self.relu(layer(x))
        penalty = penalty + self.layers[-1].compute_overflow_penalty(x)
        return penalty

    def get_quantized_params(self) -> dict:
        """Extract 2-bit quantized weights."""
        params = {}

        for i, layer in enumerate(self.layers):
            name = f'fc{i+1}'
            with torch.no_grad():
                w = layer.weight
                w_scale = torch.quantile(w.abs().flatten(), 0.95).clamp(min=1e-6)
                w_scaled = w / w_scale
                w_quant = torch.clamp(torch.round(w_scaled), -2, 1).cpu().numpy().astype(np.int8)

                b = layer.bias
                b_quant = torch.round(b * 32).cpu().numpy().astype(np.int16)

                params[f'{name}_weight'] = w_quant
                params[f'{name}_bias'] = b_quant

        return params


def generate_response(model: AutoregressiveModel, query: str,
                     query_encoder: TrigramEncoder,
                     context_encoder: ContextEncoder,
                     max_len: int = 50, use_int: bool = True) -> str:
    """Generate a response character by character."""
    model.eval()

    query_vec = query_encoder.encode(query)
    output = ""

    with torch.no_grad():
        for _ in range(max_len):
            context_vec = context_encoder.encode(output)
            full_input = np.concatenate([query_vec, context_vec])
            x = torch.tensor(full_input, dtype=torch.float32).unsqueeze(0)

            logits = model(x, use_int=use_int)
            next_char_idx = logits.argmax(dim=1).item()

            # Stop on EOS
            if next_char_idx == EOS_IDX:
                break

            next_char = idx_to_char(next_char_idx)
            output += next_char

    return output.strip()


def train_autoregressive(epochs: int = 500, lr: float = 0.01) -> AutoregressiveModel:
    """Train the autoregressive model."""
    global CHARSET, CHAR_TO_IDX, IDX_TO_CHAR, EOS_IDX, NUM_CHARS

    print("=" * 60)
    print("Autoregressive Character Model Training")
    print("=" * 60)

    # Build charset from training data
    print("\nBuilding charset from training data...")
    CHARSET = build_charset_from_data()
    CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}
    IDX_TO_CHAR = {i: c for i, c in enumerate(CHARSET)}
    EOS_IDX = len(CHARSET) - 1
    NUM_CHARS = len(CHARSET)
    print(f"Charset ({NUM_CHARS} chars): {repr(CHARSET[:-1])} + EOS")

    # Create encoders
    query_encoder = TrigramEncoder(num_buckets=128)
    context_encoder = ContextEncoder(num_buckets=128, context_len=8)

    # Load data and create examples
    print("\nLoading training data...")
    pairs = load_training_pairs()
    print(f"Loaded {len(pairs)} query-response pairs")

    # Generate character-level examples
    print("Generating character-level examples...")
    all_examples = []
    for query, response in pairs:
        examples = create_training_examples(query, response, query_encoder, context_encoder)
        all_examples.extend(examples)

    print(f"Created {len(all_examples)} character prediction examples")

    # Convert to tensors
    X = np.stack([ex[0] for ex in all_examples])
    y = np.array([ex[1] for ex in all_examples])

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    print(f"Dataset shape: {X.shape}")

    # Show character distribution
    char_counts = Counter(y.numpy())
    print(f"Most common outputs: {[(idx_to_char(i), c) for i, c in char_counts.most_common(10)]}")

    # Create model - larger architecture for better capacity
    hidden_sizes = [256, 192, 128]
    model = AutoregressiveModel(input_size=256, hidden_sizes=hidden_sizes, num_chars=NUM_CHARS)

    total_params = sum(p.numel() for p in model.parameters())
    arch_str = "256 → " + " → ".join(map(str, hidden_sizes)) + f" → {NUM_CHARS}"
    print(f"\nModel: {arch_str}")
    print(f"Parameters: {total_params:,}")

    # Try to load existing checkpoint for resume
    start_epoch = 0
    prev_best_acc = 0.0
    checkpoint_file = 'command_model_autoreg.pt'
    try:
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        # Check if architecture matches (dimensions must be same)
        arch = checkpoint.get('architecture', {})
        if arch.get('num_classes') == NUM_CHARS:
            model.load_state_dict(checkpoint['model_state'])
            start_epoch = checkpoint.get('total_epochs', 0)
            prev_best_acc = checkpoint.get('best_int_acc', 0.0)
            print(f"\nResuming from epoch {start_epoch} (best IntAcc: {prev_best_acc:.1%})")
        else:
            print(f"\nOutput size changed ({arch.get('num_classes')} → {NUM_CHARS}), starting fresh")
    except FileNotFoundError:
        print(f"\nNo checkpoint found, starting fresh")
    except Exception as e:
        print(f"\nCouldn't load checkpoint: {e}, starting fresh")

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Training for {epochs} more epochs (total will be {start_epoch + epochs})...")

    import time
    epoch_times = []

    # Track best models
    best_int_acc = prev_best_acc
    best_epoch = start_epoch
    best_state = None

    try:
      for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        model.reset_overflow_stats()
        optimizer.zero_grad()

        # Progressive quantization: never pure float, always some quantization
        # Start at 0.3, ramp to 1.0 over 80% of training
        quant_temp = 0.3 + 0.7 * min(1.0, epoch / (epochs * 0.8))

        outputs = model(X, quant_temp=quant_temp)
        ce_loss = criterion(outputs, y)

        # QAT regularization (stronger for better int accuracy)
        quant_loss = model.compute_quantization_loss() * 0.10
        overflow_loss = model.compute_total_overflow_penalty(X) * 0.03

        loss = ce_loss + quant_loss + overflow_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                acc = (preds == y).float().mean()
                # Also check integer accuracy
                int_outputs = model(X, use_int=True)
                int_preds = int_outputs.argmax(dim=1)
                int_acc = (int_preds == y).float().mean()
                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)
                avg_time = sum(epoch_times) / len(epoch_times)

                # Track best IntAcc model
                int_acc_val = int_acc.item()
                total_epoch = start_epoch + epoch + 1
                if int_acc_val > best_int_acc:
                    best_int_acc = int_acc_val
                    best_epoch = total_epoch
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    marker = " *BEST*"
                else:
                    marker = ""

                print(f"Epoch {total_epoch:3d}: CE={ce_loss.item():.4f}, Acc={acc:.1%}, IntAcc={int_acc:.1%}, QTemp={quant_temp:.2f}, Overflow={max(model.get_overflow_stats().values()):.3f}x  ({avg_time:.2f}s/epoch){marker}")

                # Sample inference every 25 epochs
                if (epoch + 1) % 5 == 0:
                    sample_queries = [
                        "HOW ARE YOU?",
                        "WHAT DO YOU THINK?",
                        "WHERE ARE YOU GOING?",
                        "THANK YOU.",
                        "I NEED HELP.",
                    ]
                    sample_q = sample_queries[(epoch // 25) % len(sample_queries)]
                    model.eval()
                    sample_resp = generate_response(model, sample_q, query_encoder, context_encoder, max_len=30)
                    model.train()
                    print(f"         → '{sample_q}' → '{sample_resp}'")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted at epoch {epoch + 1}!")

    # Report best model
    print("\n" + "=" * 60)
    print(f"Best IntAcc: {best_int_acc:.1%} at epoch {best_epoch}")
    print("=" * 60)

    # Restore best model for testing and saving
    if best_state is not None:
        model.load_state_dict(best_state)
        #print("Restored best model weights")

    # Test generation
    print("\n" + "=" * 60)
    print("Testing Generation:")
    print("=" * 60)

    test_queries = [
        "HOW ARE YOU TODAY?",
        "WHAT DO YOU THINK?",
        "I NEED SOME HELP.",
        "WHERE ARE YOU GOING?",
        "THANK YOU VERY MUCH.",
    ]

    for query in test_queries:
        response = generate_response(model, query, query_encoder, context_encoder)
        print(f"  '{query}' → '{response}'")

    # Save model with resume info
    final_epoch = start_epoch + epochs
    torch.save({
        'model_state': model.state_dict(),
        'architecture': {
            'input_size': 256,
            'hidden_sizes': hidden_sizes,
            'num_classes': NUM_CHARS,
        },
        'charset': CHARSET,
        'total_epochs': final_epoch,
        'best_int_acc': best_int_acc,
        'best_epoch': best_epoch,
    }, 'command_model_autoreg.pt')
    print(f"\nSaved to command_model_autoreg.pt (total epochs: {final_epoch}, best: {best_int_acc:.1%} @ {best_epoch})")

    return model


def load_pairs_from_stdin() -> List[Tuple[str, str]]:
    """Load pairs from stdin - pipe separated: input|response"""
    import sys
    pairs = []

    for line in sys.stdin:
        line = line.strip()
        if '|' not in line:
            continue

        parts = line.split('|', 1)
        if len(parts) != 2:
            continue

        query = parts[0].strip().upper()
        response = parts[1].strip().upper()

        if len(query) >= 2 and len(response) >= 1:
            # Truncate smartly
            if len(query) > 60:
                query = query[:60].rsplit(' ', 1)[0] if ' ' in query[40:60] else query[:60]
            if len(response) > 50:
                response = response[:50].rsplit(' ', 1)[0] if ' ' in response[30:50] else response[:50]
            pairs.append((query, response))

    return pairs


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Train autoregressive model')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Epochs to train')
    parser.add_argument('--file', '-f', type=str, default=None, help='Training data file (default: stdin)')
    parser.add_argument('--chat', action='store_true', help='Interactive chat after training')
    args = parser.parse_args()

    # Check if data is being piped in
    if not sys.stdin.isatty() or args.file:
        # Override load_training_pairs to use stdin/file
        if args.file:
            # Read from file as alternating lines
            with open(args.file) as f:
                stdin_data = f.read()
            import io
            sys.stdin = io.StringIO(stdin_data)

        # Monkey-patch to use stdin loader
        _original_load = load_training_pairs
        load_training_pairs = lambda *a, **kw: load_pairs_from_stdin()

    model = train_autoregressive(epochs=args.epochs)

    # Interactive chat session
    if args.chat:
        print("\n" + "=" * 60)
        print("Interactive Chat (type '!' to exit)")
        print("=" * 60)

        query_encoder = TrigramEncoder()
        context_encoder = ContextEncoder()

        while True:
            try:
                query = input("> ").strip()
                if not query:
                    continue
                if query == '!':
                    break
                response = generate_response(model, query, query_encoder, context_encoder, max_len=50)
                print(response)
            except (EOFError, KeyboardInterrupt):
                break

        print("\nBye!")
