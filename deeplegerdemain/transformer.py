from typing import *

import jax
import jax.numpy as jnp
import flax.linen as nn

def positional_encoding(seq_len, embedding_dim):
  r"""This implements 3.5 Positional Encoding

  Say $PE$ is a field of position encoding, to be addictively attach to original embedding,
  $$
  \begin{aligned}
  &PE[pos, 2i  ] &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
  &PE[pos, 2i+1] &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
  \end{aligned}
  $$
  """
  pos = jnp.arange(seq_len).reshape((-1, 1))
  i = jnp.arange((embedding_dim - 1) // 2 + 1).reshape((1, -1))
  denom = jnp.power(10000.0, 2 * i / embedding_dim)
  return jnp.stack([jnp.sin(pos / denom), jnp.cos(pos / denom)], axis=2).reshape(seq_len, -1)[:, :embedding_dim]


def dot_product_attention_weights(q: jnp.ndarray,
                                  k: jnp.ndarray,
                                  mask: Optional[jnp.ndarray] = None):
  r"""Computes masked dot-product attention weights given query and key.

  $$
  \text{attn\_weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
  $$

  Args:
    q: queries for calculating attention with shape of [num_q,  qk_depth].
    k: keys for calculating attention with shape of    [num_kv, qk_depth].
    mask: mask for the attention weights. This should be of shape [num_q, num_kv].
      This can be used for incorporating causal masks.
      Attention weights are masked out if their corresponding mask value is `False`.
  """
  assert q.ndim == k.ndim == 2, 'q, k must have rank 2.'
  assert q.shape[-1] == k.shape[-1], 'q, k depths must match.'

  depth = q.shape[-1]
  q = q / jnp.sqrt(depth)
  attn_weights = jnp.einsum('...qd,...kd->...qk', q, k)

  # apply attention mask
  if mask is not None:
    big_neg = jnp.finfo(attn_weights.dtype).min  # -3.4028235e+38 for jnp.float32
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  attn_weights = jax.nn.softmax(attn_weights)

  return attn_weights


# class SingleHeadAttention(nn.Module):


# class MultiHeadAttention(nn.Module):
#   num_heads: int
