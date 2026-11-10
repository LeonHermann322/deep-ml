import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    seq_len, d_k = Q.shape
    query_key = Q @ K.T
    norm_query_key = query_key / np.sqrt(d_k)
    exp_norm_query_key = np.exp(norm_query_key)
    softmax_query_key = exp_norm_query_key / np.sum(exp_norm_query_key, axis=1, keepdims=True)
    return softmax_query_key @ V


