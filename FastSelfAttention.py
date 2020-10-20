import torch
from torch import nn
import functools
import numpy as np


def generalized_kernel_feature_creator(data, projection_matrix,normalize_data, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001):
    """Constructs kernel features for fast generalized attention.
    Args:
      data: input for which features are computes
      projection_matrix: matrix used to compute features
      batch_dims_t: tuple of batch dimensions
      precision: precision parameter
      kernel_fn: kernel function used
      kernel_epsilon: additive positive term added to every feature for numerical
        stability
      normalize_data: predicate indicating whether data should be normalized.
    Returns:
      Random features for fast generalized attention.
    """
    if normalize_data:
        data_normalizer = 1.0 / (np.sqrt(np.sqrt(data.shape[-1])))
    else:
        data_normalizer = 1.0
    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon
    else:
        # data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
        data_thick_random_matrix =  projection_matrix
        # print(data_normalizer.size, data.size(), data_thick_random_matrix.size())
        data_dash = torch.einsum('bse,er->bsr', data_normalizer * data, data_thick_random_matrix)
    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime


def _numerator(z_slice_shape, seqlen, qs, ks, vs):
    init_value = torch.zeros(z_slice_shape)
    # p, W = lax.scan(body, init_value, (qs, ks, vs))
    p = init_value
    Xs = []
    for i in range(seqlen):
        q, k, v = qs[:, i, :], ks[:, i, :], vs[:, i, :]
        p += torch.einsum('br, bd->brd', k, v)
        X_slice = torch.einsum('br,brd->bd', q, p)
        Xs.append(X_slice)

    W = torch.stack(Xs)  # s b d
    return W.transpose(0, 1)


def _denominator_fwd(t_slice_shape, qs, ks, seqlen):
    p = torch.zeros(t_slice_shape)
    # p, R = lax.scan(body, p, (qs, ks))
    Xs = []
    for i in range(seqlen):
        q, k = qs[:, i, :], ks[:, i, :]
        p += k
        x = torch.einsum('br, br->b', q, p)
        Xs.append(x)

    R = torch.stack(Xs)  # s b r

    return R.transpose(0, 1)


def dot_product_attention(query, key, value, projection_matrix, kernel_feature_creator = generalized_kernel_feature_creator, unidirectional = False, renormalize_attention = True, numerical_stabilizer=0.0,):
    batchsize = query.shape[0]
    seqlen = query.shape[1]
    # Constructing tensors Q^{'} and K^{'}.
    query_prime = kernel_feature_creator(query, projection_matrix, normalize_data=True)
    key_prime = kernel_feature_creator(key, projection_matrix, normalize_data=False)

    if unidirectional:

        W = _numerator((batchsize, key_prime.shape[-1], value.shape[-1]), seqlen, query_prime, key_prime, value)

        if not renormalize_attention:
            # Unidirectional, not-normalized attention.
            result = W
            return result
        else:
            # Unidirectional, normalized attention.
            thick_all_ones = torch.ones([seqlen])

            index = attention_dims_t[0]
            t_slice_shape = key_prime.shape[0:len(batch_dims_t)] + (
                key_prime.shape[-1],)
            R = _denominator([batchsize, key_prime.shape[-1]], query_prime, key_prime)

    else:
        # Constructing Z = (K^{'})^{T}V
        # Z (bs, <non-attention dims>, num_heads, channels_m, channels_v)
        Z = torch.einsum('bsr,bsd->brd', key_prime, value)

        # Constructing W = Q^{'}Z = Q^{'}(K^{'})^{T}V
        # q (bs, <non-attention dims>, num_heads, <attention dims>, channels_m)
        # Z (bs, <non-attention dims>, num_heads, channels_m, channels_v)
        # W (bs,  <non-attention dims>, num_heads, <attention dims>, channels_v)
        W = torch.einsum('bsr,brd->bsd', query_prime, Z)

        if not renormalize_attention:
            # Bidirectional, not-normalized attention.
            return result
        else:
            # Bidirectional, normalized attention.
            thick_all_ones = torch.ones([key.shape[1]])
            # Construct T = (K^{'})^{T} 1_L
            # k (bs, <non-attention dims>, num_heads, <attention dims>, channels)
            # print(key_prime.size(), thick_all_ones.size())
            T = torch.einsum('bsr,s->br', key_prime, thick_all_ones)

            # Construct partition function: R = Q^{'} T = Q^{'}(K^{'})^{T} 1_L
            # q_p (bs, <non-attention dims>, num_heads, <attention dims>, channs_m)
            # T   (bs, <non-attention dims>, num_heads, channels_m)
            R = torch.einsum('bsr,br->bs', query_prime, T)

    # print(W.size(), R.size())
    R = R + 2 * numerical_stabilizer * (torch.abs(R) <= numerical_stabilizer).float()
    R = 1.0 / R
    R = R.unsqueeze(-1)
    # W (bs, <non-attention dims>, num_heads, <attention dims>, channels_v)
    # R (bs, <non-attention dims>, num_heads, <attention dims>, extra_channel)
    # print(W.size(), R.size())
    result = W * R
    return result
