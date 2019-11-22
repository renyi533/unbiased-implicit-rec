"""
Codes for generating semi-synthetic datasets used in the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""
import codecs
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops

from model import MF


class MovieLensLoader:
    """Load and Preprocess datasets."""

    def __init__(self) -> None:
        """Initialize Class."""

    def load(self, seed: int = 0) -> Tuple[np.array, int, int]:
        """Load and Preprocess datasets."""
        # load dataset.
        with codecs.open(f'../data/ml-100k/ml-100k.data', 'r', 'utf-8', errors='ignore') as f:
            data = pd.read_csv(f, delimiter='\t', header=None).loc[:, :2]
            data.rename(columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)

        # reset index and specifiy the number of users and items.
        data.user, data.item = data.user - 1, data.item - 1
        # train-val-test, split
        data = data.values
        num_users, num_items = data[:, 0].max() + 1, data[:, 1].max() + 1

        return data, num_users, num_items


ITERS: int = 300
BATCH_SIZE: int = 2 ** 8
DIM: int = 20
ETA: float = 0.01


def generate_sys_data(eps: float = 5, pow: float = 0.5) -> np.ndarray:
    """Generate semi-synthetic data from ml-100k dataset.

    return:
        columns |     0     |     1     |     2     |      3       |       4       |           5            |            6            |
        factors |  user_id  |  item_id  | click (Y) | exposure (O) | relevance (R) | exposure param (theta) | relevance param (gamma) |
    """
    np.random.seed(12345)
    loader = MovieLensLoader()
    data, num_users, num_items = loader.load()

    # step 1&3 in Section 5.1.1.
    # generate relevance parameter (gamma) by MF.
    ops.reset_default_graph()
    sess = tf.Session()
    tf.set_random_seed(12345)
    model = MF(num_users=num_users, num_items=num_items, dim=DIM, eta=ETA)

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # learn matrix factorization models on ML 100K
    for i in np.arange(ITERS):
        # mini-batch samples
        idx = np.random.choice(np.arange(data.shape[0], dtype=int), size=np.int(BATCH_SIZE))
        train_batch: np.ndarray = data[idx]
        # update user-item latent factors by SGD
        _ = sess.run(model.apply_grads_mse,
                     feed_dict={model.users: train_batch[:, 0], model.items: train_batch[:, 1],
                                model.labels: np.expand_dims(train_batch[:, 2], 1)})
    # obrain dense user-item matrix
    u_embed, i_embed = sess.run([model.user_embeddings, model.item_embeddings])
    mat = np.clip(u_embed @ i_embed.T, 1, 5)
    gamma = sigmoid(mat - eps)
    y = np.random.binomial(n=1, p=gamma)

    # step 2&4 in Section 5.1.1.
    # generate exposure parameter (theta) by logistic MF.
    all_data = pd.DataFrame(np.zeros((num_users, num_items))).stack().reset_index()
    all_data = all_data.values[:, :2]
    pos_data = data[:, :2]
    unlabeled_data = np.array(list(set(map(tuple, all_data)) - set(map(tuple, pos_data))), dtype=int)
    data = np.r_[np.c_[pos_data, np.ones(pos_data.shape[0])], np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])]]

    ops.reset_default_graph()
    sess = tf.Session()
    tf.set_random_seed(12345)
    model = MF(num_users=num_users, num_items=num_items, dim=DIM, eta=ETA)

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # learn matrix factorization models on ML 100K
    for i in np.arange(ITERS):
        # mini-batch samples
        idx = np.random.choice(np.arange(data.shape[0], dtype=int), size=np.int(BATCH_SIZE))
        train_batch = data[idx]
        # update user-item latent factors by SGD
        _ = sess.run(model.apply_grads_ce,
                     feed_dict={model.users: train_batch[:, 0], model.items: train_batch[:, 1],
                                model.labels: np.expand_dims(train_batch[:, 2], 1)})
    u_embed, i_embed = sess.run([model.user_embeddings, model.item_embeddings])
    theta = sigmoid(u_embed @ i_embed.T) ** pow
    expo = np.random.binomial(n=1, p=theta)

    # generate and save semi-synthetic dataset.
    os.makedirs('../data/sys_data/', exist_ok=True)
    data_ = pd.DataFrame(np.zeros((num_users, num_items))).stack().reset_index().values[:, :2]
    theta_ = pd.DataFrame(theta).stack().reset_index().values[:, -1]
    gamma_ = pd.DataFrame(gamma).stack().reset_index().values[:, -1]
    expo_ = pd.DataFrame(expo).stack().reset_index().values[:, -1]
    y_ = pd.DataFrame(y).stack().reset_index().values[:, -1]
    y_obs = expo_ * y_
    sys_data = np.c_[data_, y_obs, expo_, y_, theta_, gamma_]
    np.save(file=f'../data/sys_data/eps_{eps}_pow_{pow}.npy', arr=sys_data)

    # specifications for the columns of the generated data array
    # ==========================================================
    # columns |     0    |    1     |            2            |            3            |            4             |           5            |        6        |
    # factors |  user_id | item_id  | observed_interction (Y) | unobserved_exposure (O) | unobserved_relevance (R) | exposure_param (theta) | relevance_param |

    return sys_data


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Calculate sigmoid."""
    return 1 / (1 + np.exp(-x))
