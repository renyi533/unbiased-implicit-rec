from pathlib import Path
from typing import List
from dataclasses import dataclass

import numpy as np
from regex import W
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops

from utils.data_generator import generate_sys_data
from utils.evaluator import Evaluator
from utils.models import ImplicitRecommender, PairwiseRecommender, IPWPairwiseRecommender


def rec_trainer(sess: tf.Session, model: ImplicitRecommender,
                train: np.ndarray, val: np.ndarray, test: np.ndarray,
                max_iters: int = 500, batch_size: int = 2**12, model_name: str = 'mf',
                eps: float = 5, pow: float = 1.0, num: int = 0) -> float:
    """Train and Evaluate Implicit Recommender."""
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # count the num of train-val data.
    num_train = train.shape[0]
    num_val = val.shape[0]
    # specify model type.
    oracle = 'oracle' in model_name
    ips = 'rmf' in model_name

    # train the given implicit recommender
    np.random.seed(12345)
    for _ in np.arange(max_iters):
        idx = np.random.choice(np.arange(num_train), size=batch_size)
        train_batch = train[idx]
        train_label = train_batch[:, 6] if oracle else train_batch[:, 2]
        val_label = val[:, 6] if oracle else val[:, 2]
        train_score = np.expand_dims(train_batch[:, 5], 1) if ips else np.ones((batch_size, 1))
        val_score = np.expand_dims(val[:, 5], 1) if ips else np.ones((num_val, 1))

        _, loss = sess.run([model.apply_grads, model.weighted_ce],
                           feed_dict={model.users: train_batch[:, 0], model.items: train_batch[:, 1],
                                      model.labels: np.expand_dims(train_label, 1), model.scores: train_score})
        val_loss = sess.run(model.weighted_ce, feed_dict={model.users: val[:, 0], model.items: val[:, 1],
                                                          model.labels: np.expand_dims(val_label, 1), model.scores: val_score})
        test_loss = sess.run(model.ce, feed_dict={model.users: test[:, 0], model.items: test[:, 1],
                                                  model.labels: np.expand_dims(test[:, 4], 1)})
        train_loss_list.append(loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)

    path = Path(f'../logs/{model_name}')
    # save embeddings.
    (path / 'embeds').mkdir(parents=True, exist_ok=True)
    u_emb, i_emb, u_bias, i_bias = sess.run([model.user_embeddings, model.item_embeddings, model.user_b, model.item_b])
    np.save(file=path / 'embeds/user_embed.npy', arr=u_emb)
    np.save(file=path / 'embeds/item_embed.npy', arr=i_emb)
    np.save(file=path / 'embeds/user_bias.npy', arr=u_bias)
    np.save(file=path / 'embeds/item_bias.npy', arr=i_bias)
    # save train and val loss curves.
    (path / 'loss').mkdir(parents=True, exist_ok=True)
    np.save(file=path / f'loss/train_{eps}_{pow}_{num}.npy', arr=train_loss_list)
    np.save(file=path / f'loss/val_{eps}_{pow}_{num}.npy', arr=val_loss_list)
    np.save(file=path / f'loss/test_{eps}_{pow}_{num}.npy', arr=test_loss_list)

    sess.close()

    return test_loss_list[np.argmin(val_loss_list)]

def pairwise_rec_trainer(sess: tf.Session, model: PairwiseRecommender,
                train: np.ndarray, val: np.ndarray, test: np.ndarray,
                max_iters: int = 500, batch_size: int = 2**12, model_name: str = 'bpr',
                eps: float = 5, pow: float = 1.0, num: int = 0) -> float:
    """Train and Evaluate Implicit Recommender."""
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # count the num of train-val data.
    num_train = train.shape[0]
    num_val = val.shape[0]
    # specify model type.
    ips = 'ubpr' in model_name or 'ipwbpr' in model_name
    print('model name: %s, ips: %d' % (model_name, ips))
    # train the given implicit recommender
    np.random.seed(12345)
    for i in np.arange(max_iters):
        idx = np.random.choice(np.arange(num_train), size=batch_size)
        train_batch = train[idx]
        # update user-item latent factors
        if not ips:
            _, loss = sess.run([model.apply_grads, model.loss],
                               feed_dict={model.users: train_batch[:, 0],
                                          model.pos_items: train_batch[:, 1],
                                          model.scores1: np.ones((batch_size, 1)),
                                          model.items2: train_batch[:, 2],
                                          model.labels2: np.zeros((batch_size, 1)),
                                          model.scores2: np.ones((batch_size, 1))})
        else:
            _, loss = sess.run([model.apply_grads, model.loss],
                               feed_dict={model.users: train_batch[:, 0],
                                          model.pos_items: train_batch[:, 1],
                                          model.scores1: np.expand_dims(train_batch[:, 9], 1),
                                          model.items2: train_batch[:, 2],
                                          model.labels2: np.expand_dims(train_batch[:, 4], 1),
                                          model.scores2: np.expand_dims(train_batch[:, 10], 1)})
        train_loss_list.append(loss)
        #print('finished train of %d iter' % (i))
        # calculate a test loss
        test_loss = sess.run(model.ideal_loss,
                             feed_dict={model.users: test[:, 0],
                                        model.pos_items: test[:, 1],
                                        model.rel1: np.expand_dims(test[:, 7], 1),
                                        model.items2: test[:, 2],
                                        model.rel2: np.expand_dims(test[:, 8], 1)})
        test_loss_list.append(test_loss)
        #print('finished test of %d iter' % (i))
        # calculate a validation loss
        if not ips:
            val_loss = sess.run(model.unbiased_loss,
                            feed_dict={model.users: val[:, 0],
                                       model.pos_items: val[:, 1],
                                       model.scores1: np.ones((num_val, 1)),
                                       model.items2: val[:, 2],
                                       model.labels2: np.zeros((num_val, 1)),
                                       model.scores2: np.ones((num_val, 1))})
        else:
            val_loss = sess.run(model.unbiased_loss,
                            feed_dict={model.users: val[:, 0],
                                       model.pos_items: val[:, 1],
                                       model.scores1: np.expand_dims(val[:, 9], 1),
                                       model.items2: val[:, 2],
                                       model.labels2: np.expand_dims(val[:, 4], 1),
                                       model.scores2: np.expand_dims(val[:, 10], 1)})
        val_loss_list.append(val_loss)
        #print('finished val of %d iter' % (i))

    #u_emb, i_emb = sess.run([model.user_embeddings, model.item_embeddings])

    path = Path(f'../logs/{model_name}')
    # save embeddings.
    (path / 'embeds').mkdir(parents=True, exist_ok=True)
    u_emb, i_emb, u_bias, i_bias = sess.run([model.user_embeddings, model.item_embeddings, model.user_b, model.item_b])
    np.save(file=path / 'embeds/user_embed.npy', arr=u_emb)
    np.save(file=path / 'embeds/item_embed.npy', arr=i_emb)
    np.save(file=path / 'embeds/user_bias.npy', arr=u_bias)
    np.save(file=path / 'embeds/item_bias.npy', arr=i_bias)
    # save train and val loss curves.
    (path / 'loss').mkdir(parents=True, exist_ok=True)
    np.save(file=path / f'loss/train_{eps}_{pow}_{num}.npy', arr=train_loss_list)
    np.save(file=path / f'loss/val_{eps}_{pow}_{num}.npy', arr=val_loss_list)
    np.save(file=path / f'loss/test_{eps}_{pow}_{num}.npy', arr=test_loss_list)

    sess.close()

    return test_loss_list[np.argmin(val_loss_list)]
@dataclass
class Trainer:
    """Trainer Class for ImplicitRecommender."""
    dim: int = 5
    lam: float = 1e-5
    max_iters: int = 500
    batch_size: int = 12
    eta: float = 0.1
    model_name: str = 'oracle'
    beta: float = 0.0
    pair_weight: int = 0
    norm_weight: bool = True
    optimizer: str = 'sgd'

    def run(self, iters: int, eps: float, pow_list: List[float]) -> None:
        """Train implicit recommenders."""
        path = Path(f'../logs/{self.model_name}/results')
        path.mkdir(parents=True, exist_ok=True)

        results = []
        for pow in pow_list:
            # generate semi-synthetic data
            ips = 'ubpr' in self.model_name or self.model_name == 'rmf' or 'ipwbpr' in self.model_name
            data, pair_data = generate_sys_data(eps=eps, pow=pow, enable_ips=ips)
            
            if self.model_name == 'bpr' or 'ubpr' in self.model_name or 'ipwbpr' in self.model_name:
                num_users = np.int(data[:, 0].max() + 1)
                num_items = np.int(data[:, 1].max() + 1)
                # data splitting
                train, test = pair_data, pair_data  # train-test split
                train, val = train_test_split(train, test_size=0.1, random_state=0)  # train-val split

                for i in np.arange(iters):
                    # define the TF graph
                    # different initialization of model parameters for each iteration
                    tf.set_random_seed(i)
                    ops.reset_default_graph()
                    sess = tf.Session()
                    # define the implicit recommender model
                    if 'ipwbpr' in self.model_name:
                        rec = IPWPairwiseRecommender(num_users=num_users, num_items=num_items,
                                            dim=self.dim, lam=self.lam, eta=self.eta, 
                                            pair_weight=self.pair_weight, norm_weight=self.norm_weight,
                                            optimizer=self.optimizer)                        
                    else:
                        rec = PairwiseRecommender(num_users=num_users, num_items=num_items,
                                            dim=self.dim, lam=self.lam, eta=self.eta, beta=self.beta,
                                            optimizer=self.optimizer)
                    # train and evaluate the recommender
                    score = pairwise_rec_trainer(sess, model=rec, train=train, val=val, test=test,
                                        max_iters=self.max_iters, batch_size=2**self.batch_size,
                                        model_name=self.model_name, eps=eps, pow=pow, num=i)
                    results.append(score)
           
            else:
                num_users = np.int(data[:, 0].max() + 1)
                num_items = np.int(data[:, 1].max() + 1)
                # data splitting
                train, test = data, data[data[:, 2] == 0, :]  # train-test split
                train, val = train_test_split(train, test_size=0.1, random_state=0)  # train-val split

                for i in np.arange(iters):
                    # define the TF graph
                    # different initialization of model parameters for each iteration
                    tf.set_random_seed(i)
                    ops.reset_default_graph()
                    sess = tf.Session()
                    # define the implicit recommender model
                    rec = ImplicitRecommender(num_users=num_users, num_items=num_items,
                                            dim=self.dim, lam=self.lam, eta=self.eta,
                                            optimizer=self.optimizer)
                    # train and evaluate the recommender
                    score = rec_trainer(sess, model=rec, train=train, val=val, test=test,
                                        max_iters=self.max_iters, batch_size=2**self.batch_size,
                                        model_name=self.model_name, eps=eps, pow=pow, num=i)
                    results.append(score)

            train, test = data, data[data[:, 2] == 0, :]  # train-test split
            train, val = train_test_split(train, test_size=0.1, random_state=0)  # train-val split                    
            evaluator = Evaluator(train=train, val=val, test=test, model_name=self.model_name)
            evaluator.evaluate(eps=eps, pow=pow)     
        np.save(path / f'eps_{eps}.npy', arr=np.array(results).reshape((len(pow_list), iters)).T)

