import argparse
import yaml
import warnings

import tensorflow as tf

from trainer import Trainer

with open('./conf/params.yaml', 'rb') as f:
    conf = yaml.safe_load(f)['model_params']

parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str, choices=['oracle', 'mf', 'rmf', 'bpr', 'ubpr', 'ipwbpr', 'ipwbpr_opt0', 'ipwbpr_opt1', 'ipwbpr_opt2', 'ipwbpr_opt3', 'ubpr_nclip'])
parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'])
parser.add_argument('--eps', default=5., type=float)
parser.add_argument('--pow_list', default=[1.], type=float, nargs='*')
parser.add_argument('--iters', default=5, type=int)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    eps = args.eps
    pow_list = args.pow_list
    model_name = args.model_name
    with open('./conf/params.yaml', 'rb') as f:
      conf = yaml.safe_load(f)[model_name]
    dim = conf['dim']
    lam = conf['lam']
    eta = conf['eta']
    batch_size = conf['batch_size']
    max_iters = conf['max_iters']
    beta = conf['beta']
    pair_weight = conf['pair_weight']
    norm_weight = conf['norm_weight']
    iters = args.iters
    optimizer = args.optimizer
    
    # run simulations
    trainer = Trainer(dim=dim, lam=lam, batch_size=batch_size,
                      max_iters=max_iters, eta=eta, model_name=model_name,
                      beta=beta, pair_weight=pair_weight, 
                      norm_weight=norm_weight, optimizer=optimizer)
    trainer.run(iters=iters, eps=eps, pow_list=pow_list)

    print('\n', '=' * 25, '\n')
    print(f'Finished Running {model_name}!')
    print('\n', '=' * 25, '\n')
