import argparse

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type=int, help='epochs for training',default= 2000)
    parse.add_argument('--repetitions', type=int, help='repetition for training',default= 5)
    parse.add_argument('--n_inputs', type=int, help='Dimension of features',default= 20)
    parse.add_argument('--lr', type=float, help='learning rate',default= 0.1)
    parse.add_argument('--num', type=int, help='number of training set', default=500)
    return parse.parse_known_args()[0]


