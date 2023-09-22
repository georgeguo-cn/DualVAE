import numpy as np
import pandas as pd
import argparse
import cornac
from cornac.metrics import Recall, NDCG
from models.recom_dualvaecf import DualVAECF

parser = argparse.ArgumentParser(description="DualVAE model")
parser.add_argument("-d", "--dataset", type=str, default="ML1M",
                    help="name of the dataset, suppose ['ML1M', 'AKindle', 'Yelp']")
parser.add_argument("-k", "--latent_dim", type=int, default=20,
                    help="number of the latent dimensions")
parser.add_argument("-a", "--num_disentangle", type=int, default=5,
                    help="number of the disentangled representation")
parser.add_argument("-en", "--encoder", type=str, default="[40]",
                    help="structure of the user/item encoders")
parser.add_argument("-de", "--decoder", type=str, default="[40]",
                    help="structure of the user/item decoder")
parser.add_argument("-af", "--act_fn", type=str, default="tanh",
                    choices=["sigmoid", "tanh", "relu", "relu6", "elu"],
                    help="non-linear activation function for the encoders")
parser.add_argument("-lh", "--likelihood", type=str, default="pois",
                    choices=["pois", "bern", "gaus", "mult"],
                    help="likelihood function to fit the rating observations")
parser.add_argument("-ne", "--num_epochs", type=int, default=200,
                    help="number of training epochs")
parser.add_argument("-bs", "--batch_size", type=int, default=128,
                    help="batch size for training")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                    help="learning rate for training")
parser.add_argument("-kl", "--beta_kl", type=float, default=1.0,
                    help="beta weighting for the KL divergence")
parser.add_argument("-cl", "--gama_cl", type=float, default=0.01,
                    help="gama weighting for the contrast loss")
parser.add_argument("-tn", "--top_n", type=int, default=[20, 50],
                    help="n cut-off for top-n evaluation")
parser.add_argument("-s", "--random_seed", type=int, default=123,
                    help="random seed value")
parser.add_argument("-v", "--verbose", default=True,
                    help="increase output verbosity")
parser.add_argument("-gpu", "--gpu", type=int, default=0,
                    help="gpu-id")
args = parser.parse_args()
print(args)


def gen_cornac_dataset(data_path, t=-1):
    df = pd.read_csv(data_path)
    if df.shape[1] == 2:
        df.insert(loc=2, column='rating', value=5.0)
    df = df[df.rating > t]
    return df


def load_dataset():
    if args.dataset in ['100K', '1M', '10M']:
        data = cornac.datasets.movielens.load_feedback(variant=args.dataset)
        eval_method = cornac.eval_methods.RatioSplit(data=data, test_size=0.2, rating_threshold=1.0, seed=123, verbose=args.verbose)
    else:
        dataset_dir = f"./data/{args.dataset}/"
        train_data = gen_cornac_dataset(dataset_dir + 'train.csv')
        test_data = gen_cornac_dataset(dataset_dir + 'test.csv')
        # UIR
        eval_method = cornac.eval_methods.BaseMethod.from_splits(
            train_data=train_data.values,
            test_data=test_data.values,
            seed=args.random_seed,
            verbose=args.verbose,
            rating_threshold=1.0,
        )
    return eval_method


if __name__ == "__main__":
    eval_method = load_dataset()

    dualvae = DualVAECF(
        k=args.latent_dim,
        a=args.num_disentangle,
        encoder_structure=eval(args.encoder),
        decoder_structure=eval(args.decoder),
        act_fn=args.act_fn,
        likelihood=args.likelihood,
        n_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta_kl=args.beta_kl,
        gama_cl=args.gama_cl,
        seed=args.random_seed,
        gpu=args.gpu,
        verbose=args.verbose,
    )

    topk_metrics = [Recall(args.top_n), NDCG(args.top_n)]

    cornac.Experiment(
        eval_method=eval_method, models=[dualvae], metrics=topk_metrics, user_based=True
    ).run()