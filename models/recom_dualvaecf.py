import torch
import numpy as np
from cornac.models.recommender import Recommender
from cornac.utils.common import scale
from cornac.exception import ScoreException
from models.dualvae import DualVAE, learn


class DualVAECF(Recommender):
    def __init__(
        self,
        name="DualVAECF",
        k=20,
        a=5,
        encoder_structure=[20],
        decoder_structure=[20],
        act_fn="tanh",
        likelihood="pois",
        n_epochs=100,
        batch_size=100,
        learning_rate=0.001,
        beta_kl=1.0,
        gama_cl=0.01,
        trainable=True,
        verbose=False,
        seed=None,
        gpu=-1,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.a = a
        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure
        self.act_fn = act_fn
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.beta_kl = beta_kl
        self.gama_cl = gama_cl
        self.seed = seed
        self.gpu = gpu

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)

        self.device = (torch.device("cuda:" + str(self.gpu)) if (self.gpu >= 0 and torch.cuda.is_available()) else torch.device("cpu") )

        if self.trainable:

            if self.seed is not None:
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)

            if not hasattr(self, "dualvaecf"):
                num_items = train_set.matrix.shape[1]
                num_users = train_set.matrix.shape[0]
                self.dualvae = DualVAE(
                    k=self.k,
                    a=self.a,
                    user_encoder_structure=[num_items] + self.encoder_structure,
                    item_encoder_structure=[num_users] + self.encoder_structure,
                    user_decoder_structure=[self.k] + self.decoder_structure,
                    item_decoder_structure=[self.k] + self.decoder_structure,
                    act_fn=self.act_fn,
                    likelihood=self.likelihood,
                ).to(self.device)

            learn(
                self.dualvae,
                self.train_set,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                learn_rate=self.learning_rate,
                beta_kl=self.beta_kl,
                gama_cl=self.gama_cl,
                verbose=self.verbose,
                device=self.device,
            )

        elif self.verbose:
            print("%s is trained already (trainable = False)" % (self.name))

        return self

    def score(self, user_idx, item_idx=None):

        x = self.train_set.matrix.copy()
        x.data = np.ones_like(x.data)  # Binarize data
        tx = x.transpose()

        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException("Can't make score prediction for (user_id=%d)" % user_idx)

            # Reconstructed batch
            known_item_scores = None
            theta = self.dualvae.mu_theta[user_idx]
            beta = self.dualvae.mu_beta
            aspect_prob = self.dualvae.aspect_probability
            for a in range(self.a):
                theta_a = theta[a].view(1, -1)
                beta_a = beta[:, a, :].squeeze()
                aspect_a = aspect_prob[:, a].reshape((1, -1))
                scores_a = self.dualvae.decode_user(theta_a, beta_a)
                scores_a = scores_a * aspect_a
                known_item_scores = scores_a if known_item_scores is None else (known_item_scores + scores_a)
            known_item_scores = known_item_scores.detach().cpu().numpy().ravel()
            train_mat = self.train_set.csr_matrix
            csr_row = train_mat.getrow(user_idx)
            pos_items = [item_idx for (item_idx, rating) in zip(csr_row.indices, csr_row.data)]
            known_item_scores[pos_items] = 0.
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            pred = None
            theta = self.dualvae.mu_theta[user_idx]
            beta = self.dualvae.mu_beta[item_idx]
            aspect_prob = self.dualvae.aspect_probability
            for a in range(self.a):
                theta_a = theta[a, :]
                beta_a = beta[:, a, :].squeeze()
                aspect_a = aspect_prob[item_idx, a].reshape((1, -1))
                scores_a = self.dualvae.decode_user(theta_a, beta_a)
                scores_a = scores_a * aspect_a
                pred = scores_a if pred is None else (pred + scores_a)
            pred = torch.sigmoid(pred).cpu().numpy().ravel()

            pred = scale(pred, self.train_set.min_rating, self.train_set.max_rating, 0.0, 1.0)

            return pred