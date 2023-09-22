import itertools as it
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange


EPS = 1e-10

ACT = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
}


class DualVAE(nn.Module):
    def __init__(
        self,
        k,
        a,
        user_encoder_structure,
        item_encoder_structure,
        user_decoder_structure,
        item_decoder_structure,
        act_fn,
        likelihood,
    ):
        super(DualVAE, self).__init__()

        self.a = a
        self.mu_theta = torch.zeros((item_encoder_structure[0], a, k))  # n_users*t*k
        self.mu_beta = torch.zeros((user_encoder_structure[0], a, k))  # n_items*t*k

        self.user_preferences = nn.Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(a, k), a=np.sqrt(5)))
        self.item_topics = nn.Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(a, k), a=np.sqrt(5)))

        self.theta = torch.randn(item_encoder_structure[0], a, k) * 0.01
        self.beta = torch.randn(user_encoder_structure[0], a, k) * 0.01

        self.likelihood = likelihood
        self.act_fn = ACT.get(act_fn, None)
        if self.act_fn is None:
            raise ValueError("Supported act_fn: {}".format(ACT.keys()))

        # User Encoder
        self.user_encoder = nn.Sequential()
        for i in range(len(user_encoder_structure) - 1):
            self.user_encoder.add_module(
                "fc{}".format(i),
                nn.Linear(user_encoder_structure[i], user_encoder_structure[i + 1]),
            )
            self.user_encoder.add_module("act{}".format(i), self.act_fn)
        self.user_mu = nn.Linear(user_encoder_structure[-1], k)  # mu
        self.user_std = nn.Linear(user_encoder_structure[-1], k)

        # Item Encoder
        self.item_encoder = nn.Sequential()
        for i in range(len(item_encoder_structure) - 1):
            self.item_encoder.add_module(
                "fc{}".format(i),
                nn.Linear(item_encoder_structure[i], item_encoder_structure[i+1]),
            )
            self.item_encoder.add_module("act{}".format(i), self.act_fn)
        self.item_mu = nn.Linear(item_encoder_structure[-1], k)  # mu
        self.item_std = nn.Linear(item_encoder_structure[-1], k)

        # User Decoder
        self.user_decoder = nn.Sequential()
        for i in range(len(user_decoder_structure)-1):
            self.user_decoder.add_module(
                "fc_out{}".format(i),
                nn.Linear(user_decoder_structure[i], user_decoder_structure[i+1])
            )
            self.user_decoder.add_module("act_out{}".format(i), self.act_fn)

        # Item Decoder
        self.item_decoder = nn.Sequential()
        for i in range(len(item_decoder_structure)-1):
            self.item_decoder.add_module(
                "fc_out{}".format(i),
                nn.Linear(item_decoder_structure[i], item_decoder_structure[i+1])
            )
            self.item_decoder.add_module("act_out{}".format(i), self.act_fn)

        self.aspect_probability = None

    def to(self, device):
        self.beta = self.beta.to(device=device)
        self.theta = self.theta.to(device=device)
        self.mu_beta = self.mu_beta.to(device=device)
        self.mu_theta = self.mu_theta.to(device=device)
        return super(DualVAE, self).to(device)

    def encode_user(self, x):
        h = self.user_encoder(x)
        return self.user_mu(h), torch.sigmoid(self.user_std(h))

    def encode_item(self, x):
        h = self.item_encoder(x)
        return self.item_mu(h), torch.sigmoid(self.item_std(h))

    def decode_user(self, theta, beta):
        theta_hidden = self.user_decoder(theta)
        beta_hidden = self.item_decoder(beta)
        h_hidden = theta_hidden.mm(beta_hidden.t())
        h_hidden = nn.Tanh()(h_hidden)
        h = theta.mm(beta.t())
        if self.likelihood == 'mult':
            return h + h_hidden
        return torch.sigmoid(h + h_hidden)

    def decode_item(self, theta, beta):
        theta_hidden = self.user_decoder(theta)
        beta_hidden = self.item_decoder(beta)
        h_hidden = beta_hidden.mm(theta_hidden.t())
        h_hidden = nn.Tanh()(h_hidden)
        h = beta.mm(theta.t())
        if self.likelihood == 'mult':
            return h + h_hidden
        return torch.sigmoid(h + h_hidden)

    def reparameterize(self, mu, std):
        eps = torch.randn_like(mu)
        return mu + eps * std

    def contrast_loss(self, x, x_):
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        x_ = torch.nn.functional.normalize(x_, p=2, dim=-1)
        # positive
        pos_score = torch.sum(torch.mul(x_, x), dim=-1)
        pos_score = torch.exp(pos_score / 0.2)
        # different aspects for a user
        acl_score = torch.bmm(x_, x.transpose(1, 2))
        acl_score = torch.sum(torch.exp(acl_score / 0.2), dim=-1)
        # different users for an aspect
        ncl_score = torch.bmm(x_.transpose(0, 1), x.transpose(0, 1).transpose(1, 2))
        ncl_score = torch.sum(torch.exp(ncl_score.transpose(0, 1) / 0.2), dim=-1)
        # cl
        neg_score = acl_score + ncl_score
        info_nec_loss = torch.log(pos_score / neg_score)
        info_nec_loss = -torch.mean(torch.sum(info_nec_loss, dim=-1))
        return info_nec_loss

    def forward(self, x, user=True, beta=None, theta=None):
        if user:
            # soft-attention
            aspect_prob = torch.sum(torch.mul(beta, self.item_topics), dim=-1)
            aspect_prob = torch.softmax(aspect_prob, dim=1)

            z_u_list = []
            nei_u_list = []
            z_u_mu_list = []
            probs = None
            kl = None
            for a in range(self.a):
                aspect_a = aspect_prob[:, a].reshape((1, -1))
                # encoder
                mu, std = self.encode_user(x * aspect_a)
                # KL term
                kl_a = -0.5 * (1 + 2.0 * torch.log(std) - mu.pow(2) - std.pow(2))
                kl_a = torch.mean(torch.sum(kl_a, dim=-1))
                kl = (kl_a if (kl is None) else (kl + kl_a))
                theta = self.reparameterize(mu, std)
                # decoder
                probs_a = self.decode_user(theta, beta[:, a, :].squeeze())
                probs_a = probs_a * aspect_a
                probs = probs_a if probs is None else (probs + probs_a)
                z_u_list.append(theta)
                z_u_mu_list.append(mu)
                # neighborhood-based representation
                nei_u_list.append(torch.mm(probs_a, beta[:, a, :].squeeze()))
            z_u_list = torch.stack(z_u_list).transpose(0, 1)
            z_u_mu_list = torch.stack(z_u_mu_list).transpose(0, 1)
            nei_u_list = torch.stack(nei_u_list).transpose(0, 1)
            # KL
            kl = kl/self.a
            # CL
            cl = self.contrast_loss(z_u_list, nei_u_list)
            if self.likelihood == 'mult':
                probs = torch.softmax(probs, dim=1)
            return z_u_list, z_u_mu_list, probs, kl, cl
        # Item
        else:
            # soft-attention
            prefer_prob = torch.sum(torch.mul(theta, self.user_preferences), dim=-1)
            prefer_prob = torch.softmax(prefer_prob, dim=1)

            z_i_list = []
            nei_i_list = []
            z_i_mu_list = []
            probs = None
            kl = None
            for a in range(self.a):
                prefer_a = prefer_prob[:, a].reshape((1, -1))
                # encoder
                mu, std = self.encode_item(x * prefer_a)
                # KL term
                kl_a = -0.5 * (1 + 2.0 * torch.log(std) - mu.pow(2) - std.pow(2))
                kl_a = torch.mean(torch.sum(kl_a, dim=-1))
                kl = (kl_a if (kl is None) else (kl + kl_a))
                beta = self.reparameterize(mu, std)
                # decoder
                probs_a = self.decode_item(theta[:, a, :].squeeze(), beta)
                probs_a = probs_a * prefer_a
                probs = probs_a if probs is None else (probs + probs_a)
                z_i_list.append(beta)
                z_i_mu_list.append(mu)
                # neighborhood-based representation
                nei_i_list.append(torch.mm(probs_a, theta[:, a, :].squeeze()))
            z_i_list = torch.stack(z_i_list).transpose(0, 1)
            z_i_mu_list = torch.stack(z_i_mu_list).transpose(0, 1)
            nei_i_list = torch.stack(nei_i_list).transpose(0, 1)
            # KL
            kl = kl / self.a
            # CL
            cl = self.contrast_loss(z_i_list, nei_i_list)
            if self.likelihood == 'mult':
                probs = torch.softmax(probs, dim=1)
            return z_i_list, z_i_mu_list, probs, kl, cl

    def loss(self, x, x_, kl, kl_beta, cl, cl_gama):
        # Likelihood
        ll_choices = {
            "bern": x * torch.log(x_ + EPS) + (1 - x) * torch.log(1 - x_ + EPS),
            "gaus": -(x - x_) ** 2,
            "pois": x * torch.log(x_ + EPS) - x_,
            "mult": torch.log(x_ + EPS) * x
        }
        ll = ll_choices.get(self.likelihood, None)
        if ll is None:
            raise ValueError("Supported likelihoods: {}".format(ll_choices.keys()))
        ll = torch.mean(torch.sum(ll, dim=-1))
        return kl_beta * kl - ll + cl_gama * cl


def learn(
    dualvae,
    train_set,
    n_epochs,
    batch_size,
    learn_rate,
    beta_kl,
    gama_cl,
    verbose,
    device=torch.device("cpu"),
    dtype=torch.float32,
):
    user_params = it.chain(
        dualvae.item_topics.data,
        dualvae.user_encoder.parameters(),
        dualvae.user_mu.parameters(),
        dualvae.user_std.parameters(),
        dualvae.user_decoder.parameters(),
    )

    item_params = it.chain(
        dualvae.user_preferences.data,
        dualvae.item_encoder.parameters(),
        dualvae.item_mu.parameters(),
        dualvae.item_std.parameters(),
        dualvae.item_decoder.parameters(),
    )

    u_optimizer = torch.optim.Adam(params=user_params, lr=learn_rate)
    i_optimizer = torch.optim.Adam(params=item_params, lr=learn_rate)

    x = train_set.matrix.copy()
    x.data = np.ones_like(x.data)  # Binarize data
    tx = x.transpose()

    progress_bar = trange(1, n_epochs + 1, disable=not verbose)
    for _ in progress_bar:
        # item side
        i_sum_loss = 0.0
        i_count = 0
        for i_ids in train_set.item_iter(batch_size, shuffle=False):
            i_batch = tx[i_ids, :]
            i_batch = i_batch.A
            i_batch = torch.tensor(i_batch, dtype=dtype, device=device)

            # Reconstructed batch
            z_i_list, z_i_mu_list, probs, kl, cl = dualvae(i_batch, user=False, theta=dualvae.theta)

            i_loss = dualvae.loss(i_batch, probs, kl, beta_kl, cl, gama_cl)
            i_optimizer.zero_grad()
            i_loss.backward()
            i_optimizer.step()

            i_sum_loss += i_loss.data.item()
            i_count += len(i_batch)

            z_i_list, z_i_mu_list, _, _, _ = dualvae(i_batch, user=False, theta=dualvae.theta)

            dualvae.beta.data[i_ids] = z_i_list.data

        # user side
        u_sum_loss = 0.0
        u_count = 0
        for u_ids in train_set.user_iter(batch_size, shuffle=False):
            u_batch = x[u_ids, :]
            u_batch = u_batch.A
            u_batch = torch.tensor(u_batch, dtype=dtype, device=device)

            # Reconstructed batch
            z_u_list, z_u_mu_list, probs, kl, cl = dualvae(u_batch, user=True, beta=dualvae.beta)

            u_loss = dualvae.loss(u_batch, probs, kl, beta_kl, cl, gama_cl)
            u_optimizer.zero_grad()
            u_loss.backward()
            u_optimizer.step()

            u_sum_loss += u_loss.data.item()
            u_count += len(u_batch)

            z_u_list, z_u_mu_list, _, _, _ = dualvae(u_batch, user=True, beta=dualvae.beta)
            dualvae.theta.data[u_ids] = z_u_list.data

            progress_bar.set_postfix(
                loss_i=(i_sum_loss / i_count), loss_u=(u_sum_loss / (u_count))
            )
    # infer mu_beta
    for i_ids in train_set.item_iter(batch_size, shuffle=False):
        i_batch = tx[i_ids, :]
        i_batch = i_batch.A
        i_batch = torch.tensor(i_batch, dtype=dtype, device=device)

        z_i_list, z_i_mu_list, _, _, _ = dualvae(i_batch, user=False, theta=dualvae.theta)
        dualvae.mu_beta.data[i_ids] = z_i_mu_list.data

    # infer mu_theta
    for u_ids in train_set.user_iter(batch_size, shuffle=False):
        u_batch = x[u_ids, :]
        u_batch = u_batch.A
        u_batch = torch.tensor(u_batch, dtype=dtype, device=device)

        z_u_list, z_u_mu_list, _, _, _ = dualvae(u_batch, user=True, beta=dualvae.beta)
        dualvae.mu_theta.data[u_ids] = z_u_mu_list.data

    aspect_prob = torch.sum(torch.mul(dualvae.mu_beta, dualvae.item_topics), dim=-1)
    dualvae.aspect_probability = torch.softmax(aspect_prob, dim=1)

    return dualvae