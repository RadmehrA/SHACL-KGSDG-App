# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class GraphVAE(nn.Module):
#     """
#     GraphVAE operating on triple-level feature tensors.
#     Input:  [num_triples, feature_dim]
#     Output: reconstructed triple tensor
#     """

#     def __init__(self, input_dim, hidden_dim=256, latent_dim=64):
#         super(GraphVAE, self).__init__()

#         # ---------- Encoder ----------
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU()
#         )

#         self.mu_layer = nn.Linear(hidden_dim, latent_dim)
#         self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

#         # ---------- Decoder ----------
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim)
#         )

#     # ---------- Reparameterization ----------
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     # ---------- Forward ----------
#     def forward(self, x):

#         h = self.encoder(x)

#         mu = self.mu_layer(h)
#         logvar = self.logvar_layer(h)

#         z = self.reparameterize(mu, logvar)

#         recon = self.decoder(z)

#         return recon, mu, logvar



import torch
import torch.nn as nn

class GraphVAE(nn.Module):

    def __init__(self, num_subjects, num_predicates, num_objects, latent_dim=32):
        super().__init__()

        self.subject_emb = nn.Embedding(num_subjects, 32)
        self.predicate_emb = nn.Embedding(num_predicates, 32)

        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_objects)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, s_idx, p_idx):
        s_emb = self.subject_emb(s_idx)
        p_emb = self.predicate_emb(p_idx)

        x = torch.cat([s_emb, p_emb], dim=1)
        h = self.encoder(x)

        mu = self.mu(h)
        logvar = self.logvar(h)

        z = self.reparameterize(mu, logvar)
        output = self.decoder(z)

        return output, mu, logvar