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