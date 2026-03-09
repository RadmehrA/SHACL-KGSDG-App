import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, beta=0.001):
    """
    Standard VAE loss
    """

    recon_loss = F.mse_loss(recon_x, x)

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ---------- Optional Constraint Loss ----------
def constraint_loss(pred, constraints):
    """
    Placeholder for future constraint penalties.
    """
    return torch.tensor(0.0, device=pred.device)