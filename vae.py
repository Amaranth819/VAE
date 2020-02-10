import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, nz = 32):
        super(VAE, self).__init__()
        self.nz = nz

        self.encoder_fc = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(True)
        )
        self.mu_fc = nn.Linear(256, nz)
        self.logvar_fc = nn.Linear(256, nz)
        self.decoder_fc = nn.Sequential(
            nn.Linear(nz, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device = std.get_device(), requires_grad = True)
        return mu + eps * std

    def encode(self, x):
        x = self.encoder_fc(x)
        return self.mu_fc(x), self.logvar_fc(x)

    def decode(self, z):
        return self.decoder_fc(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

def loss_func(recon_x, x, mu, logvar):
    recon_loss = torch.sum(nn.MSELoss(reduction = 'none')(recon_x, x), dim = [1]) / x.size(0)
    kl_divergence = torch.sum((mu.pow(2) + logvar.exp() - 1 - logvar) * 0.5, dim = [1]) / x.size(0)
    return (recon_loss + kl_divergence).mean()

def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.1, 0.1)
            if m.bias is not None:
                m.bias.data.zero_()