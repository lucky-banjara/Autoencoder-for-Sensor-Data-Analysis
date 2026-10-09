import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim=100, bottleneck_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


if __name__ == "__main__":
    model = Autoencoder()

    # Test with 5 windows, each containing 100 samples
    sample_input = torch.randn(5, 100)
    sample_output = model(sample_input)

    print("Autoencoder architecture:")
    print(model)
    print("\nInput shape:", tuple(sample_input.shape))
    print("Output shape:", tuple(sample_output.shape))
    print("Trainable parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))