import torch
import torch.nn as nn


# ============================================================
# GNET: Feature Extractor + Classifier + Projection Head
# ============================================================
class GNET(nn.Module):
    """
    GNET: Backbone network used in AFLNet.

    This network serves three purposes simultaneously:
    1) Feature extractor for domain adaptation
    2) Classifier for modulation recognition
    3) Projection head for contrastive (similarity-based) learning

    Output:
        - x1: flattened deep features
        - y: classification logits
        - x2: reconstructed signal (decoder output)
        - projection: low-dimensional projected features
    """
    def __init__(self):
        super(GNET, self).__init__()

        # ----------------------------------------------------
        # Encoder: 1D CNN feature extractor (IQ signals)
        # ----------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )

        # ----------------------------------------------------
        # Decoder: signal reconstruction branch (auxiliary)
        # ----------------------------------------------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(in_channels=32, out_channels=2, kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(2)
        )

        # ----------------------------------------------------
        # Additional convolution layers before RNN
        # ----------------------------------------------------
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(in_channels=64, out_channels=50, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(50),
            nn.LeakyReLU()
        )

        # ----------------------------------------------------
        # Temporal modeling with LSTM
        # ----------------------------------------------------
        self.Rnn = nn.LSTM(128, 128, num_layers=2, batch_first=True)

        # ----------------------------------------------------
        # Classifier head
        # ----------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(in_features=6400, out_features=2048),
            nn.Dropout(0.6),
            nn.LeakyReLU(),

            nn.Linear(in_features=2048, out_features=1024),
            nn.Dropout(0.6),
            nn.LeakyReLU(),

            nn.Linear(in_features=1024, out_features=256),
            nn.Dropout(0.6),
            nn.LeakyReLU(),

            nn.Linear(in_features=256, out_features=11)
        )

        # ----------------------------------------------------
        # Projection head (for contrastive learning)
        # ----------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(in_features=6400, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=128)
        )

        # ----------------------------------------------------
        # Weight initialization
        # ----------------------------------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input IQ signal, shape [B, 2, L]

        Returns:
            x1 (Tensor): Flattened feature representation
            y (Tensor): Classification logits
            x2 (Tensor): Reconstructed signal
            projection (Tensor): Projected features for contrastive learning
        """
        x1 = self.encoder(x)
        x2 = self.decoder(x1)

        x = self.conv(x1)
        x, _ = self.Rnn(x)

        x1 = x.contiguous().view(x.size(0), -1)
        y = self.fc(x1)
        projection = self.head(x1)

        return x1, y, x2, projection


# ============================================================
# DNET: Domain Discriminator (without GRL)
# ============================================================
class DNET(nn.Module):
    """
    Domain discriminator used for domain classification
    (source vs target).
    """
    def __init__(self):
        super(DNET, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=6400, out_features=2048),
            nn.ReLU(),

            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x1 = self.fc(x)
        return x1


# ============================================================
# Gradient Reversal Layer (GRL)
# ============================================================
from torch.autograd import Function

class ReverseLayerF(Function):
    """
    Gradient Reversal Layer used in DANN.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# ============================================================
# DNET_DANN: Domain Discriminator with GRL
# ============================================================
class DNET_DANN(nn.Module):
    """
    Domain discriminator with Gradient Reversal Layer
    for adversarial domain adaptation (DANN).
    """
    def __init__(self):
        super(DNET_DANN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=6400, out_features=2048),
            nn.ReLU(),

            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, feature, alpha):
        reversefeature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.fc(reversefeature)
        return domain_output



# ============================================================
# Debug / Test
# ============================================================
if __name__ == '__main__':
    model = DNET()
    print(model)

    input = torch.randn(32, 6400)
    out = model(input)
    print(out.shape)
