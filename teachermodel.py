from copy import deepcopy

import torch
from timm.models.layers import DropPath
from torch import nn
from torch.nn.modules.dropout import _DropoutNd


class EMATeacher(nn.Module):
    """
    Exponential Moving Average (EMA) Teacher Network.

    This module maintains an exponential moving average copy of the student
    model and is used to generate pseudo-labels for self-training in
    unsupervised / semi-supervised domain adaptation.

    This implementation follows the EMA teacher paradigm used in:
    MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation
    (CVPR 2023).
    """

    def __init__(self, model, alpha, pseudo_label_weight):
        """
        Args:
            model (nn.Module): Student network whose parameters are tracked by EMA.
            alpha (float): EMA decay factor.
            pseudo_label_weight (str or None):
                - 'prob': use prediction confidence as pseudo-label weight
                - None: uniform weight
        """
        super(EMATeacher, self).__init__()

        # Create a deep copy of the student model as the EMA teacher
        self.ema_model = deepcopy(model)

        # EMA decay coefficient
        self.alpha = alpha

        # Strategy for pseudo-label weighting
        self.pseudo_label_weight = pseudo_label_weight
        if self.pseudo_label_weight == 'None':
            self.pseudo_label_weight = None

    # ------------------------------------------------------------------
    # EMA initialization
    # ------------------------------------------------------------------
    def _init_ema_weights(self, model):
        """
        Initialize EMA model weights by copying parameters
        from the student model.
        """
        for param in self.ema_model.parameters():
            param.detach_()

        mp = list(model.parameters())
        mcp = list(self.ema_model.parameters())

        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------
    def _update_ema(self, model, iter):
        """
        Update EMA weights using exponential moving average.

        Args:
            model (nn.Module): Current student model.
            iter (int): Current training iteration.
        """
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)

        for ema_param, param in zip(self.ema_model.parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    # ------------------------------------------------------------------
    # Public EMA update interface
    # ------------------------------------------------------------------
    def update_weights(self, model, iter):
        """
        Initialize or update EMA teacher weights.

        Args:
            model (nn.Module): Student model.
            iter (int): Training iteration index.
        """
        # Initialize EMA model at the first iteration
        if iter == 0:
            self._init_ema_weights(model)

        # Update EMA model after initialization
        if iter > 0:
            self._update_ema(model, iter)

    # ------------------------------------------------------------------
    # Forward: pseudo-label generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, target_img, mask=False):
        """
        Generate pseudo-labels and corresponding weights for target samples.

        Args:
            target_img (Tensor): Target domain input samples.
            mask (bool): Reserved flag (not used in this implementation).

        Returns:
            pseudo_label (Tensor): Predicted pseudo-labels.
            pseudo_weight (Tensor): Weights for pseudo-label loss.
        """

        # Disable stochastic layers (Dropout, DropPath) in EMA model
        for m in self.ema_model.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        # Forward pass through EMA teacher
        feat, logits, _, _ = self.ema_model(target_img)

        # Compute softmax probabilities
        ema_softmax = torch.softmax(logits.detach(), dim=1)

        # Select pseudo-labels and confidence scores
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)

        # Determine pseudo-label weight
        if self.pseudo_label_weight is None:
            pseudo_weight = torch.tensor(1., device=logits.device)
        elif self.pseudo_label_weight == 'prob':
            pseudo_weight = pseudo_prob
        else:
            raise NotImplementedError(self.pseudo_label_weight)

        return pseudo_label, pseudo_weight
