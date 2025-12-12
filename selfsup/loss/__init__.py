"""The lightly.loss package provides loss functions for self-supervised learning. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from selfsup.loss.barlow_twins_loss import BarlowTwinsLoss
from selfsup.loss.dcl_loss import DCLLoss, DCLWLoss
from selfsup.loss.dino_loss import DINOLoss
from selfsup.loss.negative_cosine_similarity import NegativeCosineSimilarity
from selfsup.loss.ntx_ent_loss import NTXentLoss
from selfsup.loss.swav_loss import SwaVLoss
from selfsup.loss.sym_neg_cos_sim_loss import SymNegCosineSimilarityLoss
