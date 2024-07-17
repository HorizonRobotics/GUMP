from typing import List

import torch
import torch.nn.functional as F

from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType


def calculate_perplexity(logits, targets, token_mask):
    """
    Calculates perplexity based on logits and true targets.

    Args:
    - logits (torch.Tensor): Model logits, shape (batch, seq_len, vocab_size)
    - targets (torch.Tensor): True labels, shape (batch, seq_len)
    - token_mask (torch.Tensor): Mask indicating which tokens are real and which are padding, shape (batch, seq_len)

    Returns:
    - perplexity (float)
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    # For each position in each sequence in the batch, we want to select the probability
    # corresponding to the correct label. This gives us the log probability of
    # the true label.
    log_probs = torch.gather(
        probs.log(), dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Use the mask to zero out log_probs from padding tokens
    masked_log_probs = log_probs * token_mask

    # Calculate total negative log probability for actual tokens
    total_nll = -masked_log_probs.sum()

    # Calculate mean negative log probability
    num_actual_tokens = token_mask.sum()
    mean_nll = total_nll / num_actual_tokens

    # Calculate perplexity
    perplexity = torch.exp(mean_nll).item()

    return perplexity


class PerplexityMetric(AbstractTrainingMetric):
    """
    Metric representing the IoU between occupancy prediction adn occupancy target.
    """

    def __init__(self, name: str = 'perplexity') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType,
                targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        logits = predictions['logits'].data
        token_mask = predictions['token_mask'].data
        target_tokens = predictions['target_tokens'].data
        perplexity = calculate_perplexity(logits, target_tokens, token_mask)
        return perplexity
