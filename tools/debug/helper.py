import torch
import torch.nn.functional as F


def jensen_shannon_divergence(output_logits, my_output_logits):
    # Convert logits to probabilities using softmax
    p = F.softmax(output_logits, dim=-1)
    q = F.softmax(my_output_logits, dim=-1)

    # Calculate the average distribution
    m = 0.5 * (p + q)

    # Compute KL divergence
    kl_p_m = F.kl_div(m.log(), p, reduction='batchmean', log_target=False)
    kl_q_m = F.kl_div(m.log(), q, reduction='batchmean', log_target=False)

    # Jensen-Shannon divergence
    jsd = 0.5 * (kl_p_m + kl_q_m)

    return jsd