# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.profiler import record_function

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.config import DLLMConfig, UnmaskingStrategy

DLLM_MASKED = consts.DLLM_MASKED
DLLM_UNMASKED = consts.DLLM_UNMASKED
DLLM_CACHED = consts.DLLM_CACHED


class UnmaskingProcessor:

    def __init__(self, dllm_config: DLLMConfig):
        self.dllm_config = dllm_config
        self._history_conf = {}

    def _get_scores(self, logits: torch.Tensor, token_ids: torch.Tensor):
        """Get scores."""
        scores = logits.softmax(dim=-1)
        scores = scores.gather(-1, token_ids.unsqueeze(-1)).flatten()
        return scores

    def _get_denoise_num(self):
        """Get denoise num."""
        block_size = self.dllm_config.block_length
        denoising_steps = self.dllm_config.denoising_steps
        if denoising_steps is None:
            denoising_steps = block_size
        num = block_size // self.dllm_config.denoising_steps
        num = max(1, min(num, block_size))
        return num

    def low_confidence_static(self, logits: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """static."""
        block_size = self.dllm_config.block_length
        topk = self._get_denoise_num()
        scores = self._get_scores(logits, token_ids)
        is_masked = dllm_mask == DLLM_MASKED
        scores = torch.where(is_masked, scores, scores.new_zeros((1, )))

        scores = scores.view(-1, block_size)
        dllm_mask = dllm_mask.view(-1, block_size)
        _, indices = scores.topk(topk, dim=-1)
        dllm_unmasked = dllm_mask.scatter(-1, indices, DLLM_UNMASKED)

        is_masked = is_masked.view_as(dllm_mask)
        dllm_mask = torch.where(is_masked, dllm_unmasked, dllm_mask)
        return dllm_mask.flatten()

    def low_confidence_dynamic(self, logits: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """dynamic."""
        block_size = self.dllm_config.block_length
        threshold = self.dllm_config.confidence_threshold
        scores = self._get_scores(logits, token_ids)
        is_masked = dllm_mask == DLLM_MASKED
        scores = torch.where(is_masked, scores, scores.new_zeros((1, )))

        scores = scores.view(-1, block_size)
        dllm_mask = dllm_mask.view(-1, block_size)
        _, indices = scores.topk(1, dim=-1)
        scores = scores.scatter(-1, indices, threshold)

        is_masked = is_masked.view_as(dllm_mask)
        is_masked &= scores >= threshold
        dllm_mask[is_masked] = DLLM_UNMASKED
        return dllm_mask.flatten()

    def sequential(self, dllm_mask: torch.Tensor):
        """sequential."""
        block_size = self.dllm_config.block_length
        denoise_num = self._get_denoise_num()
        dllm_mask = dllm_mask.view(-1, block_size)
        is_masked = dllm_mask == DLLM_MASKED

        # get indices
        indices = is_masked.int().argmax(dim=1)
        ranges = torch.arange(0, denoise_num, device=indices.device, dtype=indices.dtype)
        indices = indices[:, None] + ranges[None, :]
        indices = indices % block_size

        dllm_unmasked = dllm_mask.clone()
        dllm_unmasked = dllm_unmasked.scatter(-1, indices, DLLM_UNMASKED)
        dllm_mask = torch.where(is_masked, dllm_unmasked, dllm_mask)

        return dllm_mask.flatten()

    def bounded_adaptive_confidence_decoding(self, logits, token_ids, dllm_mask):
        """
        τ_0 = clamp(max(conf), lower_bound, upper_bound)
        τ_t = clamp(mean(history), lower_bound, upper_bound)
        """
        block_size = self.dllm_config.block_length
        upper_bound = getattr(self.dllm_config, 'confidence_upper_threshold', 0.9)
        lower_bound = getattr(self.dllm_config, 'confidence_lower_threshold', 0.6)
        
        probs = logits.softmax(dim=-1)
        conf = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

        is_masked = dllm_mask == DLLM_MASKED
        conf = torch.where(is_masked, conf, torch.zeros_like(conf))

        conf = conf.view(-1, block_size)
        dllm_mask_original = dllm_mask.view(-1, block_size)
        dllm_mask = dllm_mask_original.clone()
        is_masked = is_masked.view_as(dllm_mask)

        batch_size = dllm_mask.shape[0]
        new_mask = dllm_mask.clone()

        for batch_idx in range(batch_size):
            if is_masked[batch_idx].sum() == 0:
                continue

            block_conf = conf[batch_idx]
            block_masked = is_masked[batch_idx]
            
            if batch_idx not in self._history_conf or self._history_conf[batch_idx] is None:
                # τ_0 = clamp(max(conf), lower, upper)
                masked_conf = block_conf[block_masked]
                if masked_conf.numel() > 0:
                    max_conf = masked_conf.max().item()
                    tau = max(lower_bound, min(upper_bound, max_conf))
                else:
                    tau = lower_bound
            else:
                # τ_t = clamp(mean(history), lower, upper)
                raw_mean = self._history_conf[batch_idx].mean().item()
                tau = max(lower_bound, min(upper_bound, raw_mean))

            eligible = block_masked & (block_conf >= tau)
            num_eligible = eligible.sum().item()

            # progress guarantee
            if num_eligible == 0:
                _, idx = block_conf.topk(1)
                eligible[idx] = True

            # unmask
            new_mask[batch_idx][eligible] = DLLM_UNMASKED

            newly_unmasked = (dllm_mask[batch_idx] == DLLM_MASKED) & (new_mask[batch_idx] == DLLM_UNMASKED)
            new_conf = block_conf[newly_unmasked]

            if new_conf.numel() > 0:
                if batch_idx not in self._history_conf or self._history_conf[batch_idx] is None:
                    self._history_conf[batch_idx] = new_conf.detach()
                else:
                    self._history_conf[batch_idx] = torch.cat(
                        [self._history_conf[batch_idx], new_conf.detach()], dim=0
                    )

            if (new_mask[batch_idx] == DLLM_MASKED).sum() == 0:
                if batch_idx in self._history_conf and self._history_conf[batch_idx] is not None:
                    self._history_conf[batch_idx] = None

        return new_mask.flatten()

    @record_function('unmasking')
    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """call."""
        strategy = self.dllm_config.unmasking_strategy
        if strategy is None:
            return dllm_mask

        # reshape to [num_blocks, block_size]
        block_size = self.dllm_config.block_length
        dllm_mask = dllm_mask.unflatten(0, (-1, block_size))

        is_same = (dllm_mask == dllm_mask[:, :1]).all(dim=1)
        first_mask = dllm_mask[:, 0]

        # unmasked to cache
        is_block_unmasked = is_same & (first_mask == DLLM_UNMASKED)
        dllm_mask[is_block_unmasked] = DLLM_CACHED

        dllm_mask = dllm_mask.flatten()
        token_ids = torch.where(dllm_mask != DLLM_MASKED, input_ids, token_ids)
        if strategy == UnmaskingStrategy.LOW_CONFIDENCE_STATIC:
            dllm_mask = self.low_confidence_static(logits, token_ids, dllm_mask)
        elif strategy == UnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC:
            dllm_mask = self.low_confidence_dynamic(logits, token_ids, dllm_mask)
        elif strategy == UnmaskingStrategy.SEQUENTIAL:
            dllm_mask = self.sequential(dllm_mask)
        elif strategy == UnmaskingStrategy.BACD:
            dllm_mask = self.bounded_adaptive_confidence_decoding(logits, token_ids, dllm_mask)
        else:
            raise RuntimeError(f'strategy {strategy} not supported.')

        return dllm_mask, token_ids
