# Copyright (C) 2025 Embedl AB

"""Flash head implementation for faster efficient language model head."""

import json
import os
from typing import Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn


def _resolve_asset(model_or_dir: str, relative_path: str) -> str:
    if os.path.isdir(model_or_dir):
        p = os.path.join(model_or_dir, relative_path)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing local asset: {p}")
        return p
    return hf_hub_download(repo_id=model_or_dir, filename=relative_path)


def _get_centroids(
    lm_head: nn.Linear,
    model_or_dir: str,
    cache_dir: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    original_shape = lm_head.weight.shape  # (vocab, hidden)

    cache_file_rel = os.path.join(cache_dir, "clustering_cache.safetensors")
    meta_file_rel = os.path.join(cache_dir, "clustering_config.json")

    cache_file = _resolve_asset(model_or_dir, cache_file_rel)
    meta_file = _resolve_asset(model_or_dir, meta_file_rel)

    try:
        with open(meta_file, encoding="utf-8") as f:
            metadata = json.load(f)

        if metadata.get("format") not in (None, "safetensors"):
            raise ValueError(
                f"Expected safetensors format, found: {metadata.get('format')}"
            )

        if metadata.get("vocab_size") != original_shape[0]:
            raise ValueError(
                f"Cache vocab_size {metadata.get('vocab_size')} != expected {original_shape[0]}"
            )
        if metadata.get("hidden_size") != original_shape[1]:
            raise ValueError(
                f"Cache hidden_size {metadata.get('hidden_size')} != expected {original_shape[1]}"
            )

        tensors = load_file(cache_file)

        if "centroids" not in tensors or "cluster_assignments" not in tensors:
            raise KeyError(
                f"Cache missing required tensors. Found keys: {list(tensors.keys())}"
            )

        centroids = tensors["centroids"]
        cluster_assignments = tensors["cluster_assignments"]

        if (
            cluster_assignments.ndim != 1
            or cluster_assignments.shape[0] != original_shape[0]
        ):
            raise ValueError(
                f"cluster_assignments shape {tuple(cluster_assignments.shape)}; expected ({original_shape[0]},)"
            )

        device = lm_head.weight.device
        dtype = lm_head.weight.dtype
        centroids = centroids.to(device=device, dtype=dtype)
        cluster_assignments = cluster_assignments.to(device=device)

        return centroids, cluster_assignments

    except Exception as e:
        raise ValueError(f"Error loading cache: {e}") from e


def get_flash_head_parameters(
    lm_head: nn.Module,
    cache_dir: str,
    model_or_dir: str,
    n_clusters: Optional[int] = None,
) -> Tuple[torch.Tensor]:
    """
    Get parameters for the FlashHead layer.

    :param lm_head:
        The language model head to replace.
    :param model_or_dir:
        The model directory.
    :param cache_dir:
        Directory to flash head artifacts.
    :param n_clusters:
        The number of clusters.
    :return:
        The centroids and clustering assignments to use.
    """
    if n_clusters is None:
        n_clusters = int(lm_head.weight.shape[0] / 16)

    centroids, cluster_assignments = _get_centroids(
        lm_head=lm_head,
        model_or_dir=model_or_dir,
        cache_dir=cache_dir,
    )
    total_clusters = n_clusters
    original_shape = lm_head.weight.shape
    cluster_to_vocab_maps = [
        torch.where(cluster_assignments == i)[0] for i in range(total_clusters)
    ]

    combined_centroids = torch.zeros(
        (original_shape[1], total_clusters),
        device=lm_head.weight.device,
        dtype=lm_head.weight.dtype,
    )
    centroids_reshaped = centroids.squeeze(0).squeeze(0)
    combined_centroids[:, :n_clusters] = centroids_reshaped
    max_len = max(m.shape[0] for m in cluster_to_vocab_maps)
    vocab_maps_tensor = torch.full(
        (len(cluster_to_vocab_maps), max_len), -1, device=lm_head.weight.device
    )
    for i, m in enumerate(cluster_to_vocab_maps):
        length = m.shape[0]
        vocab_maps_tensor[i, :length] = m
        vocab_maps_tensor[i, length:] = m[0]
    return {
        "centroids": combined_centroids,
        "vocab_maps_tensor": vocab_maps_tensor,
    }


class FlashHead(nn.Module):
    """
    Implementation of FlashHead.

    :param lm_head:
        The original classification head.
    :param centroids:
        The cluster centroids to use.
    :param vocab_maps_tensor:
        A mapping between cluster centroid index and token index.
    :param n_probes:
        Number of probes to use.
    """

    def __init__(
        self,
        lm_head: nn.Linear,
        centroids: torch.Tensor,
        vocab_maps_tensor: torch.Tensor,
        n_probes: Optional[int] = None,
    ):
        super().__init__()

        self.original_lm_head = lm_head
        self.original_shape = lm_head.weight.shape

        self.register_buffer("vocab_maps_tensor", vocab_maps_tensor)
        self.register_buffer("centroids", centroids.contiguous())

        if n_probes is None:
            n_probes = int(centroids.shape[1] / 16)
        self.n_probes = n_probes

        pre_norm = centroids / centroids.norm(dim=0, keepdim=True)
        pre_norm = pre_norm.t().contiguous()
        self.register_buffer("pre_normalized_centroids", pre_norm)

        self.cluster_linear = nn.Linear(
            pre_norm.shape[1],
            pre_norm.shape[0],
            bias=False,
        )
        self.cluster_linear.weight = nn.Parameter(pre_norm)

        self.register_buffer(
            "vocab_maps_lengths", (vocab_maps_tensor != -1).sum(dim=1)
        )
        self.register_buffer(
            "row_indices", torch.arange(vocab_maps_tensor.shape[1])[None, :]
        )
        self.register_buffer(
            "output_buffer", torch.zeros((1, 1), dtype=torch.int64)
        )

    def _get_cluster_probs(self, hidden_states, temperature=1.0):
        hidden_states_norm = hidden_states
        similarities = torch.nn.functional.linear(
            hidden_states_norm, self.centroids.t(), bias=None
        )
        probs = torch.softmax(similarities / temperature, dim=-1)
        return probs

    def _get_top_clusters(
        self,
        hidden_states: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if do_sample:
            probs = self._get_cluster_probs(
                hidden_states=hidden_states, temperature=temperature
            )
            B, T, num_clusters = probs.shape
            probs_flat = probs.view(-1, num_clusters)
            sampled_indices = torch.multinomial(
                probs_flat, self.n_probes, replacement=False
            )
            top_clusters = sampled_indices.view(B, T, self.n_probes)
        else:
            similarities = self.cluster_linear(hidden_states)
            _, top_clusters = torch.topk(similarities, k=self.n_probes, dim=-1)
        return top_clusters

    def _get_cluster_logits(
        self,
        hidden_states: torch.Tensor,
        top_clusters: torch.Tensor,
        use_identical_tiebreak: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if top_clusters.shape[1] > 1 or top_clusters.shape[0] > 1:
            raise NotImplementedError(
                "Use original lm head for seq-len or bs > 1"
            )

        cluster_indices = top_clusters[0, 0]
        maps = self.vocab_maps_tensor.index_select(0, cluster_indices)
        indices = maps.flatten()

        mapping = None
        if use_identical_tiebreak:
            sorted = indices.sort()
            indices = sorted.values
            mapping = sorted.indices
        result = self.original_lm_head.weight.index_select(0, indices)

        final_result = (
            torch.nn.functional.linear(hidden_states, result, bias=None),
            mapping,
        )
        return final_result

    def forward(self, _hidden_states: torch.Tensor) -> torch.Tensor:
        raise ValueError(
            "Forward method not supported, please use get_next_token"
        )

    def get_next_token(
        self,
        hidden_states: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        use_identical_tiebreak: bool = False,
    ) -> torch.Tensor:
        """
        Return the next token, given `hidden_states`.

        :param hidden_states:
            The output of the model body.
        :param do_sample:
            Whether to sample the next token according to probabilities,
            or simply return the most probable.
        :param temperature:
            The temperature to use in the softmax
            (both the softmax in cluster probabilities and for the
            softmax in token probabilities).
            Only relevant when `do_sample` is ``True``.
        :param use_identical_tiebreak:
            Whether to reorder the logits so that when two logits are the same,
            the new head will use the same tiebreak as the original.
        :returns:
            The next predicted token, represented as an index.
        """
        top_clusters = self._get_top_clusters(
            hidden_states,
            do_sample=do_sample,
            temperature=temperature,
        )
        cluster_logits, mapping = self._get_cluster_logits(
            hidden_states, top_clusters, use_identical_tiebreak
        )

        if do_sample:
            probs = (cluster_logits[:, -1, :] / temperature).softmax(dim=-1)
            cluster_token_idx = torch.multinomial(probs, num_samples=1)
        else:
            cluster_token_idx = cluster_logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            if use_identical_tiebreak:
                cluster_token_idx = mapping[cluster_token_idx]

        cap = self.row_indices.shape[1]
        cluster_index = cluster_token_idx // cap
        relative_pos = cluster_token_idx % cap
        vocab_index = self.vocab_maps_tensor[
            top_clusters[0, 0, cluster_index], relative_pos
        ]
        self.output_buffer[0][0] = vocab_index

        return self.output_buffer
