# Copyright (C) 2025 Embedl AB

"""Custom Gemma model with FlashHead."""

import os

from transformers.models.gemma3 import Gemma3Config as Config
from transformers.models.gemma3 import Gemma3ForCausalLM as Model

from embedl.models import ERROR_MSG
from embedl.models.flash_head import FlashHead, get_flash_head_parameters


class FlashHeadGemma3TextConfig(Config):
    """Gemma config with FlashHead."""

    model_type = "flash_head_gemma3_text"

    def __init__(
        self,
        model_or_dir: str = None,
        flash_head_cache_dir: str = "flash_head_assets",
        flash_head_special_token_ids: list[int] = None,
        n_clusters: int = None,
        n_probes: int = None,
        creation_time: float = None,
        enforce_equal_cluster_sizes: bool = True,
        **kwargs,
    ):
        self.model_or_dir = model_or_dir
        self.flash_head_cache_dir = flash_head_cache_dir
        self.flash_head_special_token_ids = flash_head_special_token_ids
        self.n_clusters = n_clusters
        self.n_probes = n_probes
        self.creation_time = creation_time
        self.enforce_equal_cluster_sizes = enforce_equal_cluster_sizes
        super().__init__(**kwargs)


class FlashHeadGemma3ForCausalLM(Model):
    """Gemma model with FlashHead."""

    config_class = FlashHeadGemma3TextConfig

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        config = model.config

        if not hasattr(config, "flash_head_cache_dir"):
            return model

        cache_dir = config.flash_head_cache_dir
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(pretrained_model_name_or_path, cache_dir)

        flash_params = get_flash_head_parameters(
            lm_head=model.lm_head,
            cache_dir=cache_dir,
            model_or_dir=pretrained_model_name_or_path,
            n_clusters=config.n_clusters,
        )

        model.lm_head = FlashHead(
            lm_head=model.lm_head,
            n_probes=config.n_probes,
            **flash_params,
        )

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        model.lm_head = model.lm_head.to(device=device, dtype=dtype)

        return model

    def forward(self, *_args, **_kwargs):
        """
        Guard preventing accidental inference with transformers or vLLM.
        """
        raise RuntimeError(ERROR_MSG)

    def generate(self, *_args, **_kwargs):
        """
        Guard preventing accidental inference with transformers or vLLM.
        """
        raise RuntimeError(ERROR_MSG)
