from __future__ import annotations
from math import exp

import random
import logging
from functools import cache
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

import comfy.model_management
import comfy.model_base
from comfy.model_base import ModelType
import folder_paths
from comfy import model_management
from comfy.model_patcher import ModelPatcher
from .util import join_prompts, remove_empty_str

MODEL_FOLDER_NAME = "prompt_expansion"

CONFIGS_DIR = Path(__file__).parent.joinpath("configs")

logger = logging.getLogger(__name__)

fooocus_magic_split = [", extremely", ", intricate,"]

disallowed_chars_table = str.maketrans("", "", "[]【】()（）|:：")


def safe_str(x):
    x = str(x)
    # remove multiple whitespaces
    x = " ".join(x.split())
    return x.strip(",. \r\n")


def remove_pattern(x, pattern):
    for p in pattern:
        x = x.replace(p, "")
    return x


class ComfyTransformerModel(comfy.model_base.BaseModel):
    def __init__(self, model_name: str, model_type=ModelType.EPS, device=None, *args, **kwargs):
        # Find the full path to the model
        model_path = folder_paths.get_full_path("prompt_expansion", model_name)
        if model_path is None:
            raise ValueError(f"Model {model_name} not found in prompt_expansion folder.")

        # If model is a file, use the parent directory
        if Path(model_path).is_file():
            model_path = str(Path(model_path).parent)

        class MinimalConfig:
            def __init__(self):
                self.unet_config = {"disable_unet_model_creation": True}
                self.latent_format = None
                self.custom_operations = None
                self.scaled_fp8 = None
                self.memory_usage_factor = 1.0
                self.manual_cast_dtype = None
                self.optimizations = {}
                self.sampling_settings = {}

        config = MinimalConfig()

        super().__init__(config, model_type=model_type, device=device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        self.load_device = comfy.model_management.text_encoder_device()
        self.offload_device = comfy.model_management.text_encoder_offload_device()

        if "mps" in self.load_device.type:
            self.load_device = torch.device("cpu")

        if "cpu" not in self.load_device.type and comfy.model_management.should_use_fp16():
            self.model.half()

        self.model.eval()
        self.model.to(self.load_device)
        self.device = self.load_device

    def apply_model(self, prompt: str, seed: int) -> str:
        with torch.no_grad():
            origin = safe_str(prompt)
            prompt = origin + fooocus_magic_split[seed % len(fooocus_magic_split)]

            tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
            tokenized_kwargs.data["input_ids"] = tokenized_kwargs.data["input_ids"].to(self.load_device)
            tokenized_kwargs.data["attention_mask"] = tokenized_kwargs.data["attention_mask"].to(self.load_device)

            # https://huggingface.co/blog/introducing-csearch
            # https://huggingface.co/docs/transformers/generation_strategies
            features = self.model.generate(
                **tokenized_kwargs, num_beams=1, max_new_tokens=256, do_sample=True
            )

            response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
            result = response[0][len(origin):]
            result = safe_str(result)
            result = result.translate(disallowed_chars_table)
            return result


class PromptExpansion:
    # noinspection PyPep8Naming,PyMethodParameters
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("prompt_expansion"),),
                "text": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "log_prompt": ("BOOLEAN", {"default": False})
            },
        }

    RETURN_TYPES = (
        "STRING",
        "INT",
    )
    RETURN_NAMES = (
        "expanded_prompt",
        "seed",
    )
    FUNCTION = "expand_prompt"  # Function name

    CATEGORY = "utils"  # Category for organization

    @staticmethod
    @torch.no_grad()
    def expand_prompt(model_name: str, text: str, seed: int, log_prompt: bool):
        expansion_model = ComfyTransformerModel(model_name)

        prompt = remove_empty_str([safe_str(text)], default="")[0]

        max_seed = 1024 * 1024 * 1024
        if not isinstance(seed, int):
            seed = random.randint(1, max_seed)
        elif seed < 0:
            seed = abs(seed)
        elif seed > max_seed:
            seed = seed % max_seed
            
        prompt_parts = []
        expanded_parts = []
            
        # Split prompt if longer than 256
        if len(prompt) > 256:
            prompt_lines = prompt.splitlines()
            # Fill part until 256
            prompt_parts = [""]
            filled_chars = 0
            for line in prompt_lines:
                # When adding the line would exceed 256, start a new part
                if filled_chars + len(line) > 256:
                    prompt_parts.append(line)
                    filled_chars = len(line)
                else:
                    prompt_parts[-1] += line
                    filled_chars += len(line)
        else:
            prompt_parts = [prompt]
        
        for i, part in enumerate(prompt_parts):
            expansion_part = expansion_model.apply_model(part, seed)
            full_part = join_prompts(part, expansion_part)
            expanded_parts.append(full_part)
            
        expanded_prompt = "\n".join(expanded_parts)
            
        if log_prompt:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Prompt: {prompt!r}")
                logger.info(f"Expanded Prompt: {expanded_prompt!r}")
            else:
                print(f"Prompt: {prompt!r}")
                print(f"Expanded Prompt: {expanded_prompt!r}")

        return expanded_prompt, seed


# Define a mapping of node class names to their respective classes
NODE_CLASS_MAPPINGS = {"PromptExpansion": PromptExpansion}

# A dictionary that contains human-readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptExpansion": "[Inference.Core] Prompt Expansion"
}
