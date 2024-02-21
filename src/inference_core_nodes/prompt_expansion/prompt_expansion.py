from __future__ import annotations
from math import exp

import random
import logging
from functools import cache
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

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


class FooocusExpansion:
    def __init__(self, model_directory: str):
        model_directory = Path(model_directory)
        if not model_directory.exists() or not model_directory.is_dir():
            raise ValueError(f"Model directory {model_directory} does not exist")

        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        self.model = AutoModelForCausalLM.from_pretrained(model_directory)

        positive_tokens = (
            model_directory.joinpath("positive.txt").read_text().splitlines()
        )

        positive_tokens = []

        self.model.eval()

        load_device = model_management.text_encoder_device()

        if "mps" in load_device.type:
            load_device = torch.device("cpu")

        if "cpu" not in load_device.type and model_management.should_use_fp16():
            self.model.half()

        offload_device = model_management.text_encoder_offload_device()
        self.patcher = ModelPatcher(
            self.model, load_device=load_device, offload_device=offload_device
        )

    def __call__(self, prompt: str, seed: int) -> str:
        model_management.load_model_gpu(self.patcher)
        set_seed(seed)
        origin = safe_str(prompt)
        prompt = origin + fooocus_magic_split[seed % len(fooocus_magic_split)]

        tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
        tokenized_kwargs.data["input_ids"] = tokenized_kwargs.data["input_ids"].to(
            self.patcher.load_device
        )
        tokenized_kwargs.data["attention_mask"] = tokenized_kwargs.data[
            "attention_mask"
        ].to(self.patcher.load_device)

        # https://huggingface.co/blog/introducing-csearch
        # https://huggingface.co/docs/transformers/generation_strategies
        features = self.model.generate(
            **tokenized_kwargs, num_beams=1, max_new_tokens=256, do_sample=True
        )

        response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
        result = response[0][len(origin) :]
        result = safe_str(result)
        result = result.translate(disallowed_chars_table)
        return result

    def expand_and_join(self, prompt: str, seed: int) -> str:
        expansion = self(prompt, seed)
        return join_prompts(prompt, expansion)


@cache
def load_expansion_runner(model_name: str):
    model_path = folder_paths.get_full_path(MODEL_FOLDER_NAME, model_name)
    
    # If model is a file, use the parent directory
    if Path(model_path).is_file():
        model_path = str(Path(model_path).parent)
    
    return FooocusExpansion(model_path)


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
        expansion = load_expansion_runner(model_name)

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
            expansion_part = expansion(part, seed)
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
NODE_CLASS_MAPPINGS = {"Inference_Core_PromptExpansion": PromptExpansion}

# A dictionary that contains human-readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Inference_Core_PromptExpansion": "[Inference.Core] Prompt Expansion"
}
