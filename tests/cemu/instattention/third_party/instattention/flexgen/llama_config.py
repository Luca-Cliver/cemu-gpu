"""
The Llama model configurations and weight downloading utilities.

Some functions are adopted from https://github.com/alpa-projects/alpa/tree/main/examples/llm_serving/model.
"""

import argparse
import dataclasses
import glob
import os
import shutil

import numpy as np
from tqdm import tqdm

from transformers.models.llama.configuration_llama import *
from transformers.models.llama.modeling_llama import *


_CONFIG_FOR_DOC = "LlamaConfig"


@dataclasses.dataclass(frozen=True)
class InstLlamaConfig:
    name: str = "Llama-2-7b-hf"
    num_hidden_layers: int = 32
    max_seq_len: int = 4096
    hidden_size: int = 4096
    n_head: int = 32
    input_dim: int = 4096
    ffn_embed_dim: int = 11008
    pad: int = 1
    activation_fn: str = 'silu'
    vocab_size: int = 32000
    layer_norm_eps: float = 1e-6
    pad_token_id: int = 2
    dtype: type = np.float16
    intermediate_size: int = 11008
    max_position_embeddings=2048
    initializer_range=0.02
    rms_norm_eps=1e-6
    use_cache=True
    bos_token_id=1
    eos_token_id=2
    pretraining_tp=1
    tie_word_embeddings=False
    rope_theta=10000.0
    rope_scaling=None
    attention_bias=False
    attention_dropout=0.0
    head_dim = input_dim // n_head

    def model_bytes(self):
        h = self.input_dim
        return 	2 * (self.num_hidden_layers * (
        # self-attention
        h * (3 * h + 1) + h * (h + 1) +
        # mlp
        h * (4 * h + 1) + h * 4 * (h + 1) +
        # layer norm
        h * 4) +
        # embedding
        self.vocab_size * (h + 1))

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2


def Instget_llama_config(model_name_or_path):
    print("get llama config")
    llama_config = LlamaConfig.from_pretrained(model_name_or_path)
    config = InstLlamaConfig(name="Llama-2-13b-hf",
                    max_seq_len=4096, 
                    num_hidden_layers=llama_config.num_hidden_layers, 
                    n_head=llama_config.num_attention_heads,
                    hidden_size=llama_config.hidden_size,
                    input_dim=llama_config.hidden_size,
                    ffn_embed_dim=llama_config.intermediate_size,
                    intermediate_size=llama_config.intermediate_size,
                    vocab_size=llama_config.vocab_size,
                    # pad_token_id=llama_config.pad_token_id
                    )
    return config

def Instdownload_llama_weights(model_name, path, hf_model_path):
    import torch

    print("Actually just a converting function, no downloading needed. Just convert the bin file in the folder to the npy format.")

    folder = hf_model_path
    bin_files = glob.glob(os.path.join(folder, "*.bin"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file)
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())

            # shared embedding
            if "embed_tokens.weight" in name:
                shutil.copy(param_path, param_path.replace(
                    "embed_tokens.weight", "lm_head.weight"))
            state1 = state


# global torch_linear_init_backup
# global torch_layer_norm_init_backup


# def disable_torch_init():
#     """
#     Disable the redundant torch default initialization to accelerate model creation.
#     """
#     import torch
#     global torch_linear_init_backup
#     global torch_layer_norm_init_backup

#     torch_linear_init_backup = torch.nn.Linear.reset_parameters
#     setattr(torch.nn.Linear, "reset_parameters", lambda self: None)

#     torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters
#     setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


# def restore_torch_init():
#     """Rollback the change made by disable_torch_init."""
#     import torch
#     setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
#     setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)


# def disable_hf_opt_init():
#     """
#     Disable the redundant default initialization to accelerate model creation.
#     """
#     import transformers

#     setattr(transformers.models.opt.modeling_opt.OPTPreTrainedModel,
#             "_init_weights", lambda *args, **kwargs: None)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Meta/Llama-2-7b-hf")
    parser.add_argument("--path", type=str, default="~/llama_weights")
    args = parser.parse_args()

    # Instdownload_llama_weights(args.model, args.path)
