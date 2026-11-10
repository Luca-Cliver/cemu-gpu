import argparse
import json
from pathlib import Path

import numpy as np


def load_tokenizer(model_directory):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("text input/output requires transformers in the active Python environment") from error
    return AutoTokenizer.from_pretrained(
        str(model_directory), local_files_only=True, use_fast=True, trust_remote_code=False)


def prepare_prompt(model_directory, batch_size, prompt_length, decode_steps,
                   prompt_ids=None, prompt_text=None):
    if batch_size < 1 or decode_steps < 1 or (prompt_ids is None and prompt_text is None and prompt_length < 1):
        raise ValueError("batch size, synthetic prompt length and Decode count must be positive")
    if prompt_ids is not None and prompt_text is not None:
        raise ValueError("choose either prompt IDs or prompt text")
    config = json.loads((Path(model_directory) / "config.json").read_text())
    vocabulary = config["vocab_size"]
    tokenizer = None
    if prompt_ids is not None:
        prompt = np.asarray(json.loads(Path(prompt_ids).read_text()))
    elif prompt_text is not None:
        if not prompt_text.strip():
            raise ValueError("prompt text must not be blank")
        tokenizer = load_tokenizer(model_directory)
        encoded = np.asarray(tokenizer(prompt_text, add_special_tokens=True,
                                       padding=False, truncation=False)["input_ids"])
        if encoded.ndim != 1 or not encoded.size:
            raise ValueError("tokenizer must return one nonempty token sequence")
        prompt = np.tile(encoded, (batch_size, 1))
    else:
        if vocabulary <= 3:
            raise ValueError("OPT vocabulary must contain more than three tokens")
        prompt = np.arange(batch_size * prompt_length, dtype=np.int64).reshape(batch_size, prompt_length)
        prompt = prompt % (vocabulary - 3) + 3
        prompt[:, 0] = config.get("bos_token_id", 2)
    if (prompt.ndim != 2 or prompt.shape[0] != batch_size or prompt.shape[1] < 1 or
            not np.issubdtype(prompt.dtype, np.integer) or
            np.any(prompt < 0) or np.any(prompt >= vocabulary) or
            np.any(prompt == config.get("pad_token_id", 1)) or
            prompt.shape[1] + decode_steps > config["max_position_embeddings"]):
        raise ValueError("prompt must contain valid unpadded integer IDs, match batch size and fit model capacity")
    return prompt.astype(np.int64, copy=False), tokenizer


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def comparison_case(model_directory, prompt, decode_steps, top_r, compression_ratio):
    return dict(model_config=json.loads((Path(model_directory) / "config.json").read_text()),
                prompt_ids=np.asarray(prompt).tolist(), decode_steps=decode_steps,
                attention_mode="sparf", top_r=top_r, compression_ratio=compression_ratio)


def load_reference(path, case):
    reference = json.loads(Path(path).read_text())
    if not isinstance(reference, dict) or reference.get("case") != case:
        raise ValueError("reference case differs or is missing: require identical model config, prompt IDs and SparF settings")
    expected = np.asarray(reference.get("tokens"))
    shape = (len(case["prompt_ids"]), case["decode_steps"] + 1)
    if (expected.shape != shape or not np.issubdtype(expected.dtype, np.integer) or
            np.any(expected < 0) or np.any(expected >= case["model_config"]["vocab_size"])):
        raise ValueError(f"reference tokens must have shape {shape} and valid integer IDs")
    return expected


def compare_tokens(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"generated token shape mismatch: {actual.shape} != {expected.shape}")
    differences = np.argwhere(actual != expected)
    if differences.size:
        batch, step = differences[0]
        raise ValueError(f"token mismatch: batch={batch}, generated_index={step} (0=Prefill), "
                         f"actual={actual[batch, step]}, reference={expected[batch, step]}; "
                         f"{len(differences)} differing tokens")


def decode_output(tokenizer, prompt, generated):
    options = dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return dict(continuation=tokenizer.batch_decode(np.asarray(generated).tolist(), **options),
                full=tokenizer.batch_decode(np.concatenate((prompt, generated), axis=1).tolist(), **options))


def main():
    parser = argparse.ArgumentParser(description="Prepare fixed OPT inputs without loading weights or opening CEMU")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--decode-steps", type=int, default=16)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--prompt-ids")
    inputs.add_argument("--prompt-text")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prompt, _ = prepare_prompt(args.model_dir, args.batch_size, args.prompt_length,
                               args.decode_steps, args.prompt_ids, args.prompt_text)
    write_json(args.output, prompt.tolist())
    print(f"[opt-input] batch={prompt.shape[0]}, prompt_length={prompt.shape[1]}, "
          f"decode_steps={args.decode_steps}; IDs={args.output}")


if __name__ == "__main__":
    main()
