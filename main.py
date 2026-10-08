#!/usr/bin/env python3

from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import logging
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed as hf_set_seed)

###############################
#           LOGGING           #
# #############################

LOG = logging.getLogger("composition")

# Static composition controls depend on the ordered skill pair and seed, not on
# the composition weight or PEFT merge method. A single composition sweep can
# therefore reuse them across all weights/methods.
COMPOSITION_STATIC_CACHE: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
COMPOSITION_ENDPOINT_CACHE: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}


def setup_logging(run_dir: Path, level: str = "INFO") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S", )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def read_yaml(path: Path) -> Dict[str, Any]:

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate YAML key {key!r} in {path}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=UniqueKeyLoader)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hf_set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    LOG.info("Global seed set to %d", seed)


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def package_versions() -> Dict[str, str]:
    names = ["torch", "transformers", "datasets", "peft", "accelerate", "tokenizers", "safetensors", "pyyaml", "numpy"]
    result: Dict[str, str] = {}
    for name in names:
        try:
            module = __import__(name)
            result[name] = getattr(module, "__version__", "unknown")
        except Exception:
            result[name] = "not-installed"
    return result


def system_metadata() -> Dict[str, Any]:
    return {"timestamp_utc": now_ts(), "python": sys.version, "platform": platform.platform(), "git_revision": git_revision(), "packages": package_versions(), "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, }


###############################
#        ATOMIC TASKS         #
# #############################

VALUES = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
NOISE_PATTERNS = {"none": "", "prefix": "Note: process carefully. ", "suffix": " Return only the requested result."}
ALPHA_VALUES = [value for value in VALUES if value != "alpha"]
VOWEL_INITIAL_VALUES = [value for value in VALUES if value[0].lower() in "aeiou"]
CONSONANT_INITIAL_VALUES = [value for value in VALUES if value[0].lower() not in "aeiou"]


@dataclass(frozen=True)
class Example:
    task_id: str
    split: str
    latent: Dict[str, Any]
    prompt: str
    target: str


def deterministic_rng(seed: int) -> random.Random:
    return random.Random(seed)


def balanced_sequence(values: Sequence[Any], n: int, rng: random.Random, ) -> List[Any]:
    if not values:
        raise ValueError("balanced_sequence requires at least one value")
    values = list(values)
    repeats = (n + len(values) - 1) // len(values)
    sequence = (values * repeats)[:n]
    rng.shuffle(sequence)
    return sequence


def balanced_binary_labels(n: int, rng: random.Random, ) -> List[str]:
    return balanced_sequence(["YES", "NO"], n, rng)


def generate_key_value_fields(field_count: int, rng: random.Random, value_offset: int = 0, ) -> List[Tuple[str, str]]:
    if field_count < 1:
        raise ValueError("field_count must be >= 1")
    fields = []
    for j in range(field_count):
        key = chr(ord("a") + j)
        value = VALUES[(value_offset + j) % len(VALUES)]
        fields.append((key, value))
    return fields


def render_fields(fields: Sequence[Tuple[str, str]], output_format: str, ) -> str:
    if output_format == "lines":
        return "\n".join(f"{key}: {value}" for key, value in fields)
    if output_format == "comma":
        return ", ".join(f"{key}={value}" for key, value in fields)
    raise ValueError(f"Unknown output format: {output_format}")


def render_json_object(fields: Sequence[Tuple[str, str]], ) -> str:
    return ("{" + ", ".join(f'"{key}": "{value}"' for key, value in fields) + "}")


def render_values(values: Sequence[str], separator: str, ) -> str:
    if separator == "comma":
        return ", ".join(values)
    if separator == "pipe":
        return " | ".join(values)
    if separator == "space":
        return " ".join(values)
    if separator == "lines":
        return "\n".join(values)
    raise ValueError(f"Unknown separator: {separator}")


def generate_structured_extraction(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    field_counts = balanced_sequence([2, 3, 4], n, rng)
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng)
    output_formats = balanced_sequence(["lines", "comma"], n, rng)
    value_offsets = balanced_sequence([0, 1, 2, 3, 4], n, rng)
    records: List[Example] = []
    for i in range(n):
        field_count = overrides.get("field_count", field_counts[i])
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i])
        output_format = overrides.get("output_format", output_formats[i])
        value_offset = overrides.get("value_offset", value_offsets[i])
        fields = generate_key_value_fields(field_count=field_count, rng=rng, value_offset=value_offset + i)
        body = " | ".join(f"{key}={value}" for key, value in fields)
        target = render_fields(fields, output_format)
        prompt = (f"{NOISE_PATTERNS[noise_pattern]}"
                  f"Extract every key-value field from the record. "
                  f"Return all fields and do not omit any field.\n"
                  f"Record: {body}\n"
                  f"Format the answer as {output_format}. "
                  f"For 'lines', output one 'key: value' pair per line. "
                  f"For 'comma', output 'key=value' pairs separated by commas.\n"
                  f"Answer:")
        records.append(Example(task_id=task_id, split=split, latent={"input_schema": "key_value_record", "field_count": field_count, "noise_pattern": noise_pattern, "output_format": output_format, "value_offset": value_offset, }, prompt=prompt, target=target, ))
    return records


def generate_structured_classification(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    field_counts = balanced_sequence([2, 3, 4], n, rng)
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng)
    value_offsets = balanced_sequence([0, 1, 2, 3, 4], n, rng)
    labels = balanced_binary_labels(n, rng)
    records: List[Example] = []
    for i in range(n):
        field_count = overrides.get("field_count", field_counts[i])
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i])
        value_offset = overrides.get("value_offset", value_offsets[i])
        desired_label = labels[i]
        if desired_label == "YES":
            non_alpha_values = [value for value in VALUES if value != "alpha"]
            rotated_non_alpha = [non_alpha_values[(value_offset + i + j) % len(non_alpha_values)] for j in range(field_count - 1)]
            alpha_position = rng.randrange(field_count)
            values = []
            non_alpha_index = 0
            for position in range(field_count):
                if position == alpha_position:
                    values.append("alpha")
                else:
                    values.append(rotated_non_alpha[non_alpha_index])
                    non_alpha_index += 1
        else:
            non_alpha_values = [value for value in VALUES if value != "alpha"]
            values = [non_alpha_values[(value_offset + i + j) % len(non_alpha_values)] for j in range(field_count)]
        label = desired_label
        fields = [(chr(ord("a") + j), values[j], ) for j in range(field_count)]
        body = " | ".join(f"{key}={value}" for key, value in fields)
        prompt = (f"{NOISE_PATTERNS[noise_pattern]}"
                  f"Classify the record as YES if at least one "
                  f"field value is exactly 'alpha'; otherwise "
                  f"classify it as NO.\n"
                  f"Record: {body}\n"
                  f"Return exactly one label: YES or NO.\n"
                  f"Answer:")
        records.append(Example(task_id=task_id, split=split, latent={"input_schema": "key_value_record", "field_count": field_count, "noise_pattern": noise_pattern, "rule": "contains_alpha", "label_format": "label", "value_offset": value_offset, "class": label, }, prompt=prompt, target=label, ))
    return records


def generate_field_filtering(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    field_counts = balanced_sequence([3, 4, 5], n, rng)
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng)
    output_formats = balanced_sequence(["lines", "comma"], n, rng)
    value_offsets = balanced_sequence([0, 1, 2, 3, 4], n, rng)
    records: List[Example] = []
    for i in range(n):
        field_count = overrides.get("field_count", field_counts[i], )
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i], )
        output_format = overrides.get("output_format", output_formats[i], )
        value_offset = overrides.get("value_offset", value_offsets[i], )
        fields = generate_key_value_fields(field_count=field_count, rng=rng, value_offset=value_offset + i, )
        selected = fields[::2]
        target = render_fields(selected, output_format, )
        body = " | ".join(f"{key}={value}" for key, value in fields)
        prompt = (f"{NOISE_PATTERNS[noise_pattern]}"
                  f"Filter the record by keeping only fields in "
                  f"odd-numbered positions: 1st, 3rd, 5th, and so on. "
                  f"Discard all even-numbered positions.\n"
                  f"Record: {body}\n"
                  f"Format the answer as {output_format}. "
                  f"For 'lines', output one 'key: value' pair per line. "
                  f"For 'comma', output 'key=value' pairs separated by commas.\n"
                  f"Answer:")
        records.append(Example(task_id=task_id, split=split, latent={"input_schema": "key_value_record", "field_count": field_count, "noise_pattern": noise_pattern, "filter_rule": "odd_positions", "output_format": output_format, "value_offset": value_offset, }, prompt=prompt, target=target, ))
    return records


def generate_format_conversion(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    field_counts = balanced_sequence([2, 3, 4], n, rng, )
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng, )
    value_offsets = balanced_sequence([0, 1, 2, 3, 4], n, rng, )
    records: List[Example] = []
    for i in range(n):
        field_count = overrides.get("field_count", field_counts[i], )
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i], )
        source_format = overrides.get("source_format", "lines", )
        value_offset = overrides.get("value_offset", value_offsets[i], )
        fields = generate_key_value_fields(field_count=field_count, rng=rng, value_offset=value_offset + i, )
        if source_format == "lines":
            body = "\n".join(f"{key}: {value}" for key, value in fields)
        elif source_format == "comma":
            body = ", ".join(f"{key}={value}" for key, value in fields)
        else:
            raise ValueError(f"Unknown source format: {source_format}")
        target = render_json_object(fields)
        prompt = (f"{NOISE_PATTERNS[noise_pattern]}"
                  f"Convert the key-value record into a JSON object. "
                  f"Preserve every key and its value exactly.\n"
                  f"Record:\n{body}\n"
                  f"Return only the JSON object, with each key represented "
                  f"as a JSON string and each value represented as a JSON string.\n"
                  f"Answer:")
        records.append(Example(task_id=task_id, split=split, latent={"input_schema": "key_value_record", "field_count": field_count, "noise_pattern": noise_pattern, "source_format": source_format, "target_format": "json", "value_offset": value_offset, }, prompt=prompt, target=target, ))
    return records


def generate_field_sorting(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    field_counts = balanced_sequence([3, 4, 5], n, rng, )
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng, )
    output_formats = balanced_sequence(["lines", "comma"], n, rng, )
    value_offsets = balanced_sequence([0, 1, 2, 3, 4], n, rng, )
    records = []
    for i in range(n):
        field_count = overrides.get("field_count", field_counts[i], )
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i], )
        output_format = overrides.get("output_format", output_formats[i], )
        value_offset = overrides.get("value_offset", value_offsets[i], )
        fields = generate_key_value_fields(field_count=field_count, rng=rng, value_offset=value_offset + i, )
        sorted_fields = sorted(fields, key=lambda item: item[1], )
        target = render_fields(sorted_fields, output_format, )
        body = " | ".join(f"{key}={value}" for key, value in fields)
        prompt = (f"{NOISE_PATTERNS[noise_pattern]}"
                  f"Sort all fields alphabetically by their values "
                  f"in ascending order.\n"
                  f"Record: {body}\n"
                  f"Format the answer as {output_format}. "
                  f"Preserve each key-value pair exactly.\n"
                  f"Answer:")
        records.append(Example(task_id=task_id, split=split, latent={"input_schema": "key_value_record", "field_count": field_count, "noise_pattern": noise_pattern, "output_format": output_format, "value_offset": value_offset, "sort_order": "ascending_value", }, prompt=prompt, target=target, ))
    return records


def generate_value_normalization(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    field_counts = balanced_sequence([2, 3, 4], n, rng, )
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng, )
    output_formats = balanced_sequence(["lines", "comma"], n, rng, )
    value_offsets = balanced_sequence([0, 1, 2, 3, 4], n, rng, )
    records = []
    for i in range(n):
        field_count = overrides.get("field_count", field_counts[i], )
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i], )
        output_format = overrides.get("output_format", output_formats[i], )
        value_offset = overrides.get("value_offset", value_offsets[i], )
        # Use mixed-case inputs so normalization is a real operation.  The
        # previous generator produced lowercase values only, making this task
        # almost an identity function and making it a poor composition probe.
        fields = generate_key_value_fields(field_count=field_count, rng=rng, value_offset=value_offset + i, )
        fields = [(key, value.upper() if (index + i) % 2 == 0 else value.title()) for index, (key, value) in enumerate(fields)]
        normalized_fields = [(key, value.lower()) for key, value in fields]
        target = render_fields(normalized_fields, output_format, )
        body = " | ".join(f"{key}={value}" for key, value in fields)
        prompt = (f"{NOISE_PATTERNS[noise_pattern]}"
                  f"Normalize every field value by converting it to "
                  f"lowercase. Preserve all keys and all fields.\n"
                  f"Record: {body}\n"
                  f"Format the answer as {output_format}.\n"
                  f"Answer:")
        records.append(Example(task_id=task_id, split=split, latent={"input_schema": "key_value_record", "field_count": field_count, "noise_pattern": noise_pattern, "output_format": output_format, "value_offset": value_offset, "normalization": "lowercase", }, prompt=prompt, target=target, ))
    return records


def generate_field_renaming(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    field_counts = balanced_sequence([2, 3, 4], n, rng, )
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng, )
    output_formats = balanced_sequence(["lines", "comma"], n, rng, )
    value_offsets = balanced_sequence([0, 1, 2, 3, 4], n, rng, )
    records = []
    for i in range(n):
        field_count = overrides.get("field_count", field_counts[i], )
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i], )
        output_format = overrides.get("output_format", output_formats[i], )
        value_offset = overrides.get("value_offset", value_offsets[i], )
        fields = generate_key_value_fields(field_count=field_count, rng=rng, value_offset=value_offset + i, )
        renamed_fields = [(f"field_{j + 1}", value, ) for j, (_, value) in enumerate(fields)]
        target = render_fields(renamed_fields, output_format, )
        body = " | ".join(f"{key}={value}" for key, value in fields)
        prompt = (f"{NOISE_PATTERNS[noise_pattern]}"
                  f"Rename every field key using this mapping: "
                  f"the 1st field becomes field_1, the 2nd field becomes "
                  f"field_2, the 3rd field becomes field_3, and so on. "
                  f"Preserve every value exactly.\n"
                  f"Record: {body}\n"
                  f"Format the answer as {output_format}.\n"
                  f"Answer:")
        records.append(Example(task_id=task_id, split=split, latent={"input_schema": "key_value_record", "field_count": field_count, "noise_pattern": noise_pattern, "output_format": output_format, "value_offset": value_offset, "rename_scheme": "field_index", }, prompt=prompt, target=target, ))
    return records


def generate_duplicate_removal(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    field_counts = balanced_sequence([4, 5, 6], n, rng, )
    duplicate_counts = balanced_sequence([1, 2], n, rng, )
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng, )
    records = []
    for i in range(n):
        field_count = overrides.get("field_count", field_counts[i], )
        duplicate_count = overrides.get("duplicate_count", duplicate_counts[i], )
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i], )
        values = [VALUES[(i + j) % len(VALUES)] for j in range(field_count)]
        duplicate_count = min(duplicate_count, field_count - 1, )
        for d in range(duplicate_count):
            source_index = d
            target_index = field_count - 1 - d
            values[target_index] = values[source_index]
        fields = [(chr(ord("a") + j), values[j], ) for j in range(field_count)]
        seen = set()
        deduplicated = []
        for key, value in fields:
            if value not in seen:
                deduplicated.append((key, value))
                seen.add(value)
        target = render_fields(deduplicated, "comma", )
        body = " | ".join(f"{key}={value}" for key, value in fields)
        prompt = (f"{NOISE_PATTERNS[noise_pattern]}"
                  f"Remove duplicate values from the record. "
                  f"Keep the first occurrence of each value and remove "
                  f"later fields having a value that has already appeared.\n"
                  f"Record: {body}\n"
                  f"Return the remaining fields as comma-separated "
                  f"key=value pairs.\n"
                  f"Answer:")
        records.append(Example(task_id=task_id, split=split, latent={"input_schema": "key_value_record", "field_count": field_count, "noise_pattern": noise_pattern, "duplicate_count": duplicate_count, "deduplication": "keep_first", }, prompt=prompt, target=target, ))
    return records


def generate_value_selection(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    field_counts = balanced_sequence([3, 4, 5], n, rng, )
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng, )
    value_offsets = balanced_sequence([0, 1, 2, 3, 4], n, rng, )
    records = []
    for i in range(n):
        field_count = overrides.get("field_count", field_counts[i], )
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i], )
        value_offset = overrides.get("value_offset", value_offsets[i], )
        fields = generate_key_value_fields(field_count=field_count, rng=rng, value_offset=value_offset + i, )
        selected = [(key, value) for key, value in fields if value[0].lower() in "aeiou"]
        target = render_fields(selected, "comma", )
        if not target:
            target = "NONE"
        body = " | ".join(f"{key}={value}" for key, value in fields)
        prompt = (f"{NOISE_PATTERNS[noise_pattern]}"
                  f"Select every field whose value begins with a vowel "
                  f"(A, E, I, O, or U). Discard all other fields.\n"
                  f"Record: {body}\n"
                  f"Return the selected fields as comma-separated "
                  f"key=value pairs. If no field qualifies, return NONE.\n"
                  f"Answer:")
        records.append(Example(task_id=task_id, split=split, latent={"input_schema": "key_value_record", "field_count": field_count, "noise_pattern": noise_pattern, "value_offset": value_offset, "selection_rule": "value_starts_with_vowel", }, prompt=prompt, target=target, ))
    return records


def generate_value_concatenation(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    field_counts = balanced_sequence([2, 3, 4], n, rng, )
    separators = balanced_sequence(["comma", "pipe"], n, rng, )
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng, )
    value_offsets = balanced_sequence([0, 1, 2, 3, 4], n, rng, )
    records = []
    for i in range(n):
        field_count = overrides.get("field_count", field_counts[i], )
        separator = overrides.get("separator", separators[i], )
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i], )
        value_offset = overrides.get("value_offset", value_offsets[i], )
        fields = generate_key_value_fields(field_count=field_count, rng=rng, value_offset=value_offset + i, )
        values = [value for _, value in fields]
        target = render_values(values, separator, )
        body = " | ".join(f"{key}={value}" for key, value in fields)
        separator_description = {"comma": "commas", "pipe": "vertical bars (|)", "space": "spaces", }[separator]
        prompt = (f"{NOISE_PATTERNS[noise_pattern]}"
                  f"Read the field values from left to right and "
                  f"concatenate them in their original order. "
                  f"Do not include the field names.\n"
                  f"Record: {body}\n"
                  f"Separate the values using {separator_description}.\n"
                  f"Answer:")
        records.append(Example(task_id=task_id, split=split, latent={"input_schema": "key_value_record", "field_count": field_count, "noise_pattern": noise_pattern, "value_offset": value_offset, "separator": separator, "operation": "ordered_value_concatenation", }, prompt=prompt, target=target, ))
    return records


def generate_key_value_swap(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    field_counts = balanced_sequence([2, 3, 4], n, rng)
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng)
    output_formats = balanced_sequence(["lines", "comma"], n, rng)
    value_offsets = balanced_sequence([0, 1, 2, 3, 4], n, rng)
    records = []
    for i in range(n):
        field_count = overrides.get("field_count", field_counts[i])
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i])
        output_format = overrides.get("output_format", output_formats[i])
        value_offset = overrides.get("value_offset", value_offsets[i])
        fields = generate_key_value_fields(field_count=field_count, rng=rng, value_offset=value_offset + i)
        swapped = [(value, key) for key, value in fields]
        target = render_fields(swapped, output_format)
        body = " | ".join(f"{key}={value}" for key, value in fields)
        prompt = (f"{NOISE_PATTERNS[noise_pattern]}"
                  f"Swap the key and value of each field in the record. "
                  f"Each original value becomes the key, and each original key becomes the value.\n"
                  f"Record: {body}\n"
                  f"Format the answer as {output_format}. "
                  f"For 'lines', output one 'key: value' pair per line. "
                  f"For 'comma', output 'key=value' pairs separated by commas.\n"
                  f"Answer:")
        records.append(Example(task_id=task_id, split=split, latent={"input_schema": "key_value_record", "field_count": field_count, "noise_pattern": noise_pattern, "output_format": output_format, "value_offset": value_offset, "operation": "key_value_swap", }, prompt=prompt, target=target, ))
    return records


def generate_field_reversal(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    # Train on six-field records as well.  The old latent split used six
    # fields despite never showing that length during training, so failures
    # were confounded with atomic field-count extrapolation.
    field_counts = balanced_sequence([3, 4, 5, 6], n, rng)
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng)
    output_formats = balanced_sequence(["lines", "comma"], n, rng)
    value_offsets = balanced_sequence([0, 1, 2, 3, 4], n, rng)
    records = []
    for i in range(n):
        field_count = overrides.get("field_count", field_counts[i])
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i])
        output_format = overrides.get("output_format", output_formats[i])
        value_offset = overrides.get("value_offset", value_offsets[i])
        fields = generate_key_value_fields(field_count=field_count, rng=rng, value_offset=value_offset + i)
        reversed_fields = list(reversed(fields))
        target = render_fields(reversed_fields, output_format)
        body = " | ".join(f"{key}={value}" for key, value in fields)
        prompt = (f"{NOISE_PATTERNS[noise_pattern]}"
                  f"Reverse the sequence of fields in the record from right to left, "
                  f"so the last field comes first and the first field comes last.\n"
                  f"Record: {body}\n"
                  f"Format the answer as {output_format}. "
                  f"For 'lines', output one 'key: value' pair per line. "
                  f"For 'comma', output 'key=value' pairs separated by commas.\n"
                  f"Answer:")
        records.append(Example(task_id=task_id, split=split, latent={"input_schema": "key_value_record", "field_count": field_count, "noise_pattern": noise_pattern, "output_format": output_format, "value_offset": value_offset, "operation": "field_reversal", }, prompt=prompt, target=target, ))
    return records


def _generate_affix_task(task_id: str, split: str, n: int, seed: int, target_kind: str, affix: str, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    """Generate an ordinary schema-preserving key/value transformation."""
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    field_counts = balanced_sequence([3, 4, 5, 6], n, rng)
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng)
    output_formats = balanced_sequence(["lines", "comma"], n, rng)
    value_offsets = balanced_sequence([0, 1, 2, 3, 4, 5, 6], n, rng)
    operation = f"{target_kind}_affix"
    target_text = "field keys" if target_kind == "key" else "field values"
    direction_text = "before" if affix == "prefix" else "after"
    marker = ("task_" if affix == "prefix" and target_kind == "key" else
              "_task" if target_kind == "key" else
              "value_" if affix == "prefix" else "_value")
    records = []
    for i in range(n):
        field_count = int(overrides.get("field_count", field_counts[i]))
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i])
        output_format = overrides.get("output_format", output_formats[i])
        value_offset = int(overrides.get("value_offset", value_offsets[i]))
        fields = generate_key_value_fields(field_count=field_count, rng=rng, value_offset=value_offset + i)
        transformed = [(("task_" + key) if target_kind == "key" and affix == "prefix" else (key + "_task") if target_kind == "key" else key,
                        ("value_" + value) if target_kind == "value" and affix == "prefix" else (value + "_value") if target_kind == "value" else value)
                       for key, value in fields]
        body = " | ".join(f"{key}={value}" for key, value in fields)
        prompt = (f"{NOISE_PATTERNS[noise_pattern]}"
                  f"Add the fixed marker {marker!r} {direction_text} every {target_text}. "
                  "Preserve the number and order of fields exactly.\n"
                  f"Record: {body}\n"
                  f"Format the answer as {output_format}.\n"
                  "Answer:")
        records.append(Example(task_id=task_id, split=split, latent={"input_schema": "key_value_record", "field_count": field_count, "noise_pattern": noise_pattern, "output_format": output_format, "value_offset": value_offset, "target": target_kind, "affix": affix, "operation": operation}, prompt=prompt, target=render_fields(transformed, output_format)))
    return records


def generate_key_prefixing(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    return _generate_affix_task(task_id, split, n, seed, "key", "prefix", latent_overrides)


def generate_key_suffixing(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    return _generate_affix_task(task_id, split, n, seed, "key", "suffix", latent_overrides)


def generate_value_prefixing(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    return _generate_affix_task(task_id, split, n, seed, "value", "prefix", latent_overrides)


def generate_value_suffixing(task_id: str, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, ) -> List[Example]:
    return _generate_affix_task(task_id, split, n, seed, "value", "suffix", latent_overrides)


GENERATOR_REGISTRY = {"structured_extraction": generate_structured_extraction, "structured_classification": generate_structured_classification, "field_filtering": generate_field_filtering, "format_conversion": generate_format_conversion, "field_sorting": generate_field_sorting, "value_normalization": generate_value_normalization, "field_renaming": generate_field_renaming, "duplicate_removal": generate_duplicate_removal, "value_selection": generate_value_selection, "value_concatenation": generate_value_concatenation, "key_value_swap": generate_key_value_swap, "field_reversal": generate_field_reversal, "key_prefixing": generate_key_prefixing, "key_suffixing": generate_key_suffixing, "value_prefixing": generate_value_prefixing, "value_suffixing": generate_value_suffixing, }

###############################
#       COMPOSED TASKS        #
# #############################


@dataclass(frozen=True)
class StructuredRecord:
    fields: Tuple[Tuple[str, str], ...]


def transform_structured_extraction(record: StructuredRecord, ) -> StructuredRecord:
    """Extraction is intentionally kept as an atomic operation, not a composition step.

    It is an identity over the internal representation, so composing it with a
    downstream record operation would make the downstream adapter sufficient by
    itself.  The composition registry below excludes it from audited pairs.
    """
    return StructuredRecord(fields=tuple(record.fields))


def transform_field_filtering(record: StructuredRecord, ) -> StructuredRecord:
    return StructuredRecord(fields=tuple(field for index, field in enumerate(record.fields) if index % 2 == 0))


def transform_field_sorting(record: StructuredRecord, ) -> StructuredRecord:
    return StructuredRecord(fields=tuple(sorted(record.fields, key=lambda item: item[1], )))


def transform_value_selection(record: StructuredRecord, ) -> StructuredRecord:
    return StructuredRecord(fields=tuple(field for field in record.fields if field[1][0].lower() in "aeiou"))


def transform_duplicate_removal(record: StructuredRecord, ) -> StructuredRecord:
    seen = set()
    result = []
    for key, value in record.fields:
        if value not in seen:
            result.append((key, value))
            seen.add(value)
    return StructuredRecord(fields=tuple(result))


def transform_field_renaming(record: StructuredRecord, ) -> StructuredRecord:
    return StructuredRecord(fields=tuple((f"field_{i + 1}", value) for i, (_, value) in enumerate(record.fields)))


def transform_value_normalization(record: StructuredRecord, ) -> StructuredRecord:
    return StructuredRecord(fields=tuple((key, value.lower()) for key, value in record.fields))


def transform_format_conversion(record: StructuredRecord, ) -> str:
    return render_json_object(record.fields)


def transform_value_concatenation(record: StructuredRecord, ) -> str:
    return ", ".join(value for _, value in record.fields)


def transform_structured_classification(record: StructuredRecord, ) -> str:
    has_alpha = any(value == "alpha" for _, value in record.fields)
    return "YES" if has_alpha else "NO"


def transform_key_value_swap(record: StructuredRecord, ) -> StructuredRecord:
    return StructuredRecord(fields=tuple((value, key) for key, value in record.fields))


def transform_field_reversal(record: StructuredRecord, ) -> StructuredRecord:
    return StructuredRecord(fields=tuple(reversed(record.fields)))


def transform_key_prefixing(record: StructuredRecord, ) -> StructuredRecord:
    return StructuredRecord(fields=tuple((f"task_{key}", value) for key, value in record.fields))


def transform_key_suffixing(record: StructuredRecord, ) -> StructuredRecord:
    return StructuredRecord(fields=tuple((f"{key}_task", value) for key, value in record.fields))


def transform_value_prefixing(record: StructuredRecord, ) -> StructuredRecord:
    return StructuredRecord(fields=tuple((key, f"value_{value}") for key, value in record.fields))


def transform_value_suffixing(record: StructuredRecord, ) -> StructuredRecord:
    return StructuredRecord(fields=tuple((key, f"{value}_value") for key, value in record.fields))


COMPOSITION_TRANSFORMS = {"structured_extraction": transform_structured_extraction, "field_filtering": transform_field_filtering, "field_sorting": transform_field_sorting, "value_selection": transform_value_selection, "duplicate_removal": transform_duplicate_removal, "field_renaming": transform_field_renaming, "value_normalization": transform_value_normalization, "format_conversion": transform_format_conversion, "value_concatenation": transform_value_concatenation, "structured_classification": transform_structured_classification, "key_value_swap": transform_key_value_swap, "field_reversal": transform_field_reversal, "key_prefixing": transform_key_prefixing, "key_suffixing": transform_key_suffixing, "value_prefixing": transform_value_prefixing, "value_suffixing": transform_value_suffixing, }

COMPOSITION_RELATIONSHIPS: Dict[Tuple[str, str], str] = {
    # Overlapping
    ("field_filtering", "field_sorting"): "overlapping",
    ("field_sorting", "field_filtering"): "overlapping",
    ("duplicate_removal", "field_sorting"): "overlapping",
    ("field_sorting", "duplicate_removal"): "overlapping",
    ("duplicate_removal", "field_filtering"): "overlapping",
    ("field_filtering", "duplicate_removal"): "overlapping",
    ("field_filtering", "value_normalization"): "overlapping",
    ("value_normalization", "field_filtering"): "overlapping",
    ("field_filtering", "format_conversion"): "overlapping",
    ("duplicate_removal", "format_conversion"): "overlapping",
    ("field_renaming", "field_sorting"): "overlapping",
    ("field_sorting", "field_renaming"): "overlapping",
    ("value_normalization", "field_sorting"): "overlapping",
    ("field_sorting", "value_normalization"): "overlapping",
    ("value_normalization", "value_concatenation"): "overlapping",
    ("value_normalization", "key_value_swap"): "overlapping",
    ("key_value_swap", "value_normalization"): "overlapping",
    ("value_normalization", "field_reversal"): "overlapping",
    ("field_reversal", "value_normalization"): "overlapping",
    ("field_renaming", "field_reversal"): "overlapping",

    # Related
    ("value_selection", "field_sorting"): "related",
    ("field_sorting", "value_selection"): "related",
    ("field_filtering", "value_concatenation"): "related",
    ("field_sorting", "value_concatenation"): "related",
    ("value_selection", "value_concatenation"): "related",
    ("field_sorting", "format_conversion"): "related",
    ("value_selection", "format_conversion"): "related",
    ("field_filtering", "field_reversal"): "related",
    ("field_reversal", "field_filtering"): "related",
    ("value_selection", "field_reversal"): "related",
    ("field_reversal", "value_selection"): "related",
    ("key_value_swap", "field_sorting"): "related",
    ("field_sorting", "key_value_swap"): "related",
    ("key_value_swap", "field_filtering"): "related",
    ("field_filtering", "key_value_swap"): "related",
    ("field_reversal", "value_concatenation"): "related",
    ("key_value_swap", "value_concatenation"): "related",
    ("duplicate_removal", "field_reversal"): "related",
    ("field_reversal", "duplicate_removal"): "related",
    ("value_selection", "key_value_swap"): "related",
    ("key_value_swap", "value_selection"): "related",

    # Highly similar
    ("field_filtering", "value_selection"): "highly_similar",
    ("value_selection", "field_filtering"): "highly_similar",
    ("value_normalization", "format_conversion"): "highly_similar",
    ("duplicate_removal", "value_selection"): "highly_similar",
    ("field_sorting", "field_reversal"): "highly_similar",
    ("field_reversal", "field_sorting"): "highly_similar",
    ("field_renaming", "key_value_swap"): "highly_similar",
    ("key_value_swap", "field_renaming"): "highly_similar",
    ("key_value_swap", "field_reversal"): "highly_similar",
    ("field_reversal", "key_value_swap"): "highly_similar",
    ("duplicate_removal", "key_value_swap"): "highly_similar",
    ("key_value_swap", "duplicate_removal"): "highly_similar",

    # Same-schema atomic transforms. These provide several ordinary,
    # bidirectional composition candidates with independent key/value effects.
    ("key_prefixing", "value_prefixing"): "related",
    ("value_prefixing", "key_prefixing"): "related",
    ("key_suffixing", "value_suffixing"): "related",
    ("value_suffixing", "key_suffixing"): "related",

    # Unrelated (Negative controls)
    ("structured_classification", "field_sorting"): "unrelated",
    ("structured_classification", "field_filtering"): "unrelated",
    ("structured_classification", "value_concatenation"): "unrelated",
    ("structured_classification", "field_reversal"): "unrelated",
    ("structured_classification", "key_value_swap"): "unrelated",
    ("field_filtering", "structured_classification"): "unrelated",
    ("value_selection", "structured_classification"): "unrelated",
    ("key_value_swap", "structured_classification"): "unrelated",
    ("field_reversal", "structured_classification"): "unrelated",
}

COMPOSITION_DESCRIPTIONS: Dict[Tuple[str, str], str] = {("field_filtering", "field_sorting"): "Sort all fields in odd-numbered positions alphabetically by their values. Discard even-numbered positions.", ("field_sorting", "field_filtering"): "Sort all fields alphabetically by their values and return only the odd-positioned fields among the sorted fields.", ("value_selection", "field_sorting"): "Select every field whose value begins with a vowel and sort the selected fields alphabetically by their values.", ("field_sorting", "value_selection"): "Sort all fields alphabetically by their values and select only those fields whose values begin with a vowel.", ("field_filtering", "value_selection"): "From the fields in odd-numbered positions, select only those whose values begin with a vowel.", ("value_selection", "field_filtering"): "From the fields whose values begin with a vowel, return only the odd-numbered positions (1st, 3rd, 5th, etc.).", ("duplicate_removal", "field_sorting"): "Remove duplicate values (keeping the first occurrence) and sort the remaining fields alphabetically by their values.", ("field_sorting", "duplicate_removal"): "Sort all fields alphabetically by their values and remove later duplicate values, keeping the first occurrence.", ("duplicate_removal", "field_filtering"): "Remove duplicate values (keeping the first occurrence) and return only the odd-positioned fields among the remaining fields.", ("field_filtering", "duplicate_removal"): "From the fields in odd-numbered positions, remove duplicate values, keeping the first occurrence.", ("field_renaming", "field_sorting"): "Rename each field key using field_1, field_2, and so on, then sort the fields alphabetically by their values.", ("field_sorting", "field_renaming"): "Sort all fields alphabetically by their values, then rename the field keys in order as field_1, field_2, and so on.", ("field_filtering", "value_concatenation"): "Read the values of the odd-positioned fields from left to right and concatenate them, without field names.", ("field_sorting", "value_concatenation"): "Sort all fields alphabetically by their values and concatenate the values in that order from left to right, without field names.", ("value_selection", "value_concatenation"): "Select every field whose value begins with a vowel and concatenate the selected values from left to right, without field names.", ("value_normalization", "field_sorting"): "Normalize all field values to lowercase and sort the fields alphabetically by their values.", ("field_sorting", "value_normalization"): "Sort all fields alphabetically by their values and normalize all field values to lowercase.", ("field_filtering", "value_normalization"): "Keep only the odd-positioned fields and normalize their values to lowercase.", ("value_normalization", "field_filtering"): "Normalize all field values to lowercase and return only the odd-positioned fields.", ("value_normalization", "value_concatenation"): "Normalize all field values to lowercase and concatenate them from left to right, without field names.", ("field_sorting", "format_conversion"): "Sort all fields alphabetically by their values and convert the record into a JSON object.", ("field_filtering", "format_conversion"): "Keep only the odd-positioned fields and convert the record into a JSON object.", ("value_selection", "format_conversion"): "Select fields whose values begin with a vowel and convert them into a JSON object.", ("duplicate_removal", "format_conversion"): "Remove duplicate values (keeping the first occurrence) and convert the remaining fields into a JSON object.", ("value_normalization", "format_conversion"): "Normalize all field values to lowercase and convert the record into a JSON object.", ("duplicate_removal", "value_selection"): "Remove duplicate values, then select all remaining fields whose values begin with a vowel.", ("field_filtering", "structured_classification"): "Classify the record as YES if at least one odd-positioned field value is exactly 'alpha'; otherwise classify it as NO.", ("structured_classification", "field_sorting"): "Sort all fields alphabetically by their values in ascending order.", ("structured_classification", "field_filtering"): "Filter the record by keeping only fields in odd-numbered positions.", ("structured_classification", "value_concatenation"): "Read the field values from left to right and concatenate them in their original order, without field names.", ("field_filtering", "field_reversal"): "Keep only the fields in odd-numbered positions, then reverse their order from right to left.", ("field_reversal", "field_filtering"): "Reverse the order of all fields from right to left, then keep only the odd-positioned fields among the reversed fields.", ("field_sorting", "field_reversal"): "Sort all fields alphabetically by their values in descending order (reverse alphabetical order).", ("field_reversal", "field_sorting"): "Reverse the order of all fields from right to left, then sort all fields alphabetically by their values.", ("value_selection", "field_reversal"): "Select every field whose value begins with a vowel, then reverse the order of the selected fields from right to left.", ("field_reversal", "value_selection"): "Reverse the order of all fields from right to left, then select only those fields whose values begin with a vowel.", ("duplicate_removal", "field_reversal"): "Remove duplicate values, keeping the first occurrence, then reverse the remaining fields from right to left.", ("field_reversal", "duplicate_removal"): "Reverse the order of all fields from right to left, then remove duplicate values, keeping the first occurrence.", ("field_renaming", "field_reversal"): "Rename each field key to field_1, field_2, and so on, then reverse the order of the fields from right to left.", ("value_normalization", "field_reversal"): "Normalize all field values to lowercase, then reverse the order of the fields from right to left.", ("field_reversal", "value_normalization"): "Reverse the order of all fields from right to left, then normalize all field values to lowercase.", ("field_reversal", "value_concatenation"): "Reverse the order of all fields from right to left, then concatenate their values without field names.", ("field_filtering", "key_value_swap"): "Keep only the fields in odd-numbered positions, and swap the key and value of each retained field.", ("key_value_swap", "field_filtering"): "Swap the key and value of each field, then keep only the odd-positioned fields.", ("field_sorting", "key_value_swap"): "Sort the fields alphabetically by their values, then swap the key and value of each sorted field.", ("key_value_swap", "field_sorting"): "Swap the key and value of each field, then sort the resulting fields alphabetically by their values.", ("value_selection", "key_value_swap"): "Select every field whose value begins with a vowel, then swap the key and value of each selected field.", ("key_value_swap", "value_selection"): "Swap the key and value of each field, then select only those fields whose values begin with a vowel.", ("duplicate_removal", "key_value_swap"): "Remove duplicate values, keeping the first occurrence, then swap the key and value of each remaining field.", ("key_value_swap", "duplicate_removal"): "Swap the key and value of each field, then remove duplicate values, keeping the first occurrence.", ("field_renaming", "key_value_swap"): "Rename each field key to field_1, field_2, and so on, then swap the key and value of each field.", ("key_value_swap", "field_renaming"): "Swap the key and value of each field, then rename the resulting keys in order as field_1, field_2, and so on.", ("value_normalization", "key_value_swap"): "Normalize all field values to lowercase, then swap the key and value of each field.", ("key_value_swap", "value_normalization"): "Swap the key and value of each field, then normalize all field values to lowercase.", ("key_value_swap", "value_concatenation"): "Swap the key and value of each field, then concatenate the resulting values in order without field names.", ("key_value_swap", "field_reversal"): "Swap the key and value of each field, then reverse their order from right to left.", ("field_reversal", "key_value_swap"): "Reverse the order of all fields from right to left, then swap the key and value of each field.", ("structured_classification", "field_reversal"): "Reverse the sequence of fields in the record from right to left.", ("structured_classification", "key_value_swap"): "Swap the key and value of each field in the record.", ("value_selection", "structured_classification"): "Select fields whose values begin with a vowel, then classify as YES if any value is 'alpha', otherwise NO.", ("key_value_swap", "structured_classification"): "Swap the key and value of each field, then classify as YES if any field value is 'alpha', otherwise NO.", ("field_reversal", "structured_classification"): "Reverse the sequence of fields, then classify as YES if any field value is 'alpha', otherwise NO.", }

COMPOSITION_DESCRIPTIONS.update({
    ("key_prefixing", "value_prefixing"): "Add task_ before every field key, then add value_ before every field value.",
    ("value_prefixing", "key_prefixing"): "Add value_ before every field value, then add task_ before every field key.",
    ("key_suffixing", "value_suffixing"): "Add _task after every field key, then add _value after every field value.",
    ("value_suffixing", "key_suffixing"): "Add _value after every field value, then add _task after every field key.",
})

PEFT_COMBINATION_TYPES = {"additive": "linear", "linear": "linear", "cat": "cat", "svd": "svd", "ties": "ties", "ties_svd": "ties_svd", "dare_linear": "dare_linear", "dare_ties": "dare_ties", "dare_linear_svd": "dare_linear_svd", "dare_ties_svd": "dare_ties_svd", "magnitude_prune": "magnitude_prune", "magnitude_prune_svd": "magnitude_prune_svd", }
SEQUENTIAL_COMPOSITION_METHODS = {"sequential"}
ROUTED_COMPOSITION_METHODS = {"routed"}


def build_composite_prompt(composition: CompositionSpec, body: str, output_format: str, noise_pattern: str, ) -> str:
    key = (composition.skill_a, composition.skill_b)
    if key not in COMPOSITION_DESCRIPTIONS:
        raise ValueError(f"No audited composite prompt exists for {key}")
    operation = COMPOSITION_DESCRIPTIONS[key]
    empty_rule = " If no field qualifies, return NONE." if "value_selection" in key else ""
    if composition.skill_b == "format_conversion":
        format_instruction = "Return only the JSON object, with each key represented as a JSON string and each value represented as a JSON string.\n"
    elif composition.skill_b == "structured_classification":
        format_instruction = "Return exactly one label: YES or NO.\n"
    elif composition.skill_b == "value_concatenation":
        sep_desc = {"comma": "commas", "pipe": "vertical bars (|)", "space": "spaces", "lines": "new lines"}.get(output_format)
        if sep_desc is None:
            raise ValueError(f"Unsupported value_concatenation output format: {output_format}")
        format_instruction = f"Separate the values using {sep_desc}.\n"
    else:
        format_instruction = (f"Format the answer as {output_format}. "
                              f"For 'lines', output one 'key: value' pair per line. "
                              f"For 'comma', output 'key=value' pairs separated by commas.\n")

    return (f"{NOISE_PATTERNS[noise_pattern]}"
            f"{operation}{empty_rule}\n"
            f"Record: {body}\n"
            f"{format_instruction}"
            f"Answer:")


def apply_composition_sampling_constraints(fields: List[Tuple[str, str]], composition: CompositionSpec, sampling: Dict[str, Any], example_index: int, ) -> List[Tuple[str, str]]:
    """Apply the controlled sampling rule for selection followed by sorting."""
    pair = (composition.skill_a, composition.skill_b)
    if pair != ("value_selection", "field_sorting"):
        return fields
    require_unselected = bool(sampling.get("require_unselected_field", True))
    selected_candidates = [int(value) for value in sampling.get("selected_counts", [2, 3])]
    if not selected_candidates or min(selected_candidates) < 2:
        raise ValueError("composite_sampling.selected_counts must contain values >= 2")
    maximum = len(fields) - (1 if require_unselected else 0)
    selected_count = selected_candidates[example_index % len(selected_candidates)]
    selected_count = min(selected_count, maximum)
    if selected_count < 2:
        raise ValueError(f"field_count={len(fields)} cannot realize at least two selected fields with the configured unselected-field requirement")

    # Reverse lexical order makes sorting observably change the intermediate
    # record.  Rotate the vocabulary across examples while preserving that
    # property and use consonant-initial values for the remaining fields.
    vowels = sorted(VOWEL_INITIAL_VALUES)
    offset = example_index % len(vowels)
    selected_vowels = [vowels[(offset + j) % len(vowels)] for j in range(selected_count)]
    ordered_vowels = sorted(selected_vowels, reverse=True)
    consonants = sorted(CONSONANT_INITIAL_VALUES)
    result_values = ordered_vowels + [consonants[(offset + j) % len(consonants)] for j in range(len(fields) - selected_count)]
    return [(key, result_values[index]) for index, (key, _) in enumerate(fields)]


def generate_composite_examples(composition: CompositionSpec, split: str, n: int, seed: int, latent_overrides: Dict[str, Any] | None = None, sampling_config: Dict[str, Any] | None = None, ) -> List[CompositeExample]:
    rng = deterministic_rng(seed)
    overrides = latent_overrides or {}
    sampling = sampling_config or {}
    if composition.skill_a not in COMPOSITION_TRANSFORMS:
        raise ValueError(f"No composition transform registered for {composition.skill_a}")
    if composition.skill_b not in COMPOSITION_TRANSFORMS:
        raise ValueError(f"No composition transform registered for {composition.skill_b}")
    field_counts = balanced_sequence([int(value) for value in sampling.get("field_counts", [3, 4, 5])], n, rng, )
    value_offsets = balanced_sequence([int(value) for value in sampling.get("value_offsets", [0, 1, 2, 3, 4])], n, rng, )
    noise_patterns = balanced_sequence(list(NOISE_PATTERNS.keys()), n, rng, )
    configured_formats = list(sampling.get("output_formats", (["comma", "pipe"] if composition.skill_b == "value_concatenation" else ["lines", "comma"])))
    if composition.skill_b == "value_concatenation":
        invalid_formats = set(configured_formats) - {"comma", "pipe", "space", "lines"}
        if invalid_formats:
            raise ValueError(f"Unsupported value_concatenation output formats: {sorted(invalid_formats)}")
    output_formats = balanced_sequence(configured_formats, n, rng, )
    records = []
    for i in range(n):
        field_count = overrides.get("field_count", field_counts[i], )
        value_offset = overrides.get("value_offset", value_offsets[i], )
        noise_pattern = overrides.get("noise_pattern", noise_patterns[i], )
        output_format = overrides.get("output_format", output_formats[i], )
        fields = generate_key_value_fields(field_count=field_count, rng=rng, value_offset=value_offset + i, )
        if "duplicate_removal" in (composition.skill_a, composition.skill_b) and len(fields) >= 3:
            # Make duplicate removal meaningful while preserving deterministic data.
            fields[-1] = (fields[-1][0], fields[0][1])
        if (composition.skill_a, composition.skill_b) == ("duplicate_removal", "field_filtering") and len(fields) >= 3:
            # Put the duplicate in a field retained by the downstream filter.
            fields[2] = (fields[2][0], fields[0][1])
        if (composition.skill_a, composition.skill_b) == ("field_filtering", "value_selection"):
            vowel_offset = (i + int(value_offset)) % len(VOWEL_INITIAL_VALUES)
            consonant_offset = (i + int(value_offset)) % len(CONSONANT_INITIAL_VALUES)
            fields[0] = (fields[0][0], VOWEL_INITIAL_VALUES[vowel_offset])
            fields[1] = (fields[1][0], VOWEL_INITIAL_VALUES[(vowel_offset + 1) % len(VOWEL_INITIAL_VALUES)])
            fields[2] = (fields[2][0], CONSONANT_INITIAL_VALUES[consonant_offset])
            if len(fields) >= 4:
                fields[3] = (fields[3][0], CONSONANT_INITIAL_VALUES[(consonant_offset + 1) % len(CONSONANT_INITIAL_VALUES)])
            if len(fields) >= 5:
                fields[4] = (fields[4][0], VOWEL_INITIAL_VALUES[(vowel_offset + 2) % len(VOWEL_INITIAL_VALUES)] if i % 2 else CONSONANT_INITIAL_VALUES[(consonant_offset + 2) % len(CONSONANT_INITIAL_VALUES)])
        elif (composition.skill_a, composition.skill_b) == ("value_selection", "field_filtering"):
            vowel_offset = (i + int(value_offset)) % len(VOWEL_INITIAL_VALUES)
            consonant_offset = (i + int(value_offset)) % len(CONSONANT_INITIAL_VALUES)
            fields[0] = (fields[0][0], VOWEL_INITIAL_VALUES[vowel_offset])
            fields[1] = (fields[1][0], CONSONANT_INITIAL_VALUES[consonant_offset])
            fields[2] = (fields[2][0], VOWEL_INITIAL_VALUES[(vowel_offset + 1) % len(VOWEL_INITIAL_VALUES)])
            if len(fields) >= 4:
                fields[3] = (fields[3][0], CONSONANT_INITIAL_VALUES[(consonant_offset + 1) % len(CONSONANT_INITIAL_VALUES)])
        elif composition.skill_a == "field_sorting":
            forced_values = ["india", "bravo", "echo", "delta", "alpha"]
            fields = [(key, forced_values[index % len(forced_values)]) for index, (key, _) in enumerate(fields)]
            if composition.skill_b == "duplicate_removal" and len(fields) >= 3:
                fields[-1] = (fields[-1][0], fields[0][1])
        fields = apply_composition_sampling_constraints(fields=fields, composition=composition, sampling=sampling, example_index=i)
        if "value_normalization" in (composition.skill_a, composition.skill_b):
            # Atomic examples use lowercase values, which makes normalization
            # an identity operation in a composite and causes the downstream
            # adapter to solve the task by itself.  Add deterministic casing
            # variation only to composite inputs; the normalization transform
            # then has observable work to do while preserving the same value
            # vocabulary and latent-OOD structure.
            fields = [
                (key, value.upper() if (index + i) % 2 == 0 else value.title())
                for index, (key, value) in enumerate(fields)
            ]
        record = StructuredRecord(fields=tuple(fields))
        transform_a = COMPOSITION_TRANSFORMS[composition.skill_a]
        transform_b = COMPOSITION_TRANSFORMS[composition.skill_b]
        intermediate = transform_a(record)
        final = transform_b(intermediate if isinstance(intermediate, StructuredRecord) else record)
        if isinstance(final, StructuredRecord):
            target = render_fields(final.fields, output_format, ) if final.fields else "NONE"
        elif composition.skill_b == "value_concatenation":
            values = [value for _, value in intermediate.fields] if isinstance(intermediate, StructuredRecord) else [str(final)]
            target = render_values(values, output_format)
        else:
            target = str(final)
        body = " | ".join(f"{key}={value}" for key, value in fields)
        prompt = build_composite_prompt(composition=composition, body=body, output_format=output_format, noise_pattern=noise_pattern, )
        final_schema = ("structured_record" if isinstance(final, StructuredRecord) else "json_object" if composition.skill_b == "format_conversion" else "label" if composition.skill_b == "structured_classification" else "value_sequence")
        intermediate_fields = intermediate.fields if isinstance(intermediate, StructuredRecord) else ()
        final_fields = final.fields if isinstance(final, StructuredRecord) else ()
        second_operation_changes_order = bool(isinstance(intermediate, StructuredRecord) and isinstance(final, StructuredRecord) and intermediate.fields != final.fields and sorted(intermediate.fields) == sorted(final.fields))
        records.append(CompositeExample(composition_id=composition.composition_id, split=split, latent={"input_schema": "key_value_record", "intermediate_schema": "structured_record", "final_schema": final_schema, "field_count": field_count, "value_offset": value_offset, "noise_pattern": noise_pattern, "output_format": output_format, "skill_a": composition.skill_a, "skill_b": composition.skill_b, "relationship_type": COMPOSITION_RELATIONSHIPS[(composition.skill_a, composition.skill_b)], "is_negative_control": COMPOSITION_RELATIONSHIPS[(composition.skill_a, composition.skill_b)] == "unrelated", "intermediate_field_count": len(intermediate_fields), "target_field_count": len(final_fields), "second_operation_changes_order": second_operation_changes_order, "sampling_mode": "controlled" if (composition.skill_a, composition.skill_b) == ("value_selection", "field_sorting") else "default", "skill_parameters": {"field_count": field_count, "value_offset": value_offset, "output_format": output_format, }, }, prompt=prompt, target=target, intermediate=intermediate, source=record, skill_a=composition.skill_a, skill_b=composition.skill_b, ))
    return records


@dataclass(frozen=True)
class CompositionSpec:
    composition_id: str
    skill_a: str
    skill_b: str
    composition_method: str
    weight_a: float
    weight_b: float
    ordered: bool
    generator_version: str
    adapter_a: str = ""
    adapter_b: str = ""
    base_model: str = ""
    model_revision: str = "unknown"
    tokenizer_revision: str = "unknown"
    relationship_type: str = ""


@dataclass(frozen=True)
class CompositeExample:
    composition_id: str
    split: str
    latent: Dict[str, Any]
    prompt: str
    target: str
    intermediate: Any
    source: StructuredRecord
    skill_a: str
    skill_b: str


def make_composition_id(skill_a: str, skill_b: str, method: str, weight_a: float, weight_b: float, ) -> str:
    return (f"{skill_a}__{skill_b}"
            f"__{method}"
            f"__wa{weight_a:g}"
            f"_wb{weight_b:g}")


def build_composition_spec(skill_a: str, skill_b: str, method: str, weight_a: float = 0.5, weight_b: float = 0.5, config: Dict[str, Any] | None = None, adapter_a: str = "", adapter_b: str = "", ) -> CompositionSpec:
    get_skill(skill_a)
    get_skill(skill_b)
    supported_methods = set(PEFT_COMBINATION_TYPES) | SEQUENTIAL_COMPOSITION_METHODS | ROUTED_COMPOSITION_METHODS
    if method not in supported_methods:
        raise ValueError(f"Unsupported composition method: {method}. Supported methods: {sorted(supported_methods)}")
    if (skill_a, skill_b) not in COMPOSITION_RELATIONSHIPS:
        reverse = (skill_b, skill_a)
        if reverse in COMPOSITION_RELATIONSHIPS:
            raise ValueError(f"Unsupported ordered composition: {skill_a}:{skill_b}. The reverse order {skill_b}:{skill_a} is audited; composition order is significant because {skill_a} does not produce the structured-record schema required by {skill_b}.")
        raise ValueError(f"Unsupported or unaudited composition: {skill_a}:{skill_b}")
    model_name = str((config or {}).get("model", {}).get("name", ""))
    return CompositionSpec(composition_id=make_composition_id(skill_a=skill_a, skill_b=skill_b, method=method, weight_a=weight_a, weight_b=weight_b, ), skill_a=skill_a, skill_b=skill_b, composition_method=method, weight_a=float(weight_a), weight_b=float(weight_b), ordered=True, generator_version="v002", adapter_a=adapter_a, adapter_b=adapter_b, base_model=model_name, model_revision=str((config or {}).get("model", {}).get("revision", "unknown")), tokenizer_revision=str((config or {}).get("model", {}).get("tokenizer_revision", "unknown")), relationship_type=COMPOSITION_RELATIONSHIPS[(skill_a, skill_b)], )


def generate_ordered_pairs(skills: Sequence[str], include_self_pairs: bool = False, ) -> List[Tuple[str, str]]:
    pairs = []
    for skill_a in skills:
        for skill_b in skills:
            if not include_self_pairs and skill_a == skill_b:
                continue
            if (skill_a, skill_b) in COMPOSITION_RELATIONSHIPS:
                pairs.append((skill_a, skill_b))
    return pairs


###############################
#           SKILLS            #
# #############################


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    family: str
    relationship_group: str
    description: str
    generator: str
    generator_version: str
    latent_schema: Tuple[str, ...]


LATENT_OOD_SPECS: Dict[str, Dict[str, Any]] = {"structured_extraction": {"overrides": {"field_count": 5, "value_offset": 7, }, "ood_parameters": ("field_count", "value_offset", ), }, "structured_classification": {"overrides": {"field_count": 5, "value_offset": 7, }, "ood_parameters": ("field_count", "value_offset", ), }, "field_filtering": {"overrides": {"value_offset": 7, }, "ood_parameters": ("value_offset", ), }, "format_conversion": {"overrides": {"value_offset": 7, }, "ood_parameters": ("value_offset", ), }, "field_sorting": {"overrides": {"field_count": 6, "value_offset": 7, }, "ood_parameters": ("field_count", "value_offset", ), }, "value_normalization": {"overrides": {"field_count": 5, "value_offset": 7, }, "ood_parameters": ("field_count", "value_offset", ), }, "field_renaming": {"overrides": {"field_count": 5, "value_offset": 7, }, "ood_parameters": ("field_count", "value_offset", ), }, "duplicate_removal": {"overrides": {"field_count": 7, "duplicate_count": 3, }, "ood_parameters": ("field_count", "duplicate_count", ), }, "value_selection": {"overrides": {"field_count": 6, "value_offset": 7, }, "ood_parameters": ("field_count", "value_offset", ), }, "value_concatenation": {"overrides": {"value_offset": 7, }, "ood_parameters": ("value_offset", ), }, "key_value_swap": {"overrides": {"field_count": 5, "value_offset": 7, }, "ood_parameters": ("field_count", "value_offset", ), }, "field_reversal": {"overrides": {"field_count": 6, "value_offset": 7, }, "ood_parameters": ("field_count", "value_offset", ), }, }

SKILL_REGISTRY: Dict[str, SkillSpec] = {"structured_extraction": SkillSpec(skill_id="structured_extraction", family="extraction", relationship_group="extraction", description=("Extract every key-value field from a structured record."), generator="structured_extraction", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "output_format", "value_offset")), "structured_classification": SkillSpec(skill_id="structured_classification", family="classification", relationship_group="classification", description=("Classify a structured record as yes/no according to whether any field has the value alpha."), generator="structured_classification", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "rule", "label_format", "value_offset", "class")), "field_sorting": SkillSpec(skill_id="field_sorting", family="ordering", relationship_group="transformation", description=("Sort fields alphabetically by their values."), generator="field_sorting", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "output_format", "value_offset", "sort_order")), "field_filtering": SkillSpec(skill_id="field_filtering", family="filtering", relationship_group="filtering", description=("Select the odd-positioned fields from a structured record."), generator="field_filtering", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "filter_rule", "output_format", "value_offset")), "format_conversion": SkillSpec(skill_id="format_conversion", family="transformation", relationship_group="transformation", description=("Convert a key-value record into a JSON object."), generator="format_conversion", generator_version="v003", latent_schema=("input_schema", "field_count", "noise_pattern", "source_format", "target_format", "value_offset")), "value_normalization": SkillSpec(skill_id="value_normalization", family="transformation", relationship_group="normalization", description="Normalize values according to a deterministic mapping.", generator="value_normalization", generator_version="v001", latent_schema=tuple(LATENT_OOD_SPECS["value_normalization"])), "field_renaming": SkillSpec(skill_id="field_renaming", family="transformation", relationship_group="transformation", description=("Rename alphabetic field keys to indexed field names."), generator="field_renaming", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "output_format", "value_offset", "rename_scheme")), "duplicate_removal": SkillSpec(skill_id="duplicate_removal", family="deduplication", relationship_group="filtering", description=("Remove later fields whose values have already appeared."), generator="duplicate_removal", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "duplicate_count", "deduplication")), "value_selection": SkillSpec(skill_id="value_selection", family="filtering", relationship_group="filtering", description=("Select fields whose values begin with a vowel."), generator="value_selection", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "value_offset", "selection_rule")), "value_concatenation": SkillSpec(skill_id="value_concatenation", family="aggregation", relationship_group="transformation", description=("Concatenate field values in their original order."), generator="value_concatenation", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "value_offset", "separator", "operation")), "key_value_swap": SkillSpec(skill_id="key_value_swap", family="transformation", relationship_group="transformation", description=("Swap key and value of each field in a structured record."), generator="key_value_swap", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "output_format", "value_offset", "operation")), "field_reversal": SkillSpec(skill_id="field_reversal", family="ordering", relationship_group="transformation", description=("Reverse the order of fields in a structured record from right to left."), generator="field_reversal", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "output_format", "value_offset", "operation")), }

SKILL_REGISTRY.update({
    "key_prefixing": SkillSpec(skill_id="key_prefixing", family="transformation", relationship_group="transformation", description="Add a fixed prefix to every field key.", generator="key_prefixing", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "output_format", "value_offset", "target", "affix", "operation")),
    "key_suffixing": SkillSpec(skill_id="key_suffixing", family="transformation", relationship_group="transformation", description="Add a fixed suffix to every field key.", generator="key_suffixing", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "output_format", "value_offset", "target", "affix", "operation")),
    "value_prefixing": SkillSpec(skill_id="value_prefixing", family="transformation", relationship_group="transformation", description="Add a fixed prefix to every field value.", generator="value_prefixing", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "output_format", "value_offset", "target", "affix", "operation")),
    "value_suffixing": SkillSpec(skill_id="value_suffixing", family="transformation", relationship_group="transformation", description="Add a fixed suffix to every field value.", generator="value_suffixing", generator_version="v001", latent_schema=("input_schema", "field_count", "noise_pattern", "output_format", "value_offset", "target", "affix", "operation")),
})

# Dataset identity must change when a generator's semantics change; otherwise
# stale atomic bundles can be mistaken for data produced by the fixed code.
SKILL_REGISTRY["value_normalization"] = replace(SKILL_REGISTRY["value_normalization"], generator_version="v002")
SKILL_REGISTRY["field_reversal"] = replace(SKILL_REGISTRY["field_reversal"], generator_version="v002")

# Keep field-count extrapolation explicit: six fields are now in training and
# seven fields are the held-out latent regime.
LATENT_OOD_SPECS["field_reversal"] = {"overrides": {"field_count": 7, "value_offset": 7}, "ood_parameters": ("field_count", "value_offset")}
LATENT_OOD_SPECS.update({
    "key_prefixing": {"overrides": {"field_count": 7, "value_offset": 7}, "ood_parameters": ("field_count", "value_offset")},
    "key_suffixing": {"overrides": {"field_count": 7, "value_offset": 7}, "ood_parameters": ("field_count", "value_offset")},
    "value_prefixing": {"overrides": {"field_count": 7, "value_offset": 7}, "ood_parameters": ("field_count", "value_offset")},
    "value_suffixing": {"overrides": {"field_count": 7, "value_offset": 7}, "ood_parameters": ("field_count", "value_offset")},
})


def get_skill(skill_id: str) -> SkillSpec:
    try:
        return SKILL_REGISTRY[skill_id]
    except KeyError:
        raise ValueError(f"Unknown skill_id={skill_id}. Available skills: {sorted(SKILL_REGISTRY)}")


###############################
#          TRAINING           #
# #############################


def generate_latent_ood(skill: SkillSpec, generator, n: int, seed: int, ) -> List[Example]:
    spec = LATENT_OOD_SPECS.get(skill.skill_id)
    if spec is None:
        raise ValueError(f"No latent-OOD specification for {skill.skill_id}")
    return generator(task_id=skill.skill_id, split="test_latent", n=n, seed=seed, latent_overrides=dict(spec["overrides"]), )


def generate_skill_dataset(skill_id: str, split_sizes: Dict[str, int], seed: int, test_sizes: Dict[str, int], ) -> Dict[str, List[Example]]:
    skill = get_skill(skill_id)
    if skill.generator not in GENERATOR_REGISTRY: raise ValueError(f"No generator registered for {skill.generator}")
    generator = GENERATOR_REGISTRY[skill.generator]
    data: Dict[str, List[Example]] = {}
    split_seed_offsets = {"train": 0, "validation": 10_000, }
    for split, n in split_sizes.items():
        if split not in {"train", "validation"}: raise ValueError(f"Unexpected atomic split: {split}. Only train and validation are expected.")
        data[split] = generator(task_id=skill_id, split=split, n=n, seed=seed + split_seed_offsets[split], )
    data["test_iid"] = generator(task_id=skill_id, split="test_iid", n=int(test_sizes["iid"]), seed=seed + 40_000, )
    data["test_latent"] = generate_latent_ood(skill, generator, n=int(test_sizes["latent"]), seed=seed + 50_000, )
    return data


def generate_composite_dataset(composition: CompositionSpec, config: Dict[str, Any], seed: int, ) -> Dict[str, List[CompositeExample]]:
    sizes = config["composition"]["composite_dataset_sizes"]
    training_data = dict(config["composition"].get("composite_training_data", {}))
    sampling_config = config["composition"].get("composite_sampling", {})
    training_data.update(sampling_config)
    split_offsets = {"train": 0, "validation": 10_000, "iid": 20_000, "latent": 30_000, "composition_ood": 40_000, }
    data = {}
    data["train"] = generate_composite_examples(composition=composition, split="train", n=int(sizes["train"]), seed=seed + split_offsets["train"], sampling_config=training_data)
    data["validation"] = generate_composite_examples(composition=composition, split="validation", n=int(sizes["validation"]), seed=seed + split_offsets["validation"], sampling_config=training_data)
    data["test_iid"] = generate_composite_examples(composition=composition, split="test_iid", n=int(sizes["iid"]), seed=seed + split_offsets["iid"], sampling_config=training_data, )
    # Training uses field counts 3--6. Keep latent and composition-OOD on
    # disjoint held-out regimes rather than evaluating latent examples on a
    # field count already present in composite training.
    data["test_latent"] = generate_composite_examples(composition=composition, split="test_latent", n=int(sizes["latent"]), seed=seed + split_offsets["latent"], latent_overrides={"field_count": 7, "value_offset": 7}, sampling_config=training_data, )
    data["test_composition_ood"] = generate_composite_examples(composition=composition, split="test_composition_ood", n=int(sizes["composition_ood"]), seed=seed + split_offsets["composition_ood"], latent_overrides={"field_count": 8, "value_offset": 9, "output_format": "comma"}, sampling_config=training_data, )
    if (composition.skill_a, composition.skill_b) == ("value_selection", "field_sorting"):
        mode_counts = {split: Counter(example.latent.get("sampling_mode") for example in examples) for split, examples in data.items()}
        nontrivial_counts = {split: sum(bool(example.latent.get("second_operation_changes_order")) for example in examples) for split, examples in data.items()}
        LOG.info("CONTROLLED COMPOSITION AUDIT | %s | modes=%s | nontrivial_sort_counts=%s", composition.composition_id, mode_counts, nontrivial_counts)
        if any(counts != Counter({"controlled": len(data[split])}) for split, counts in mode_counts.items()):
            raise ValueError("Controlled sampling was not applied to every composite split")
        if any(nontrivial_counts[split] != len(data[split]) for split in data):
            raise ValueError("Controlled sampling produced a vacuous sorting example")
    validate_composite_dataset(dataset=data, composition=composition, )
    return data


def save_dataset_bundle(dataset_dir: Path, dataset: Dict[str, List[Example]], config: Dict[str, Any], ) -> Dict[str, Any]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": now_ts(), "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(), "splits": {}, }
    for split, examples in dataset.items():
        path = dataset_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"count": len(examples), "file": str(path), "sha256": sha256_file(path), }
    write_json(dataset_dir / "manifest.json", manifest)
    LOG.info("Dataset saved to %s", dataset_dir)
    return manifest


def save_composite_dataset_bundle(dataset_dir: Path, dataset: Dict[str, List[CompositeExample]], config: Dict[str, Any], ) -> Dict[str, Any]:
    dataset_dir.mkdir(parents=True, exist_ok=True, )
    manifest = {"created_at": now_ts(), "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True, ).encode()).hexdigest(), "splits": {}, }
    for split, examples in dataset.items():
        path = dataset_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", ) as f:
            for example in examples:
                record = asdict(example)
                f.write(json.dumps(record, ensure_ascii=False, ) + "\n")
        manifest["splits"][split] = {"count": len(examples), "file": str(path), "sha256": sha256_file(path), }
    write_json(dataset_dir / "manifest.json", manifest, )
    return manifest


def build_skill_qualification(skill: SkillSpec, evaluation: Dict[str, Any], training_metrics: Dict[str, Any], ) -> Dict[str, Any]:
    base_iid = evaluation["base"]["test_iid"]["accuracy"]
    lora_iid = evaluation["adapter"]["test_iid"]["accuracy"]
    base_latent = evaluation["base"]["test_latent"]["accuracy"]
    lora_latent = evaluation["adapter"]["test_latent"]["accuracy"]
    return {"skill_id": skill.skill_id, "family": skill.family, "relationship_group": skill.relationship_group, "generator": skill.generator, "generator_version": skill.generator_version, "latent_schema": list(skill.latent_schema), "base_test_iid_accuracy": base_iid, "adapter_test_iid_accuracy": lora_iid, "iid_gain_over_base": lora_iid - base_iid, "base_test_latent_accuracy": base_latent, "adapter_test_latent_accuracy": lora_latent, "latent_gain_over_base": lora_latent - base_latent, "training_time_seconds": training_metrics.get("wall_time_seconds"), "train_examples": training_metrics.get("train_examples"), "validation_examples": training_metrics.get("validation_examples"), }


def load_model_and_tokenizer(model_name: str):
    LOG.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, )
    model.config.use_cache = False
    return model, tokenizer


def format_chat_prompt(prompt: str, tokenizer, ) -> str:
    messages = [{"role": "user", "content": prompt, }]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, )


def tokenize_training_example(example: Any, tokenizer, max_seq_length: int, ) -> Dict[str, List[int]]:
    formatted_prompt = format_chat_prompt(example.prompt, tokenizer, )
    prompt_ids = tokenizer(formatted_prompt, add_special_tokens=False, truncation=False, )["input_ids"]
    target_ids = tokenizer(example.target, add_special_tokens=False, truncation=False, )["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    if len(target_ids) >= max_seq_length:
        target_ids = target_ids[:max_seq_length]
        if tokenizer.eos_token_id is not None:
            target_ids[-1] = tokenizer.eos_token_id
        prompt_ids = []
    else:
        available_prompt_tokens = (max_seq_length - len(target_ids))
        prompt_ids = prompt_ids[:available_prompt_tokens]
    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = ([-100] * len(prompt_ids) + target_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, }


def tokenize_dataset(examples: Sequence[Any], tokenizer, max_seq_length: int, ) -> Dataset:
    rows = [tokenize_training_example(example=e, tokenizer=tokenizer, max_seq_length=max_seq_length, ) for e in examples]
    return Dataset.from_dict({"input_ids": [row["input_ids"] for row in rows], "attention_mask": [row["attention_mask"] for row in rows], "labels": [row["labels"] for row in rows], })


def build_lora_model(model, cfg: Dict[str, Any]):
    lora = cfg["lora"]
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=int(lora["rank"]), lora_alpha=int(lora["alpha"]), lora_dropout=float(lora["dropout"]), target_modules=list(lora["target_modules"]), bias="none", )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def train_adapter(config: Dict[str, Any], dataset: Dict[str, List[Any]], output_dir: Path, seed: int, ) -> Dict[str, Any]:
    set_global_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = config["model"]["name"]
    model, tokenizer = load_model_and_tokenizer(model_name)
    model = build_lora_model(model, config)
    max_len = int(config["training"]["max_seq_length"])
    train_ds = tokenize_dataset(dataset["train"], tokenizer, max_len)
    val_ds = tokenize_dataset(dataset["validation"], tokenizer, max_len)
    training_data_diagnostics = dataset_diagnostics(dataset, tokenizer)
    write_json(output_dir / "training_data_diagnostics.json", training_data_diagnostics)
    write_json(output_dir / "tokenizer_signature.json", tokenizer_signature(tokenizer))
    write_json(output_dir / "adapter_state_before_training.json", adapter_state_summary(model))
    write_json(output_dir / "training_configuration.json", {"seed": seed, "model": model_name, "training": config["training"], "lora": config["lora"], "train_tokenized_rows": len(train_ds), "validation_tokenized_rows": len(val_ds), "max_seq_length": max_len})
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt", )
    t = config["training"]
    eval_strategy = str(t.get("eval_strategy", "steps"))
    save_strategy = str(t.get("save_strategy", eval_strategy))
    load_best = bool(t.get("load_best_model_at_end", True)) and eval_strategy != "no" and save_strategy == eval_strategy
    if eval_strategy == "steps" and int(t.get("eval_steps", 0)) < 1:
        raise ValueError("training.eval_steps must be >= 1 when eval_strategy=steps")
    if save_strategy == "steps" and int(t.get("save_steps", 0)) < 1:
        raise ValueError("training.save_steps must be >= 1 when save_strategy=steps")
    LOG.info("TRAINING SCHEDULE | eval_strategy=%s eval_steps=%s save_strategy=%s save_steps=%s load_best=%s", eval_strategy, t.get("eval_steps"), save_strategy, t.get("save_steps"), load_best)
    args = TrainingArguments(output_dir=str(output_dir / "trainer"), overwrite_output_dir=True, seed=seed, data_seed=seed, num_train_epochs=float(t["num_train_epochs"]), per_device_train_batch_size=int(t["per_device_train_batch_size"]), per_device_eval_batch_size=int(t["per_device_eval_batch_size"]), gradient_accumulation_steps=int(t["gradient_accumulation_steps"]), learning_rate=float(t["learning_rate"]), weight_decay=float(t["weight_decay"]), warmup_ratio=float(t["warmup_ratio"]), logging_steps=int(t["logging_steps"]), eval_strategy=eval_strategy, eval_steps=(int(t["eval_steps"]) if eval_strategy == "steps" else None), save_strategy=save_strategy, save_steps=(int(t["save_steps"]) if save_strategy == "steps" else None), save_total_limit=1, fp16=torch.cuda.is_available(), report_to=[], remove_unused_columns=False, gradient_checkpointing=bool(t["gradient_checkpointing"]), optim=t["optimizer"], load_best_model_at_end=load_best, metric_for_best_model="eval_loss", greater_is_better=False, )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, data_collator=collator, )
    LOG.info("Starting LoRA training")
    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started
    LOG.info("Training complete in %.1f seconds", elapsed)
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metrics = dict(result.metrics)
    metrics["wall_time_seconds"] = elapsed
    metrics["train_examples"] = len(dataset["train"])
    metrics["validation_examples"] = len(dataset["validation"])
    metrics["trainer_state"] = {"global_step": int(trainer.state.global_step), "epoch": trainer.state.epoch, "best_model_checkpoint": trainer.state.best_model_checkpoint, "best_metric": trainer.state.best_metric, "log_history_entries": len(trainer.state.log_history)}
    write_json(output_dir / "trainer_log_history.json", trainer.state.log_history)
    write_json(output_dir / "training_metrics.json", metrics)
    write_json(output_dir / "adapter_state_after_training.json", adapter_state_summary(model))
    write_json(output_dir / "adapter_files.json", directory_file_manifest(output_dir / "adapter"))
    return {"adapter_dir": str(adapter_dir), "training_metrics": metrics, }


def train_composite_adapter(config: Dict[str, Any], dataset: Dict[str, List[CompositeExample]], output_dir: Path, seed: int, ) -> Dict[str, Any]:
    """Train the dedicated A+B reference adapter using the same pipeline as atomic."""
    return train_adapter(config=config, dataset=dataset, output_dir=output_dir, seed=seed)


###############################
#         EVALUATION          #
# #############################


def load_base_model_for_eval(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    LOG.info("Base-model evaluation device: %s", device)
    return model, tokenizer


def load_adapter_for_eval(model_name: str, adapter_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True, )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (torch.float16 if torch.cuda.is_available() else torch.float32)
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, )
    model = PeftModel.from_pretrained(base, adapter_dir, )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    LOG.info("Evaluation device: %s", device)
    if device.type == "cuda":
        LOG.info("GPU: %s", torch.cuda.get_device_name(0), )
        LOG.info("GPU memory allocated: %.2f GB", torch.cuda.memory_allocated() / 1024**3, )
    return model, tokenizer


def load_composed_adapter_for_eval(model_name: str, adapter_a_dir: Path, adapter_b_dir: Path, composition: CompositionSpec, ):
    if composition.composition_method not in set(PEFT_COMBINATION_TYPES) | SEQUENTIAL_COMPOSITION_METHODS | ROUTED_COMPOSITION_METHODS:
        raise ValueError(f"Unsupported composition method: {composition.composition_method}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_a_dir, use_fast=True, )
    tokenizer_b = AutoTokenizer.from_pretrained(adapter_b_dir, use_fast=True, )
    tokenizer_base = AutoTokenizer.from_pretrained(model_name, use_fast=True, )
    for candidate in (tokenizer, tokenizer_b, tokenizer_base):
        candidate.padding_side = "left"
        if candidate.pad_token is None:
            candidate.pad_token = candidate.eos_token
    tokenizer_signatures = {"base": tokenizer_signature(tokenizer_base), "adapter_a": tokenizer_signature(tokenizer), "adapter_b": tokenizer_signature(tokenizer_b)}
    LOG.info("TOKENIZER SIGNATURES | %s", tokenizer_signatures)
    if len({json.dumps(signature, sort_keys=True) for signature in tokenizer_signatures.values()}) != 1:
        raise RuntimeError(f"Tokenizer mismatch in composition: {tokenizer_signatures}")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (torch.float16 if torch.cuda.is_available() else torch.float32)
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, )
    if composition.composition_method in SEQUENTIAL_COMPOSITION_METHODS:
        raise RuntimeError("The sequential composition method is implemented as an explicit "
                           "two-stage textual chain, not as PEFT mixed-adapter activation. "
                           "Use evaluate_chained_composite_model through composition.")
    model = PeftModel.from_pretrained(base, adapter_a_dir, adapter_name="adapter_a", )
    model.load_adapter(adapter_b_dir, adapter_name="adapter_b", )
    if composition.composition_method in ROUTED_COMPOSITION_METHODS:
        raise NotImplementedError("Routed composition requires a task router; use --sweep-methods with weighted or sequential methods, or configure a router implementation first")
    else:
        combination_type = PEFT_COMBINATION_TYPES[composition.composition_method]
        LOG.info("  combination_type=%s", combination_type)
        model.add_weighted_adapter(adapters=["adapter_a", "adapter_b"], weights=[composition.weight_a, composition.weight_b, ], adapter_name="composed", combination_type=combination_type, )
        model.set_adapter("composed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    LOG.info("Loaded composition %s", composition.composition_id, )
    LOG.info("  A=%s weight=%.3f", adapter_a_dir, composition.weight_a, )
    LOG.info("  B=%s weight=%.3f", adapter_b_dir, composition.weight_b, )
    LOG.info("  active_adapter=%s adapters=%s", getattr(model, "active_adapter", None), list(getattr(model, "peft_config", {}).keys()))
    for name, cfg in getattr(model, "peft_config", {}).items():
        LOG.info("  adapter_config[%s]=%s", name, cfg)
    return model, tokenizer


def tokenizer_signature(tokenizer) -> Dict[str, Any]:
    return {"vocab_size": int(len(tokenizer)), "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id, "bos_token_id": tokenizer.bos_token_id, "chat_template": str(getattr(tokenizer, "chat_template", None))}


def hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(path)).encode())
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def adapter_state_summary(model) -> Dict[str, Any]:
    summary = {"parameters": 0, "trainable_parameters": 0, "lora_parameters": 0, "lora_norms": {}, "lora_modules": {}}
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        summary["parameters"] += count
        if parameter.requires_grad:
            summary["trainable_parameters"] += count
        if "lora_" in name:
            summary["lora_parameters"] += count
            norm = float(parameter.detach().float().norm().cpu())
            summary["lora_norms"][name] = norm
            marker = ".lora_A."
            if marker not in name:
                marker = ".lora_B."
            if marker in name:
                module_name, adapter_name = name.split(marker, 1)
                module_entry = summary["lora_modules"].setdefault(module_name, {})
                component = "A" if ".lora_A." in name else "B"
                adapter_key = adapter_name.rsplit(".weight", 1)[0]
                module_entry.setdefault(adapter_key, {})[f"norm_{component}"] = norm
                module_entry[adapter_key][f"parameters_{component}"] = count
    norms = list(summary["lora_norms"].values())
    summary["lora_norm_total"] = float(sum(norms))
    summary["lora_norm_rms"] = float(np.sqrt(np.mean(np.square(norms)))) if norms else 0.0
    summary["lora_nonzero_tensors"] = int(sum(value > 0.0 for value in norms))
    for module_entry in summary["lora_modules"].values():
        for adapter_entry in module_entry.values():
            norm_a = float(adapter_entry.get("norm_A", 0.0))
            norm_b = float(adapter_entry.get("norm_B", 0.0))
            adapter_entry["effective_update_norm_proxy"] = norm_a * norm_b
    return summary


def summarize_jsonable_values(values: Sequence[Any], limit: int = 50) -> Dict[str, Any]:
    """Return compact value-frequency diagnostics for latent metadata."""
    counts: Dict[str, int] = {}
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {"unique": len(counts), "top": [{"value": key, "count": count} for key, count in ordered[:limit]]}


def dataset_diagnostics(dataset: Dict[str, List[Any]], tokenizer=None, max_input_tokens: int | None = None) -> Dict[str, Any]:
    """Capture the data distribution and tokenization actually used by training/eval."""
    diagnostics: Dict[str, Any] = {"splits": {}}
    for split, examples in dataset.items():
        prompts = [str(example.prompt) for example in examples]
        targets = [str(example.target) for example in examples]
        latent_keys = sorted({key for example in examples for key in getattr(example, "latent", {}).keys()})
        latent = {key: summarize_jsonable_values([getattr(example, "latent", {}).get(key) for example in examples]) for key in latent_keys}
        entry: Dict[str, Any] = {"count": len(examples), "unique_prompts": len(set(prompts)), "unique_targets": len(set(targets)), "prompt_chars": {"min": min(map(len, prompts)) if prompts else 0, "max": max(map(len, prompts)) if prompts else 0, "mean": float(np.mean([len(value) for value in prompts])) if prompts else 0.0}, "target_chars": {"min": min(map(len, targets)) if targets else 0, "max": max(map(len, targets)) if targets else 0, "mean": float(np.mean([len(value) for value in targets])) if targets else 0.0}, "latent": latent, }
        if examples and hasattr(examples[0], "source"):
            entry["source_schemas"] = summarize_jsonable_values([getattr(example, "latent", {}).get("input_schema", getattr(example.source, "schema", None)) for example in examples])
            entry["intermediate_schemas"] = summarize_jsonable_values([getattr(example, "latent", {}).get("intermediate_schema", getattr(getattr(example, "intermediate", None), "schema", None)) for example in examples])
            entry["source_record_types"] = summarize_jsonable_values([type(example.source).__name__ for example in examples])
            entry["intermediate_record_types"] = summarize_jsonable_values([type(getattr(example, "intermediate", None)).__name__ for example in examples])
        if tokenizer is not None:
            prompt_token_lengths = [len(tokenizer(format_chat_prompt(prompt, tokenizer), add_special_tokens=False)["input_ids"]) for prompt in prompts]
            target_token_lengths = [len(tokenizer(target, add_special_tokens=False)["input_ids"]) for target in targets]
            entry["prompt_tokens"] = {"min": min(prompt_token_lengths) if prompt_token_lengths else 0, "max": max(prompt_token_lengths) if prompt_token_lengths else 0, "mean": float(np.mean(prompt_token_lengths)) if prompt_token_lengths else 0.0, "over_max_input_tokens": (sum(value > max_input_tokens for value in prompt_token_lengths) if max_input_tokens is not None else None), "max_input_tokens": max_input_tokens}
            entry["target_tokens"] = {"min": min(target_token_lengths) if target_token_lengths else 0, "max": max(target_token_lengths) if target_token_lengths else 0, "mean": float(np.mean(target_token_lengths)) if target_token_lengths else 0.0}
        diagnostics["splits"][split] = entry
    return diagnostics


def directory_file_manifest(path: Path) -> Dict[str, Any]:
    files = []
    if path.exists():
        for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
            files.append({"path": str(file_path.relative_to(path)), "bytes": file_path.stat().st_size, "sha256": sha256_file(file_path)})
    return {"path": str(path), "exists": path.exists(), "files": files}


def save_model_application_diagnostics(output_dir: Path, label: str, model, tokenizer, eval_cfg: Dict[str, Any], dataset: Dict[str, List[Any]], metadata: Dict[str, Any] | None = None) -> None:
    """Persist the exact model/tokenizer/generation context used for an evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    context = {"created_at": now_ts(), "label": label, "model_class": model.__class__.__name__, "device": str(next(model.parameters()).device), "training_mode": bool(model.training), "active_adapter": getattr(model, "active_adapter", None), "available_adapters": list(getattr(model, "peft_config", {}).keys()), "tokenizer": tokenizer_signature(tokenizer), "generation": {"max_input_tokens": int(eval_cfg["max_input_tokens"]), "max_new_tokens": int(eval_cfg["max_new_tokens"]), "batch_size": int(eval_cfg.get("batch_size", 16)), "do_sample": False, "padding_side": getattr(tokenizer, "padding_side", None)}, "model_state": adapter_state_summary(model), "dataset": dataset_diagnostics(dataset, tokenizer, int(eval_cfg["max_input_tokens"])), "metadata": metadata or {}, }
    write_json(output_dir / "application_diagnostics.json", context)
    LOG.info("Saved application diagnostics for %s to %s", label, output_dir / "application_diagnostics.json")


def log_prediction_diagnostics(label: str, split: str, examples: Sequence[Any], predictions: Sequence[str]) -> Dict[str, Any]:
    lengths = [len(str(p)) for p in predictions]
    empty = sum(not str(p).strip() for p in predictions)
    diagnostics = {"label": label, "split": split, "count": len(predictions), "empty_count": empty, "mean_chars": float(np.mean(lengths)) if lengths else 0.0, "min_chars": min(lengths) if lengths else 0, "max_chars": max(lengths) if lengths else 0, "unique_predictions": len(set(predictions))}
    LOG.info("PREDICTION DIAGNOSTICS | %s/%s | %s", label, split, diagnostics)
    for index in range(min(5, len(examples))):
        LOG.info("PREDICTION SAMPLE | %s/%s/%d | target=%r | prediction=%r | match=%s", label, split, index, examples[index].target, predictions[index], normalize_answer(predictions[index]) == normalize_answer(examples[index].target))
    return diagnostics


@torch.inference_mode()
def generate_predictions(model, tokenizer, examples, max_input_tokens, max_new_tokens, batch_size=16, ):
    device = next(model.parameters()).device
    outputs = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        prompts = [format_chat_prompt(ex.prompt, tokenizer, ) for ex in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens, ).to(device)
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, )
        input_width = inputs["input_ids"].shape[1]
        for i in range(len(batch)):
            new_tokens = generated[i, input_width:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True, ).strip()
            outputs.append(text)
        completed = min(start + batch_size, len(examples), )
        LOG.info("Generated %d/%d predictions", completed, len(examples), )
    return outputs


def save_prediction_records(output_dir: Path, split: str, examples: Sequence[Example], predictions: Sequence[str], ) -> Path:
    if len(examples) != len(predictions):
        raise ValueError(f"Prediction count mismatch for {split}: "
                         f"{len(examples)} examples vs {len(predictions)} predictions")
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    path = prediction_dir / f"{split}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for index, (example, prediction) in enumerate(zip(examples, predictions)):
            comparison = compare_answers(prediction, example.target, )
            record = {"index": index, "task_id": example.task_id, "split": example.split, "latent": example.latent, "prompt": example.prompt, "target": example.target, "prediction": prediction, "target_normalized": comparison["target_normalized"], "prediction_normalized": comparison["prediction_normalized"], "exact_match_raw": comparison["exact_match"], "exact_match_normalized": comparison["normalized_match"], "semantic_match": semantic_match(prediction, example.target, example.latent.get("output_format")), "prediction_length": comparison["prediction_length"], "target_length": comparison["target_length"], "prediction_repr": comparison["prediction_repr"], "target_repr": comparison["target_repr"], }
            f.write(json.dumps(record, ensure_ascii=False, ) + "\n")
    LOG.info("Saved %d prediction records to %s", len(examples), path, )
    return path


def save_composite_prediction_records(output_dir: Path, split: str, examples: Sequence[CompositeExample], predictions: Sequence[str], ) -> Path:
    if len(examples) != len(predictions):
        raise ValueError(f"Prediction count mismatch for {split}")
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True, )
    path = prediction_dir / f"{split}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for index, (example, prediction) in enumerate(zip(examples, predictions)):
            comparison = compare_answers(prediction, example.target, )
            record = {"index": index, "composition_id": example.composition_id, "skill_a": example.skill_a, "skill_b": example.skill_b, "split": example.split, "latent": example.latent, "prompt": example.prompt, "source": asdict(example.source), "intermediate": (asdict(example.intermediate) if hasattr(example.intermediate, "__dataclass_fields__") else example.intermediate), "target": example.target, "prediction": prediction, "target_normalized": comparison["target_normalized"], "prediction_normalized": comparison["prediction_normalized"], "exact_match_normalized": comparison["normalized_match"], "structured_semantic_match": structured_semantic_match(prediction, example.target), "operation_diagnostics": structured_operation_diagnostics(example, prediction), "prediction_repr": comparison["prediction_repr"], "target_repr": comparison["target_repr"], "prediction_length": comparison["prediction_length"], "target_length": comparison["target_length"], }
            f.write(json.dumps(record, ensure_ascii=False, ) + "\n")
    return path


def clean_model_output(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(?:answer|output|result)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_answer(text: str) -> str:
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join(text.split())
    return text.strip().lower()


def parse_structured_fields(text: str) -> List[Tuple[str, str]] | None:
    cleaned = clean_model_output(text)
    normalized = normalize_answer(cleaned)
    if normalized in {"", "none"}:
        return []
    fields = re.findall(r"([a-z0-9_]+)\s*[:=]\s*([a-z0-9_]+)", normalized)
    if not fields:
        return None
    remainder = re.sub(r"([a-z0-9_]+)\s*[:=]\s*([a-z0-9_]+)", "", normalized)
    remainder = remainder.replace(",", "").replace("|", "").replace(";", "").strip()
    return fields if not remainder else None


def structured_semantic_match(prediction: str, target: str) -> bool:
    predicted_fields = parse_structured_fields(prediction)
    target_fields = parse_structured_fields(target)
    return predicted_fields is not None and target_fields is not None and predicted_fields == target_fields


def compare_structured_pairs(candidate: List[Tuple[str, str]] | None, reference: List[Tuple[str, str]] | None) -> Dict[str, Any]:
    """Compare structured records without losing duplicate-field information.

    ``same_pairs_unordered`` is deliberately multiset-aware.  The older set
    comparison is retained as ``same_unique_pairs_unordered`` for diagnosing
    duplicate predictions in historical runs.
    """
    if candidate is None or reference is None:
        return {"available": False}
    candidate_counter = Counter(candidate)
    reference_counter = Counter(reference)
    common = sum((candidate_counter & reference_counter).values())
    predicted_count = len(candidate)
    reference_count = len(reference)
    precision = common / predicted_count if predicted_count else (1.0 if reference_count == 0 else 0.0)
    recall = common / reference_count if reference_count else (1.0 if predicted_count == 0 else 0.0)
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    missing_counts = reference_counter - candidate_counter
    extra_counts = candidate_counter - reference_counter
    return {
        "available": True,
        "exact_order": candidate == reference,
        "same_pairs_unordered": candidate_counter == reference_counter,
        "same_unique_pairs_unordered": set(candidate) == set(reference),
        "predicted_count": predicted_count,
        "reference_count": reference_count,
        "common_pairs": common,
        "missing_pairs": [list(pair) for pair in sorted(missing_counts)],
        "extra_pairs": [list(pair) for pair in sorted(extra_counts)],
        "missing_pair_counts": {f"{key}={value}": count for (key, value), count in sorted(missing_counts.items())},
        "extra_pair_counts": {f"{key}={value}": count for (key, value), count in sorted(extra_counts.items())},
        "duplicate_predicted_count": int(sum(count - 1 for count in candidate_counter.values() if count > 1)),
        "duplicate_reference_count": int(sum(count - 1 for count in reference_counter.values() if count > 1)),
        "content_precision": precision,
        "content_recall": recall,
        "content_f1": f1,
        "order_given_content": bool(candidate == reference) if candidate_counter == reference_counter else False,
    }


def structured_operation_diagnostics_from_record(record: Dict[str, Any], prediction: str | None = None) -> Dict[str, Any]:
    """Recompute structured diagnostics from a saved composite prediction record."""
    prediction_text = str(record.get("prediction", "") if prediction is None else prediction)
    predicted = parse_structured_fields(prediction_text)
    target = parse_structured_fields(str(record.get("target", "")))

    def fields(value: Any) -> List[Tuple[str, str]] | None:
        if isinstance(value, dict) and isinstance(value.get("fields"), list):
            return [(str(pair[0]).strip().lower(), str(pair[1]).strip().lower()) for pair in value["fields"] if isinstance(pair, (list, tuple)) and len(pair) == 2]
        return None

    source = fields(record.get("source"))
    intermediate = fields(record.get("intermediate"))
    return {
        "prediction_parseable": predicted is not None,
        "target_parseable": target is not None,
        "prediction_equals_source": compare_structured_pairs(predicted, source),
        "prediction_equals_intermediate": compare_structured_pairs(predicted, intermediate),
        "prediction_equals_target": compare_structured_pairs(predicted, target),
        "target_equals_intermediate": compare_structured_pairs(target, intermediate),
        "source_field_count": len(source) if source is not None else None,
        "intermediate_field_count": len(intermediate) if intermediate is not None else None,
        "target_field_count": len(target) if target is not None else None,
        "prediction_field_count": len(predicted) if predicted is not None else None,
    }


def structured_operation_diagnostics(example: CompositeExample, prediction: str) -> Dict[str, Any]:
    """Explain whether a prediction follows source, intermediate, and final structure."""
    predicted = parse_structured_fields(prediction)
    target = parse_structured_fields(example.target)
    source_record = example.source if hasattr(example, "source") else None
    intermediate_record = example.intermediate if hasattr(example, "intermediate") else None
    source = list(source_record.fields) if hasattr(source_record, "fields") else None
    intermediate = list(intermediate_record.fields) if hasattr(intermediate_record, "fields") else None

    return {"prediction_parseable": predicted is not None, "target_parseable": target is not None, "prediction_equals_source": compare_structured_pairs(predicted, source), "prediction_equals_intermediate": compare_structured_pairs(predicted, intermediate), "prediction_equals_target": compare_structured_pairs(predicted, target), "target_equals_intermediate": compare_structured_pairs(target, intermediate), "source_field_count": len(source) if source is not None else None, "intermediate_field_count": len(intermediate) if intermediate is not None else None, "target_field_count": len(target) if target is not None else None, "prediction_field_count": len(predicted) if predicted is not None else None, }


def parse_json_object(text: str) -> Dict[str, Any] | None:
    cleaned = clean_model_output(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return {str(k).strip().lower(): str(v).strip().lower() for k, v in data.items()}
    except Exception:
        pass
    m = re.findall(r'["\']?([a-zA-Z0-9_]+)["\']?\s*:\s*["\']?([^,"\';}\s]+)["\']?', cleaned)
    if m:
        return {k.strip().lower(): v.strip().lower() for k, v in m}
    return None


def parse_value_sequence(text: str) -> List[str] | None:
    cleaned = clean_model_output(text)
    norm = normalize_answer(cleaned)
    if norm in {"", "none"}:
        return []
    tokens = re.split(r"[,\|\n]+", norm)
    vals = [t.strip() for t in tokens if t.strip()]
    return vals if vals else None


def parse_label(text: str) -> str:
    cleaned = clean_model_output(text)
    norm = normalize_answer(cleaned)
    norm = re.sub(r"[^\w\s]", "", norm).strip().lower()
    if "yes" in norm.split():
        return "yes"
    if "no" in norm.split():
        return "no"
    return norm


def semantic_match(prediction: str, target: str, schema: str | None = None) -> bool:
    if normalize_answer(prediction) == normalize_answer(target):
        return True

    # JSON schema or target that looks like JSON
    if schema == "json_object" or (target.strip().startswith("{") and target.strip().endswith("}")):
        pj = parse_json_object(prediction)
        tj = parse_json_object(target)
        if pj is not None and tj is not None:
            return pj == tj

    # Structured record or key-value format
    if schema == "structured_record" or any(delim in target for delim in ("=", ":")):
        if structured_semantic_match(prediction, target):
            return True

    # Value sequence
    if schema == "value_sequence":
        pv = parse_value_sequence(prediction)
        tv = parse_value_sequence(target)
        if pv is not None and tv is not None:
            return pv == tv

    # Classification label
    if schema == "label" or target.strip().upper() in {"YES", "NO"}:
        return parse_label(prediction) == parse_label(target)

    return False


def exact_match(pred: str, target: str) -> bool:
    return normalize_answer(pred) == normalize_answer(target)


def compare_answers(pred: str, target: str) -> Dict[str, Any]:
    pred_norm = normalize_answer(pred)
    target_norm = normalize_answer(target)
    return {"exact_match": pred == target, "normalized_match": pred_norm == target_norm, "prediction_length": len(pred), "target_length": len(target), "prediction_normalized": pred_norm, "target_normalized": target_norm, "prediction_repr": repr(pred), "target_repr": repr(target), }


def compare_prediction_files(path_a: Path, path_b: Path, ) -> bool:
    records_a = [json.loads(line) for line in path_a.read_text(encoding="utf-8").splitlines() if line.strip()]
    records_b = [json.loads(line) for line in path_b.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records_a) != len(records_b):
        return False
    for a, b in zip(records_a, records_b):
        if a["prediction"] != b["prediction"]:
            return False
    return True


def evaluate_examples(model, tokenizer, examples: Sequence[Example], eval_cfg: Dict[str, Any], output_dir: Path, split: str, ) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = generate_predictions(model, tokenizer, examples, max_input_tokens=int(eval_cfg["max_input_tokens"]), max_new_tokens=int(eval_cfg["max_new_tokens"]), batch_size=int(eval_cfg.get("batch_size", 16)), )
    debug_n = min(5, len(examples))
    for i in range(debug_n):
        LOG.info("=" * 60)
        LOG.info("EVAL DEBUG | split=%s | example=%d", split, i)
        LOG.info("PROMPT:\n%s", examples[i].prompt)
        LOG.info("TARGET:\n%s", examples[i].target)
        LOG.info("PREDICTION:\n%s", predictions[i])
        LOG.info("EXACT_MATCH=%s", exact_match(predictions[i], examples[i].target), )
    correct = [semantic_match(pred, ex.target, ex.latent.get("output_format")) for pred, ex in zip(predictions, examples)]
    accuracy = sum(correct) / max(1, len(correct))
    prediction_path = save_prediction_records(output_dir=output_dir, split=split, examples=examples, predictions=predictions, )
    return {"n": len(examples), "accuracy": accuracy, "correct": int(sum(correct)), "incorrect": int(len(correct) - sum(correct)), "prediction_file": str(prediction_path), }


def evaluate_composite_examples(model, tokenizer, examples: Sequence[CompositeExample], eval_cfg: Dict[str, Any], output_dir: Path, split: str, ) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = generate_predictions(model=model, tokenizer=tokenizer, examples=examples, max_input_tokens=int(eval_cfg["max_input_tokens"]), max_new_tokens=int(eval_cfg["max_new_tokens"]), batch_size=int(eval_cfg.get("batch_size", 16)), )
    diagnostics = log_prediction_diagnostics(output_dir.name, split, examples, predictions)
    semantic_correct = [semantic_match(pred, example.target, example.latent.get("final_schema")) for pred, example in zip(predictions, examples)]
    exact_correct = [normalize_answer(pred) == normalize_answer(example.target) for pred, example in zip(predictions, examples)]
    accuracy = sum(semantic_correct) / max(1, len(semantic_correct))
    exact_accuracy = sum(exact_correct) / max(1, len(exact_correct))
    operation_diagnostics = [structured_operation_diagnostics(example, prediction) for example, prediction in zip(examples, predictions)]
    target_comparisons = [item["prediction_equals_target"] for item in operation_diagnostics]
    operation_summary = {"prediction_parseable": int(sum(item["prediction_parseable"] for item in operation_diagnostics)), "equals_source_order": int(sum(item["prediction_equals_source"].get("exact_order", False) for item in operation_diagnostics)), "equals_intermediate_order": int(sum(item["prediction_equals_intermediate"].get("exact_order", False) for item in operation_diagnostics)), "equals_target_order": int(sum(item["prediction_equals_target"].get("exact_order", False) for item in operation_diagnostics)), "same_target_pairs_unordered": int(sum(item.get("same_pairs_unordered", False) for item in target_comparisons)), "same_target_unique_pairs_unordered": int(sum(item.get("same_unique_pairs_unordered", False) for item in target_comparisons)), "mean_target_content_precision": float(np.mean([item["content_precision"] for item in target_comparisons if item.get("available")])) if any(item.get("available") for item in target_comparisons) else 0.0, "mean_target_content_recall": float(np.mean([item["content_recall"] for item in target_comparisons if item.get("available")])) if any(item.get("available") for item in target_comparisons) else 0.0, "mean_target_content_f1": float(np.mean([item["content_f1"] for item in target_comparisons if item.get("available")])) if any(item.get("available") for item in target_comparisons) else 0.0, "duplicate_prediction_count": int(sum(item.get("duplicate_predicted_count", 0) for item in target_comparisons if item.get("available"))), "mean_prediction_fields": float(np.mean([item["prediction_field_count"] for item in operation_diagnostics if item["prediction_field_count"] is not None])) if any(item["prediction_field_count"] is not None for item in operation_diagnostics) else 0.0, "mean_target_fields": float(np.mean([item["target_field_count"] for item in operation_diagnostics if item["target_field_count"] is not None])) if any(item["target_field_count"] is not None for item in operation_diagnostics) else 0.0, }
    write_json(output_dir / f"{split}_operation_diagnostics.json", {"summary": operation_summary, "examples": operation_diagnostics})
    LOG.info("COMPOSITE SCORES | %s | semantic_accuracy=%.4f exact_accuracy=%.4f (n=%d)", split, accuracy, exact_accuracy, len(examples))
    prediction_path = save_composite_prediction_records(output_dir=output_dir, split=split, examples=examples, predictions=predictions, )
    return {"n": len(examples), "accuracy": accuracy, "semantic_accuracy": accuracy, "exact_accuracy": exact_accuracy, "correct": int(sum(semantic_correct)), "exact_correct": int(sum(exact_correct)), "semantic_correct": int(sum(semantic_correct)), "incorrect": int(len(examples) - sum(semantic_correct)), "prediction_file": str(prediction_path), "diagnostics": diagnostics, "operation_diagnostics": operation_summary, }


def evaluate_model(model, tokenizer, dataset: Dict[str, List[Example]], eval_cfg: Dict[str, Any], output_dir: Path, model_label: str, ) -> Dict[str, Any]:
    save_model_application_diagnostics(output_dir, model_label, model, tokenizer, eval_cfg, dataset)
    results = {}
    for split in ["test_iid", "test_latent", ]:
        LOG.info("Evaluating %s on split=%s", model_label, split, )
        results[split] = evaluate_examples(model=model, tokenizer=tokenizer, examples=dataset[split], eval_cfg=eval_cfg, output_dir=output_dir, split=f"{model_label}_{split}", )
        LOG.info("  %s/%s accuracy=%.4f", model_label, split, results[split]["accuracy"], )
    return results


def evaluate_composite_model(model, tokenizer, dataset: Dict[str, List[CompositeExample]], eval_cfg: Dict[str, Any], output_dir: Path, label: str, ) -> Dict[str, Any]:
    save_model_application_diagnostics(output_dir, label, model, tokenizer, eval_cfg, dataset)
    results = {}
    for split in ["test_iid", "test_latent", "test_composition_ood", ]:
        results[split] = evaluate_composite_examples(model=model, tokenizer=tokenizer, examples=dataset[split], eval_cfg=eval_cfg, output_dir=output_dir, split=f"{label}_{split}", )
    return results


def _record_body(record: StructuredRecord) -> str:
    """Render the internal chain representation in the atomic prompt format."""
    return " | ".join(f"{key}={value}" for key, value in record.fields) or "NONE"


def _atomic_prompt_for_record(skill_id: str, record: StructuredRecord, output_format: str = "comma", seed: int = 0) -> str:
    """Use an atomic task's native instruction, replacing only its input record.

    This is important for chaining: stage B sees the representation emitted by
    stage A, while retaining the prompt distribution on which its adapter was
    trained.  We deliberately do not reuse the composite prompt, which would
    make the chain evaluate a different instruction than either adapter saw in
    atomic.
    """
    skill = get_skill(skill_id)
    generator = GENERATOR_REGISTRY[skill.generator]
    template = generator(task_id=skill_id, split="chain_template", n=1, seed=seed, latent_overrides={"field_count": max(1, len(record.fields)), "value_offset": 0, "noise_pattern": "none", "output_format": output_format, }, )[0]
    lines = template.prompt.splitlines()
    replacements = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.strip().lower() == "record:":
            lines[index] = f"Record: {_record_body(record)}"
            # format_conversion uses a multiline Record header; remove its
            # original body so the stage cannot see two different records.
            if index + 1 < len(lines) and lines[index + 1].strip() and not lines[index + 1].strip().lower().startswith(("return", "format", "answer", "convert", "output")):
                del lines[index + 1]
            replacements += 1
        elif line.strip().lower().startswith("record:"):
            lines[index] = f"Record: {_record_body(record)}"
            replacements += 1
        index += 1
    if replacements != 1:
        raise ValueError(f"Cannot construct chain prompt for {skill_id}: expected one Record line, found {replacements}")
    return "\n".join(lines)


def _chain_stage_target(skill_id: str, record: StructuredRecord, output_format: str = "comma") -> str:
    transformed = COMPOSITION_TRANSFORMS[skill_id](record)
    if not isinstance(transformed, StructuredRecord):
        raise ValueError(f"True adapter chaining requires {skill_id} to emit a structured record; "
                         f"it emits {type(transformed).__name__}. This pair needs an explicit "
                         "serialization bridge before it can use the textual-chain evaluator.")
    return render_fields(transformed.fields, output_format) if transformed.fields else "NONE"


def validate_true_chain_compatibility(composition: CompositionSpec, dataset: Dict[str, List[CompositeExample]]) -> None:
    """Fail early for chains whose intermediate is not a structured record."""
    examples = next((items for items in dataset.values() if items), [])
    if not examples:
        raise ValueError(f"Cannot validate empty chain dataset for {composition.composition_id}")
    intermediate = COMPOSITION_TRANSFORMS[composition.skill_a](examples[0].source)
    if not isinstance(intermediate, StructuredRecord):
        raise ValueError(f"Sequential/textual chaining is not defined for {composition.skill_a} -> {composition.skill_b}: "
                         f"stage A emits {type(intermediate).__name__}, not a structured record.")
    LOG.info("CHAIN COMPATIBILITY | %s -> %s | intermediate=StructuredRecord", composition.skill_a, composition.skill_b)


def evaluate_chained_composite_model(model_a, tokenizer_a, model_b, tokenizer_b, dataset: Dict[str, List[CompositeExample]], eval_cfg: Dict[str, Any], output_dir: Path, label: str = "chained", ) -> Dict[str, Any]:
    """Evaluate a real A -> text -> parse -> B chain, with per-stage artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model_application_diagnostics(output_dir / "stage_a", f"{label}_stage_a", model_a, tokenizer_a, eval_cfg, dataset, {"chain_role": "producer", "output_representation": "structured_record_text"})
    save_model_application_diagnostics(output_dir / "stage_b", f"{label}_stage_b", model_b, tokenizer_b, eval_cfg, dataset, {"chain_role": "consumer", "input_representation": "structured_record_text"})
    results: Dict[str, Any] = {}
    for split in ["test_iid", "test_latent", "test_composition_ood"]:
        examples = dataset[split]
        stage_a_examples = [Example(task_id=example.skill_a, split=split, latent={**example.latent, "chain_stage": "a"}, prompt=_atomic_prompt_for_record(example.skill_a, example.source, "comma", index), target=_chain_stage_target(example.skill_a, example.source, "comma")) for index, example in enumerate(examples)]
        stage_a_predictions = generate_predictions(model_a, tokenizer_a, stage_a_examples, int(eval_cfg["max_input_tokens"]), int(eval_cfg["max_new_tokens"]), int(eval_cfg.get("batch_size", 16)))
        parsed_intermediates = [parse_structured_fields(prediction) for prediction in stage_a_predictions]
        stage_a_correct = [structured_semantic_match(prediction, stage_example.target) for prediction, stage_example in zip(stage_a_predictions, stage_a_examples)]

        stage_b_examples: List[Example] = []
        valid_indices: List[int] = []
        for index, (example, parsed) in enumerate(zip(examples, parsed_intermediates)):
            if parsed is None:
                continue
            predicted_record = StructuredRecord(fields=tuple(parsed))
            stage_b_examples.append(Example(task_id=example.skill_b, split=split, latent={**example.latent, "chain_stage": "b", "predicted_intermediate": True}, prompt=_atomic_prompt_for_record(example.skill_b, predicted_record, example.latent.get("output_format", "comma"), index), target=example.target))
            valid_indices.append(index)
        stage_b_predictions_valid = generate_predictions(model_b, tokenizer_b, stage_b_examples, int(eval_cfg["max_input_tokens"]), int(eval_cfg["max_new_tokens"]), int(eval_cfg.get("batch_size", 16))) if stage_b_examples else []
        stage_b_predictions = [""] * len(examples)
        for index, prediction in zip(valid_indices, stage_b_predictions_valid):
            stage_b_predictions[index] = prediction
        final_correct = [(structured_semantic_match(prediction, example.target) if example.latent.get("final_schema") == "structured_record" else semantic_match(prediction, example.target, example.latent.get("final_schema"))) for prediction, example in zip(stage_b_predictions, examples)]
        records = []
        for index, (example, stage_a_example, stage_a_prediction, parsed, final_prediction) in enumerate(zip(examples, stage_a_examples, stage_a_predictions, parsed_intermediates, stage_b_predictions)):
            stage_b_prompt = stage_b_examples[valid_indices.index(index)].prompt if index in valid_indices else None
            records.append({"index": index, "composition_id": example.composition_id, "source": asdict(example.source), "gold_intermediate": asdict(example.intermediate) if hasattr(example.intermediate, "fields") else example.intermediate, "gold_target": example.target, "stage_a_prompt": stage_a_example.prompt, "stage_a_target": stage_a_example.target, "stage_a_prediction": stage_a_prediction, "stage_a_correct": stage_a_correct[index], "predicted_intermediate": parsed, "intermediate_parseable": parsed is not None, "stage_b_prompt": stage_b_prompt, "stage_b_prediction": final_prediction, "final_correct": final_correct[index], "failure_reason": ("stage_a_unparseable" if parsed is None else None)})
        prediction_path = output_dir / "predictions" / f"{label}_{split}.jsonl"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = {"n": len(examples), "stage_a_accuracy": sum(stage_a_correct) / max(1, len(examples)), "intermediate_parse_rate": sum(parsed is not None for parsed in parsed_intermediates) / max(1, len(examples)), "stage_b_evaluated": len(stage_b_examples), "stage_b_given_parseable_accuracy": (sum((structured_semantic_match(prediction, stage_example.target) if stage_example.latent.get("final_schema") == "structured_record" else semantic_match(prediction, stage_example.target, stage_example.latent.get("final_schema"))) for prediction, stage_example in zip(stage_b_predictions_valid, stage_b_examples)) / max(1, len(stage_b_examples))), "end_to_end_accuracy": sum(final_correct) / max(1, len(examples)), "prediction_file": str(prediction_path), "unparseable_stage_a": sum(parsed is None for parsed in parsed_intermediates)}
        write_json(output_dir / f"{split}_chain_diagnostics.json", {"summary": summary, "records": records})
        results[split] = {"n": len(examples), "accuracy": summary["end_to_end_accuracy"], "semantic_accuracy": summary["end_to_end_accuracy"], "stage_a_accuracy": summary["stage_a_accuracy"], "intermediate_parse_rate": summary["intermediate_parse_rate"], "stage_b_given_parseable_accuracy": summary["stage_b_given_parseable_accuracy"], "prediction_file": str(prediction_path), "chain_diagnostics": str(output_dir / f"{split}_chain_diagnostics.json")}
        LOG.info("CHAIN SCORES | %s | %s", split, summary)
    write_json(output_dir / "chain_summary.json", {"label": label, "execution_mode": "textual_two_stage_chain", "results": results})
    return results


def evaluate_adapter(config: Dict[str, Any], dataset: Dict[str, List[Example]], adapter_dir: Path, output_dir: Path, ) -> Dict[str, Any]:
    LOG.info("Loading adapter for evaluation: %s", adapter_dir, )
    model, tokenizer = load_adapter_for_eval(config["model"]["name"], adapter_dir, )
    results = evaluate_model(model=model, tokenizer=tokenizer, dataset=dataset, eval_cfg=config["evaluation"], output_dir=output_dir, model_label="adapter", )
    return results


def evaluate_base(config: Dict[str, Any], dataset: Dict[str, List[Example]], output_dir: Path, ) -> Dict[str, Any]:
    LOG.info("Evaluating base model: %s", config["model"]["name"], )
    model, tokenizer = load_base_model_for_eval(config["model"]["name"], )
    results = evaluate_model(model=model, tokenizer=tokenizer, dataset=dataset, eval_cfg=config["evaluation"], output_dir=output_dir, model_label="base", )
    return results


def evaluate_adapter_endpoint(config: Dict[str, Any], dataset: Dict[str, List[Example]], adapter_dir: Path, output_dir: Path, label: str, ) -> Dict[str, Any]:
    model, tokenizer = load_adapter_for_eval(config["model"]["name"], adapter_dir, )
    return evaluate_model(model=model, tokenizer=tokenizer, dataset=dataset, eval_cfg=config["evaluation"], output_dir=output_dir, model_label=label, )


def assert_endpoint_invariant(atomic_eval: Dict[str, Any], composition_eval: Dict[str, Any], tolerance: float, label: str, ) -> None:
    for split in ["test_iid", "test_latent"]:
        expected = atomic_eval[split]["accuracy"]
        observed = composition_eval[split]["accuracy"]
        delta = abs(expected - observed)
        if delta > tolerance:
            raise RuntimeError(f"Endpoint invariant failed for {label}/{split}: atomic={expected:.6f}, composition={observed:.6f}, delta={delta:.6f}, tolerance={tolerance:.6f}")


def calculate_token_statistics(dataset: Dict[str, List[Example]], tokenizer, ) -> Dict[str, Any]:
    result = {}
    for split, examples in dataset.items():
        if not examples:
            continue
        input_lengths = []
        target_lengths = []
        for example in examples:
            formatted_prompt = format_chat_prompt(example.prompt, tokenizer, )
            input_ids = tokenizer(formatted_prompt, add_special_tokens=False, truncation=False, )["input_ids"]
            target_ids = tokenizer(example.target, add_special_tokens=False, truncation=False, )["input_ids"]
            input_lengths.append(len(input_ids))
            target_lengths.append(len(target_ids))

        def stats(values):
            values = sorted(values)
            return {"mean": float(np.mean(values)), "median": float(np.median(values)), "min": int(min(values)), "max": int(max(values)), }

        result[split] = {"input_tokens": stats(input_lengths), "target_tokens": stats(target_lengths), }
    return result


def calculate_composition_metrics(base_score: float, a_score: float, b_score: float, composed_score: float, dedicated_score: float = 0.0, ) -> Dict[str, float]:
    constituent_max = max(float(base_score), float(a_score), float(b_score))
    return {"base_accuracy": float(base_score), "adapter_a_accuracy": float(a_score), "adapter_b_accuracy": float(b_score), "composed_accuracy": float(composed_score), "dedicated_accuracy": float(dedicated_score), "composition_gain": float(composed_score) - constituent_max, "composition_deficit": float(dedicated_score) - float(composed_score), "gain_over_base": float(composed_score) - float(base_score), "constituent_max_accuracy": constituent_max, }


def calculate_atomic_retention(standalone_a: float, composed_a: float, base_a: float = 0.0, standalone_b: float = 0.0, composed_b: float = 0.0, base_b: float = 0.0, ) -> Dict[str, float]:
    return {"retention_a": (composed_a / standalone_a if standalone_a > 0 else 0.0), "retention_b": (composed_b / standalone_b if standalone_b > 0 else 0.0), "interference_a": standalone_a - composed_a, "interference_b": standalone_b - composed_b, "delta_a_standalone": standalone_a - base_a, "delta_a_composed": composed_a - base_a, "delta_b_standalone": standalone_b - base_b, "delta_b_composed": composed_b - base_b, }


def find_atomic_manifest(config: Dict[str, Any], skill_id: str, seed: int) -> Path | None:
    root = Path(config["paths"]["runs"]) / "atomic"
    candidates = sorted(root.glob(f"*_{skill_id}_seed{seed}/manifest.json"))
    return candidates[-1] if candidates else None


def compare_with_atomic_manifest(config: Dict[str, Any], skill_id: str, seed: int, evaluation: Dict[str, Any]) -> Dict[str, Any]:
    path = find_atomic_manifest(config, skill_id, seed)
    if path is None:
        LOG.warning("No atomic manifest found for %s seed %s; endpoint comparison is unavailable", skill_id, seed)
        return {"available": False, "skill_id": skill_id, "seed": seed}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest["evaluation"]["adapter"]
    deltas = {split: float(evaluation[split]["accuracy"] - expected[split]["accuracy"]) for split in ("test_iid", "test_latent")}
    result = {"available": True, "manifest": str(path), "skill_id": skill_id, "seed": seed, "accuracy_deltas": deltas, "passed": all(abs(value) <= 0.001 for value in deltas.values())}
    LOG.info("ATOMIC ENDPOINT CHECK | %s | %s", skill_id, result)
    return result


def evaluate_composition_endpoints(config: Dict[str, Any], composition: CompositionSpec, adapter_a_dir: Path, adapter_b_dir: Path, atomic_a: Dict[str, List[Example]], atomic_b: Dict[str, List[Example]], output_dir: Path, ) -> Dict[str, Any]:
    """Check λ endpoints against independently loaded adapters on atomic tasks."""
    results: Dict[str, Any] = {}
    for endpoint, wa, wb, reference_dir, atomic_dataset, skill in (("a", 1.0, 0.0, adapter_a_dir, atomic_a, composition.skill_a), ("b", 0.0, 1.0, adapter_b_dir, atomic_b, composition.skill_b)):
        if composition.composition_method in SEQUENTIAL_COMPOSITION_METHODS:
            # A textual chain has no meaningful one-hot PEFT endpoint.  The
            # endpoint invariant is therefore checked with the corresponding
            # independently loaded atomic adapter.
            model, tokenizer = load_adapter_for_eval(config["model"]["name"], reference_dir)
            endpoint_mode = "atomic_reference"
        else:
            endpoint_spec = replace(composition, composition_id=f"{composition.composition_id}__endpoint_{endpoint}", weight_a=wa, weight_b=wb)
            model, tokenizer = load_composed_adapter_for_eval(config["model"]["name"], adapter_a_dir, adapter_b_dir, endpoint_spec)
            endpoint_mode = "weighted_peft_endpoint"
        composed_eval = evaluate_model(model, tokenizer, atomic_dataset, config["evaluation"], output_dir / f"endpoint_{endpoint}", f"endpoint_{endpoint}")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        reference_model, reference_tokenizer = load_adapter_for_eval(config["model"]["name"], reference_dir)
        reference_eval = evaluate_model(reference_model, reference_tokenizer, atomic_dataset, config["evaluation"], output_dir / f"reference_{endpoint}", f"reference_{endpoint}")
        del reference_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        deltas = {split: float(composed_eval[split]["accuracy"] - reference_eval[split]["accuracy"]) for split in ("test_iid", "test_latent")}
        results[endpoint] = {"weights": {"a": wa, "b": wb}, "mode": endpoint_mode, "composed": composed_eval, "reference": reference_eval, "accuracy_deltas": deltas, "passed": all(abs(value) <= float(config["composition"]["gates"]["endpoint_accuracy_tolerance"]) for value in deltas.values()), "reference_adapter": str(reference_dir), "skill": skill}
        LOG.info("COMPOSITION ENDPOINT | %s | %s", endpoint, results[endpoint])
    return results


###############################
#        ORCHESTRATION        #
# #############################


def validate_atomic_dataset(dataset: Dict[str, List[Example]], skill: SkillSpec, ) -> None:
    required_splits = {"train", "validation", "test_iid", "test_latent", }
    if set(dataset) != required_splits: raise ValueError(f"Unexpected dataset splits for {skill.skill_id}: {sorted(dataset)}")
    for split in required_splits:
        if not dataset[split]: raise ValueError(f"Empty {split} split for {skill.skill_id}")
    if skill.skill_id == "structured_classification":
        for split in ["train", "validation", "test_iid", "test_latent", ]:
            labels = [example.target for example in dataset[split]]
            yes_count = labels.count("YES")
            no_count = labels.count("NO")
            if abs(yes_count - no_count) > 1: raise ValueError(f"Classification labels are not balanced for split={split}: YES={yes_count}, NO={no_count}")
            invalid = set(labels) - {"YES", "NO"}
            if invalid: raise ValueError(f"Invalid classification labels in split={split}: {invalid}")
        train_rules = {e.latent["rule"] for e in dataset["train"]}
        latent_rules = {e.latent["rule"] for e in dataset["test_latent"]}
        if train_rules != {"contains_alpha"}: raise ValueError("Classification training must use only the contains_alpha operation.")
        if latent_rules != {"contains_alpha"}: raise ValueError("Classification latent OOD must preserve the contains_alpha operation.")
    operation_invariants = {"structured_extraction": {"input_schema": "key_value_record", }, "structured_classification": {"input_schema": "key_value_record", "rule": "contains_alpha", "label_format": "label", }, "field_filtering": {"input_schema": "key_value_record", "filter_rule": "odd_positions", }, "format_conversion": {"input_schema": "key_value_record", "target_format": "json", }, "field_sorting": {"input_schema": "key_value_record", "sort_order": "ascending_value", }, "value_normalization": {"input_schema": "key_value_record", "normalization": "lowercase", }, "field_renaming": {"input_schema": "key_value_record", "rename_scheme": "field_index", }, "duplicate_removal": {"input_schema": "key_value_record", "deduplication": "keep_first", }, "value_selection": {"input_schema": "key_value_record", "selection_rule": "value_starts_with_vowel", }, "value_concatenation": {"input_schema": "key_value_record", "operation": "ordered_value_concatenation", }, "key_value_swap": {"input_schema": "key_value_record", "operation": "key_value_swap", }, "field_reversal": {"input_schema": "key_value_record", "operation": "field_reversal", }, }
    operation_invariants.update({
        "key_prefixing": {"input_schema": "key_value_record", "target": "key", "affix": "prefix", "operation": "key_affix"},
        "key_suffixing": {"input_schema": "key_value_record", "target": "key", "affix": "suffix", "operation": "key_affix"},
        "value_prefixing": {"input_schema": "key_value_record", "target": "value", "affix": "prefix", "operation": "value_affix"},
        "value_suffixing": {"input_schema": "key_value_record", "target": "value", "affix": "suffix", "operation": "value_affix"},
    })
    invariants = operation_invariants.get(skill.skill_id)
    if invariants is None: raise ValueError(f"No operation invariants defined for {skill.skill_id}")
    for latent_key, expected_value in invariants.items():
        train_values = {example.latent.get(latent_key) for example in dataset["train"]}
        latent_values = {example.latent.get(latent_key) for example in dataset["test_latent"]}
        if train_values != {expected_value}: raise ValueError(f"{skill.skill_id}: training parameter {latent_key!r} must equal {expected_value!r}; found {train_values}")
        if latent_values != {expected_value}: raise ValueError(f"{skill.skill_id}: latent-OOD parameter {latent_key!r} must preserve the operation value {expected_value!r}; found {latent_values}")
    latent_spec = LATENT_OOD_SPECS.get(skill.skill_id)
    if latent_spec is None: raise ValueError(f"No LATENT_OOD_SPECS entry for {skill.skill_id}")
    ood_parameters = tuple(latent_spec["ood_parameters"])
    if not ood_parameters: raise ValueError(f"{skill.skill_id}: latent-OOD must hold out at least one parameter.")
    for parameter in ood_parameters:
        train_values = {example.latent.get(parameter) for example in dataset["train"]}
        latent_values = {example.latent.get(parameter) for example in dataset["test_latent"]}
        if None in train_values: raise ValueError(f"{skill.skill_id}: training examples are missing latent parameter {parameter!r}")
        if None in latent_values: raise ValueError(f"{skill.skill_id}: latent-OOD examples are missing latent parameter {parameter!r}")
        overlap = train_values & latent_values
        if overlap: raise ValueError(f"{skill.skill_id}: latent-OOD parameter {parameter!r} overlaps with training values: {overlap}")
    train_latents = dataset["train"]
    latent_examples = dataset["test_latent"]
    declared_latent_parameters = set(ood_parameters)
    for parameter in skill.latent_schema:
        if parameter in declared_latent_parameters: continue
        train_values = {example.latent.get(parameter) for example in train_latents}
        latent_values = {example.latent.get(parameter) for example in latent_examples}
        if not train_values: raise ValueError(f"{skill.skill_id}: no training values found for latent parameter {parameter!r}")
        if not latent_values: raise ValueError(f"{skill.skill_id}: no latent-OOD values found for latent parameter {parameter!r}")
        outside_training = latent_values - train_values
        if outside_training: raise ValueError(f"{skill.skill_id}: non-OOD parameter {parameter!r} contains values outside its training distribution: {outside_training}")


def validate_composite_dataset(dataset: Dict[str, List[CompositeExample]], composition: CompositionSpec, ) -> None:
    required = {"train", "validation", "test_iid", "test_latent", "test_composition_ood", }
    if set(dataset) != required: raise ValueError(f"Unexpected composite splits: {sorted(dataset)}")
    direct_a_matches = 0
    direct_b_matches = 0
    total = 0
    for split, examples in dataset.items():
        if not examples: raise ValueError(f"Empty composite split: {split}")
        for example in examples:
            if example.composition_id != composition.composition_id: raise ValueError(f"Composition ID mismatch in {split}")
            if example.skill_a != composition.skill_a: raise ValueError(f"skill_a mismatch in {split}")
            if example.skill_b != composition.skill_b: raise ValueError(f"skill_b mismatch in {split}")
            expected_intermediate = COMPOSITION_TRANSFORMS[composition.skill_a](example.source)
            is_unrelated = COMPOSITION_RELATIONSHIPS.get((composition.skill_a, composition.skill_b)) == "unrelated"
            if not isinstance(expected_intermediate, StructuredRecord) and not is_unrelated:
                raise ValueError(
                    f"Invalid composite task {composition.composition_id}: "
                    f"skill A emits {type(expected_intermediate).__name__}, but skill B requires a structured record. "
                    "Use a serialization bridge or classify this pair as an unrelated negative control."
                )
            expected_final = COMPOSITION_TRANSFORMS[composition.skill_b](expected_intermediate if isinstance(expected_intermediate, StructuredRecord) else example.source)
            direct_final = COMPOSITION_TRANSFORMS[composition.skill_b](example.source)
            direct_a_final = COMPOSITION_TRANSFORMS[composition.skill_a](example.source)
            if example.latent.get("sampling_mode") == "controlled" and (composition.skill_a, composition.skill_b) == ("value_selection", "field_sorting"):
                if not isinstance(expected_intermediate, StructuredRecord) or len(expected_intermediate.fields) < 2:
                    raise ValueError(f"Controlled sampling produced fewer than two selected fields in {split}")
                if expected_intermediate.fields == expected_final.fields:
                    raise ValueError(f"Controlled sampling produced a vacuous sorting step in {split}")
            if expected_intermediate != example.intermediate: raise ValueError(f"Intermediate ground truth mismatch in {split}")
            if isinstance(expected_final, StructuredRecord):
                expected_target = render_fields(expected_final.fields, example.latent["output_format"]) if expected_final.fields else "NONE"
            elif composition.skill_b == "value_concatenation":
                values = [value for _, value in expected_intermediate.fields] if isinstance(expected_intermediate, StructuredRecord) else [str(expected_final)]
                expected_target = render_values(values, example.latent["output_format"])
            else:
                expected_target = str(expected_final)
            if normalize_answer(expected_target) != normalize_answer(example.target):
                raise ValueError(f"Final ground truth mismatch in {split}: expected {expected_target!r} vs target {example.target!r}")
            direct_b_matches += int(expected_final == direct_final)
            direct_a_matches += int(expected_final == direct_a_final)
            total += 1
    direct_ratio = direct_b_matches / max(1, total)
    direct_a_ratio = direct_a_matches / max(1, total)
    LOG.info("COMPOSITE TASK AUDIT | %s | direct_A_equivalence=%.3f (%d/%d) direct_B_equivalence=%.3f (%d/%d)", composition.composition_id, direct_a_ratio, direct_a_matches, total, direct_ratio, direct_b_matches, total)
    if direct_ratio > 0.20 and not is_unrelated:
        raise ValueError(f"Weak composite task {composition.composition_id}: B alone computes the final transform for {direct_ratio:.1%} of examples; require <=20%")
    if direct_a_ratio > 0.20 and not is_unrelated:
        raise ValueError(f"Weak composite task {composition.composition_id}: A alone computes the final transform for {direct_a_ratio:.1%} of examples; require <=20%")


def run_single_skill_seed(config: Dict[str, Any], skill_id: str, seed: int, run_name: str, ) -> Dict[str, Any]:
    skill = get_skill(skill_id)
    run_id = f"{run_name}_{skill_id}_seed{seed}"
    run_root = (Path(config["paths"]["runs"]) / "atomic" / run_id)
    setup_logging(run_root, config["experiment"].get("log_level", "INFO"), )
    LOG.info("=" * 72)
    LOG.info("ATOMIC SKILL RUN: %s", run_id)
    LOG.info("=" * 72)
    write_json(run_root / "config.json", config, )
    write_json(run_root / "skill.json", asdict(skill), )
    write_json(run_root / "system_metadata.json", system_metadata(), )
    set_global_seed(seed)
    dataset_cfg = config["dataset"]
    dataset = generate_skill_dataset(skill_id=skill_id, split_sizes=dataset_cfg["split_sizes"], seed=seed, test_sizes=dataset_cfg["test_sizes"], )
    validate_atomic_dataset(dataset=dataset, skill=skill, )
    dataset_version = (f"{skill_id}_{skill.generator_version}_seed{seed}")
    dataset_dir = (Path(config["paths"]["datasets"]) / dataset_version)
    dataset_manifest = save_dataset_bundle(dataset_dir, dataset, config, )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], use_fast=True, )
    token_stats = calculate_token_statistics(dataset, tokenizer, )
    data_diagnostics = dataset_diagnostics(dataset, tokenizer)
    write_json(run_root / "data_diagnostics.json", data_diagnostics)
    write_json(run_root / "token_statistics.json", token_stats, )
    adapter_dir = (Path(config["paths"]["adapters"]) / skill_id / f"seed{seed}")
    train_result = train_adapter(config=config, dataset=dataset, output_dir=adapter_dir, seed=seed, )
    write_json(run_root / "trained_adapter_provenance.json", {"adapter_dir": train_result["adapter_dir"], "adapter_files": directory_file_manifest(Path(train_result["adapter_dir"])), "training_metrics": train_result["training_metrics"], "adapter_state_after_training": json.loads((adapter_dir / "adapter_state_after_training.json").read_text()) if (adapter_dir / "adapter_state_after_training.json").exists() else None})
    base_eval = evaluate_base(config=config, dataset=dataset, output_dir=run_root, )
    adapter_eval = evaluate_adapter(config=config, dataset=dataset, adapter_dir=Path(train_result["adapter_dir"]), output_dir=run_root, )
    eval_result = {"base": base_eval, "adapter": adapter_eval, }
    write_json(run_root / "evaluation_metrics.json", eval_result, )
    qualification = build_skill_qualification(skill=skill, evaluation=eval_result, training_metrics=train_result["training_metrics"], )
    qualification["token_statistics"] = token_stats
    write_json(run_root / "qualification.json", qualification, )
    manifest = {"run_id": run_id, "skill_id": skill_id, "seed": seed, "completed_at": now_ts(), "dataset_dir": str(dataset_dir), "dataset_manifest": dataset_manifest, "adapter_dir": train_result["adapter_dir"], "training": train_result["training_metrics"], "evaluation": eval_result, "qualification": qualification, }
    write_json(run_root / "manifest.json", manifest, )
    return manifest


def aggregate_skill_runs(skill_id: str, manifests: List[Dict[str, Any]], ) -> Dict[str, Any]:
    if not manifests:
        raise ValueError(f"No manifests available for {skill_id}")
    skill = get_skill(skill_id)
    metric_names = ["base_test_iid_accuracy", "adapter_test_iid_accuracy", "iid_gain_over_base", "base_test_latent_accuracy", "adapter_test_latent_accuracy", "latent_gain_over_base", "training_time_seconds", ]
    summary = {}
    for metric in metric_names:
        values = [float(m["qualification"][metric]) for m in manifests if m["qualification"].get(metric) is not None]
        if not values:
            continue
        summary[metric] = {"mean": float(np.mean(values)), "std": (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0), "min": float(np.min(values)), "max": float(np.max(values)), "values": values, }
    return {"skill_id": skill.skill_id, "family": skill.family, "relationship_group": skill.relationship_group, "generator": skill.generator, "generator_version": skill.generator_version, "seed_count": len(manifests), "seeds": [m["seed"] for m in manifests], "metrics": summary, }


def composition_static_cache_key(config: Dict[str, Any], composition: CompositionSpec, seed: int) -> Tuple[str, str, int, str]:
    cache_config = {"model": config.get("model"), "dataset": config.get("dataset"), "lora": config.get("lora"), "training": config.get("training"), "composite_dataset_sizes": config.get("composition", {}).get("composite_dataset_sizes"), "composite_training_data": config.get("composition", {}).get("composite_training_data"), "composite_sampling": config.get("composition", {}).get("composite_sampling")}
    return (composition.skill_a, composition.skill_b, int(seed), json.dumps(cache_config, sort_keys=True, default=str))


def remap_cached_paths(value: Any, source_root: Path, target_root: Path) -> Any:
    if isinstance(value, str):
        return value.replace(str(source_root), str(target_root))
    if isinstance(value, list):
        return [remap_cached_paths(item, source_root, target_root) for item in value]
    if isinstance(value, dict):
        return {key: remap_cached_paths(item, source_root, target_root) for key, item in value.items()}
    return value


def copy_static_composition_artifacts(source_root: Path, target_root: Path, composition_id: str, artifact_dirs: Sequence[str]) -> None:
    for relative in artifact_dirs:
        source = source_root / relative
        target = target_root / relative
        if source.exists():
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    # Static prediction records were generated on the first sweep member.  The
    # examples are identical, but update the metadata so each run remains
    # self-describing.
    for prediction_path in target_root.glob("*/predictions/*.jsonl"):
        records = []
        changed = False
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("composition_id") != composition_id and "composition_id" in record:
                record["composition_id"] = composition_id
                changed = True
            records.append(record)
        if changed:
            prediction_path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


def evaluate_composition_static_controls(config: Dict[str, Any], composition: CompositionSpec, seed: int, run_root: Path, dataset: Dict[str, List[CompositeExample]], atomic_a: Dict[str, List[Example]], atomic_b: Dict[str, List[Example]], adapter_a_dir: Path, adapter_b_dir: Path) -> Dict[str, Any]:
    """Train/evaluate controls invariant across weights and merge methods."""
    dedicated_output = Path(config["paths"]["adapters"]) / "composites" / composition.composition_id / f"seed{seed}"
    dedicated_train = train_composite_adapter(config=config, dataset=dataset, output_dir=dedicated_output, seed=seed)
    write_json(run_root / "dedicated_adapter_provenance.json", {"adapter_dir": dedicated_train["adapter_dir"], "adapter_files": directory_file_manifest(Path(dedicated_train["adapter_dir"])), "training_metrics": dedicated_train["training_metrics"], "training_data_diagnostics": json.loads((dedicated_output / "training_data_diagnostics.json").read_text()) if (dedicated_output / "training_data_diagnostics.json").exists() else None})
    dedicated_eval_model, dedicated_tokenizer = load_adapter_for_eval(config["model"]["name"], Path(dedicated_train["adapter_dir"]))
    dedicated_eval = evaluate_composite_model(dedicated_eval_model, dedicated_tokenizer, dataset, config["composition"]["evaluation"], run_root / "dedicated", "dedicated")
    dedicated_atomic_a = evaluate_model(dedicated_eval_model, dedicated_tokenizer, atomic_a, config["evaluation"], run_root / "atomic_a_dedicated", "dedicated_atomic_a")
    dedicated_atomic_b = evaluate_model(dedicated_eval_model, dedicated_tokenizer, atomic_b, config["evaluation"], run_root / "atomic_b_dedicated", "dedicated_atomic_b")
    del dedicated_eval_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    base_model, base_tokenizer = load_base_model_for_eval(config["model"]["name"])
    base_composite_eval = evaluate_composite_model(base_model, base_tokenizer, dataset, config["composition"]["evaluation"], run_root / "base", "base")
    base_atomic_a = evaluate_model(base_model, base_tokenizer, atomic_a, config["evaluation"], run_root / "atomic_a_base", "base_atomic_a")
    base_atomic_b = evaluate_model(base_model, base_tokenizer, atomic_b, config["evaluation"], run_root / "atomic_b_base", "base_atomic_b")
    del base_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    adapter_a_eval = evaluate_constituent_on_composite(config, adapter_a_dir, dataset, run_root / "adapter_a", f"adapter_a_{composition.skill_a}")
    standalone_a_model, standalone_a_tokenizer = load_adapter_for_eval(config["model"]["name"], adapter_a_dir)
    standalone_a_atomic = evaluate_model(standalone_a_model, standalone_a_tokenizer, atomic_a, config["evaluation"], run_root / "atomic_a_standalone", "adapter_a_atomic")
    adapter_a_on_b = evaluate_model(standalone_a_model, standalone_a_tokenizer, atomic_b, config["evaluation"], run_root / "atomic_b_by_adapter_a", "adapter_a_on_b")
    del standalone_a_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    adapter_b_eval = evaluate_constituent_on_composite(config, adapter_b_dir, dataset, run_root / "adapter_b", f"adapter_b_{composition.skill_b}")
    standalone_b_model, standalone_b_tokenizer = load_adapter_for_eval(config["model"]["name"], adapter_b_dir)
    standalone_b_atomic = evaluate_model(standalone_b_model, standalone_b_tokenizer, atomic_b, config["evaluation"], run_root / "atomic_b_standalone", "adapter_b_atomic")
    adapter_b_on_a = evaluate_model(standalone_b_model, standalone_b_tokenizer, atomic_a, config["evaluation"], run_root / "atomic_a_by_adapter_b", "adapter_b_on_a")
    del standalone_b_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    static = {"dedicated_train": dedicated_train, "dedicated_eval": dedicated_eval, "dedicated_atomic_a": dedicated_atomic_a, "dedicated_atomic_b": dedicated_atomic_b, "base_composite_eval": base_composite_eval, "base_atomic_a": base_atomic_a, "base_atomic_b": base_atomic_b, "adapter_a_eval": adapter_a_eval, "standalone_a_atomic": standalone_a_atomic, "adapter_a_on_b": adapter_a_on_b, "adapter_b_eval": adapter_b_eval, "standalone_b_atomic": standalone_b_atomic, "adapter_b_on_a": adapter_b_on_a, "source_run_root": str(run_root), "artifact_dirs": ["dedicated", "atomic_a_dedicated", "atomic_b_dedicated", "base", "atomic_a_base", "atomic_b_base", "adapter_a", "atomic_a_standalone", "atomic_b_by_adapter_a", "adapter_b", "atomic_b_standalone", "atomic_a_by_adapter_b", "dedicated_adapter_provenance.json"]}
    return static


def run_single_composition(config: Dict[str, Any], composition: CompositionSpec, seed: int, ) -> Dict[str, Any]:
    run_id = f"{composition.composition_id}_seed{seed}"
    run_root = Path(config["paths"]["runs"]) / "composition" / composition.composition_id / f"seed_{seed}"
    setup_logging(run_root, config["experiment"].get("log_level", "INFO"), )
    LOG.info("=" * 72)
    LOG.info("COMPOSITION RUN: %s", run_id)
    LOG.info("=" * 72)
    write_json(run_root / "config.json", config)
    write_json(run_root / "system_metadata.json", system_metadata())
    set_global_seed(seed)
    dataset = generate_composite_dataset(composition=composition, config=config, seed=seed, )
    if composition.composition_method in SEQUENTIAL_COMPOSITION_METHODS:
        validate_true_chain_compatibility(composition, dataset)
    dataset_dir = Path(config["paths"]["datasets"]) / "composite" / composition.composition_id / f"seed_{seed}"
    dataset_manifest = save_composite_dataset_bundle(dataset_dir=dataset_dir, dataset=dataset, config=config, )
    write_json(run_root / "composite_data_diagnostics.json", dataset_diagnostics(dataset))
    adapter_a_dir = (Path(config["paths"]["adapters"]) / composition.skill_a / f"seed{seed}" / "adapter")
    adapter_b_dir = (Path(config["paths"]["adapters"]) / composition.skill_b / f"seed{seed}" / "adapter")
    if not adapter_a_dir.exists(): raise FileNotFoundError(f"Missing atomic adapter A: {adapter_a_dir}")
    if not adapter_b_dir.exists(): raise FileNotFoundError(f"Missing atomic adapter B: {adapter_b_dir}")
    adapter_hashes_before = {"a": hash_directory(adapter_a_dir), "b": hash_directory(adapter_b_dir)}
    composition = replace(composition, adapter_a=str(adapter_a_dir), adapter_b=str(adapter_b_dir))
    write_json(run_root / "composition.json", asdict(composition))
    write_json(run_root / "constituent_adapter_provenance.json", {"a": {"skill": composition.skill_a, "path": str(adapter_a_dir), "hash_before": adapter_hashes_before["a"], "files": directory_file_manifest(adapter_a_dir)}, "b": {"skill": composition.skill_b, "path": str(adapter_b_dir), "hash_before": adapter_hashes_before["b"], "files": directory_file_manifest(adapter_b_dir)}})

    specs_dir = Path(config["paths"]["compositions"]) / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    write_json(specs_dir / f"{composition.composition_id}.json", asdict(composition))

    write_json(run_root / "dataset_statistics.json", {split: {"count": len(examples), "unique_targets": len({e.target for e in examples}), "unique_sources": len({json.dumps(asdict(e.source), sort_keys=True) for e in examples}), "target_lengths": {"min": min(len(e.target) for e in examples), "max": max(len(e.target) for e in examples), "mean": float(np.mean([len(e.target) for e in examples]))}} for split, examples in dataset.items()})

    atomic_a = generate_skill_dataset(composition.skill_a, config["dataset"]["split_sizes"], seed, config["dataset"]["test_sizes"])
    atomic_b = generate_skill_dataset(composition.skill_b, config["dataset"]["split_sizes"], seed, config["dataset"]["test_sizes"])
    validate_atomic_dataset(atomic_a, get_skill(composition.skill_a))
    validate_atomic_dataset(atomic_b, get_skill(composition.skill_b))
    save_dataset_bundle(run_root / "atomic_dataset_a", atomic_a, config)
    save_dataset_bundle(run_root / "atomic_dataset_b", atomic_b, config)
    write_json(run_root / "atomic_data_diagnostics.json", {"a": dataset_diagnostics(atomic_a), "b": dataset_diagnostics(atomic_b)})

    # Static controls are invariant across composition weights and methods.
    # Cache them once per ordered skill pair and seed, then copy their saved
    # artifacts into each sweep member so every run remains inspectable.
    reuse_static = bool(config.get("composition", {}).get("reuse_static_evaluations", True))
    static_key = composition_static_cache_key(config, composition, seed)
    static = COMPOSITION_STATIC_CACHE.get(static_key) if reuse_static else None
    if static is None:
        static = evaluate_composition_static_controls(config, composition, seed, run_root, dataset, atomic_a, atomic_b, adapter_a_dir, adapter_b_dir)
        if reuse_static:
            COMPOSITION_STATIC_CACHE[static_key] = copy.deepcopy(static)
        LOG.info("STATIC CONTROLS | computed once | pair=%s seed=%s", f"{composition.skill_a}:{composition.skill_b}", seed)
    else:
        source_root = Path(static["source_run_root"])
        copy_static_composition_artifacts(source_root, run_root, composition.composition_id, static["artifact_dirs"])
        static = remap_cached_paths(copy.deepcopy(static), source_root, run_root)
        LOG.info("STATIC CONTROLS | reused | pair=%s seed=%s", f"{composition.skill_a}:{composition.skill_b}", seed)

    dedicated_train = static["dedicated_train"]
    dedicated_eval = static["dedicated_eval"]
    dedicated_atomic_a = static["dedicated_atomic_a"]
    dedicated_atomic_b = static["dedicated_atomic_b"]
    base_composite_eval = static["base_composite_eval"]
    base_atomic_a = static["base_atomic_a"]
    base_atomic_b = static["base_atomic_b"]
    adapter_a_eval = static["adapter_a_eval"]
    standalone_a_atomic = static["standalone_a_atomic"]
    adapter_a_on_b = static["adapter_a_on_b"]
    adapter_b_eval = static["adapter_b_eval"]
    standalone_b_atomic = static["standalone_b_atomic"]
    adapter_b_on_a = static["adapter_b_on_a"]
    endpoint_key = (composition.skill_a, composition.skill_b, int(seed), composition.composition_method)
    endpoint_cache = COMPOSITION_ENDPOINT_CACHE.get(endpoint_key) if reuse_static else None
    if endpoint_cache is None:
        endpoint_results = evaluate_composition_endpoints(config, composition, adapter_a_dir, adapter_b_dir, atomic_a, atomic_b, run_root / "endpoints")
        if reuse_static:
            COMPOSITION_ENDPOINT_CACHE[endpoint_key] = {"endpoint_results": copy.deepcopy(endpoint_results), "source_run_root": str(run_root)}
        LOG.info("ENDPOINT CONTROLS | computed once | pair=%s seed=%s method=%s", f"{composition.skill_a}:{composition.skill_b}", seed, composition.composition_method)
    else:
        source_root = Path(endpoint_cache["source_run_root"])
        copy_static_composition_artifacts(source_root, run_root, composition.composition_id, ["endpoints"])
        endpoint_results = remap_cached_paths(copy.deepcopy(endpoint_cache["endpoint_results"]), source_root, run_root)
        LOG.info("ENDPOINT CONTROLS | reused | pair=%s seed=%s method=%s", f"{composition.skill_a}:{composition.skill_b}", seed, composition.composition_method)

    # 5. Composed Model (A + B).  "Sequential" is a real dataflow chain:
    # adapter A produces text, that text is parsed into a structured record,
    # and the parsed record is supplied to adapter B.  Other methods remain
    # parameter-space PEFT compositions.
    if composition.composition_method in SEQUENTIAL_COMPOSITION_METHODS:
        chain_a_model, chain_a_tokenizer = load_adapter_for_eval(config["model"]["name"], adapter_a_dir)
        chain_b_model, chain_b_tokenizer = load_adapter_for_eval(config["model"]["name"], adapter_b_dir)
        composed_eval = evaluate_chained_composite_model(chain_a_model, chain_a_tokenizer, chain_b_model, chain_b_tokenizer, dataset, config["composition"]["evaluation"], run_root / "composed", "composed")
        # Atomic retention for a chain is measured on the corresponding stage,
        # not by feeding both adapters the same prompt.
        composed_atomic_a = evaluate_model(chain_a_model, chain_a_tokenizer, atomic_a, config["evaluation"], run_root / "atomic_a_composed", "composed_atomic_a")
        composed_atomic_b = evaluate_model(chain_b_model, chain_b_tokenizer, atomic_b, config["evaluation"], run_root / "atomic_b_composed", "composed_atomic_b")
        deterministic_examples = dataset["test_iid"][:int(config["composition"].get("determinism_examples", 5))]
        first = evaluate_chained_composite_model(chain_a_model, chain_a_tokenizer, chain_b_model, chain_b_tokenizer, {"test_iid": deterministic_examples, "test_latent": deterministic_examples, "test_composition_ood": deterministic_examples}, config["composition"]["evaluation"], run_root / "determinism_probe", "determinism")
        second = evaluate_chained_composite_model(chain_a_model, chain_a_tokenizer, chain_b_model, chain_b_tokenizer, {"test_iid": deterministic_examples, "test_latent": deterministic_examples, "test_composition_ood": deterministic_examples}, config["composition"]["evaluation"], run_root / "determinism_probe_2", "determinism")
        first_predictions = Path(first["test_iid"]["prediction_file"]).read_bytes()
        second_predictions = Path(second["test_iid"]["prediction_file"]).read_bytes()
        determinism_check = {"n": len(deterministic_examples), "passed": first_predictions == second_predictions, "first": first["test_iid"], "second": second["test_iid"]}
        composed_state = {"execution_mode": "textual_two_stage_chain", "stage_a": adapter_state_summary(chain_a_model), "stage_b": adapter_state_summary(chain_b_model)}
        write_json(run_root / "composed_adapter_application.json", {"composition": asdict(composition), "execution_mode": "textual_two_stage_chain", "adapter_a": {"path": str(adapter_a_dir), "hash": hash_directory(adapter_a_dir), "role": "producer"}, "adapter_b": {"path": str(adapter_b_dir), "hash": hash_directory(adapter_b_dir), "role": "consumer"}, "state": composed_state, "determinism": determinism_check})
        del chain_a_model, chain_b_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        composed_model, composed_tokenizer = load_composed_adapter_for_eval(model_name=config["model"]["name"], adapter_a_dir=adapter_a_dir, adapter_b_dir=adapter_b_dir, composition=composition, )
        write_json(run_root / "composed_adapter_application.json", {"composition": asdict(composition), "execution_mode": "weighted_peft_composition", "adapter_a": {"path": str(adapter_a_dir), "hash": hash_directory(adapter_a_dir)}, "adapter_b": {"path": str(adapter_b_dir), "hash": hash_directory(adapter_b_dir)}, "active_adapter": getattr(composed_model, "active_adapter", None), "available_adapters": list(getattr(composed_model, "peft_config", {}).keys()), "state": adapter_state_summary(composed_model)})
        composed_eval = evaluate_composite_model(model=composed_model, tokenizer=composed_tokenizer, dataset=dataset, eval_cfg=config["composition"]["evaluation"], output_dir=run_root / "composed", label="composed", )
        composed_atomic_a = evaluate_model(composed_model, composed_tokenizer, atomic_a, config["evaluation"], run_root / "atomic_a_composed", "composed_atomic_a")
        composed_atomic_b = evaluate_model(composed_model, composed_tokenizer, atomic_b, config["evaluation"], run_root / "atomic_b_composed", "composed_atomic_b")
        deterministic_examples = dataset["test_iid"][:int(config["composition"].get("determinism_examples", 5))]
        deterministic_first = generate_predictions(composed_model, composed_tokenizer, deterministic_examples, int(config["composition"]["evaluation"]["max_input_tokens"]), int(config["composition"]["evaluation"]["max_new_tokens"]), int(config["composition"]["evaluation"].get("batch_size", 16)))
        deterministic_second = generate_predictions(composed_model, composed_tokenizer, deterministic_examples, int(config["composition"]["evaluation"]["max_input_tokens"]), int(config["composition"]["evaluation"]["max_new_tokens"]), int(config["composition"]["evaluation"].get("batch_size", 16)))
        determinism_check = {"n": len(deterministic_examples), "passed": deterministic_first == deterministic_second, "first": deterministic_first, "second": deterministic_second}
        LOG.info("COMPOSITION DETERMINISM CHECK | %s", determinism_check)
        composed_state = adapter_state_summary(composed_model)
        del composed_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    adapter_hashes_after = {"a": hash_directory(adapter_a_dir), "b": hash_directory(adapter_b_dir)}
    mutation_check = {"before": adapter_hashes_before, "after": adapter_hashes_after, "unchanged": adapter_hashes_before == adapter_hashes_after}
    metrics = {"composite": {split: calculate_composition_metrics(base_score=base_composite_eval[split]["accuracy"], a_score=adapter_a_eval[split]["accuracy"], b_score=adapter_b_eval[split]["accuracy"], composed_score=composed_eval[split]["accuracy"], dedicated_score=dedicated_eval[split]["accuracy"], ) for split in ("test_iid", "test_latent", "test_composition_ood")}, "composition_deficit": {split: float(dedicated_eval[split]["accuracy"] - composed_eval[split]["accuracy"]) for split in ("test_iid", "test_latent", "test_composition_ood")}, "composite_semantic_accuracy": {split: {"base": base_composite_eval[split].get("semantic_accuracy", base_composite_eval[split]["accuracy"]), "adapter_a": adapter_a_eval[split].get("semantic_accuracy", adapter_a_eval[split]["accuracy"]), "adapter_b": adapter_b_eval[split].get("semantic_accuracy", adapter_b_eval[split]["accuracy"]), "dedicated": dedicated_eval[split].get("semantic_accuracy", dedicated_eval[split]["accuracy"]), "composed": composed_eval[split].get("semantic_accuracy", composed_eval[split]["accuracy"]), } for split in ("test_iid", "test_latent", "test_composition_ood")}, "atomic": calculate_atomic_retention(standalone_a=standalone_a_atomic["test_iid"]["accuracy"], composed_a=composed_atomic_a["test_iid"]["accuracy"], base_a=base_atomic_a["test_iid"]["accuracy"], standalone_b=standalone_b_atomic["test_iid"]["accuracy"], composed_b=composed_atomic_b["test_iid"]["accuracy"], base_b=base_atomic_b["test_iid"]["accuracy"], ), "atomic_latent": calculate_atomic_retention(standalone_a=standalone_a_atomic["test_latent"]["accuracy"], composed_a=composed_atomic_a["test_latent"]["accuracy"], base_a=base_atomic_a["test_latent"]["accuracy"], standalone_b=standalone_b_atomic["test_latent"]["accuracy"], composed_b=composed_atomic_b["test_latent"]["accuracy"], base_b=base_atomic_b["test_latent"]["accuracy"], ), "cross_task_constituent": {"adapter_a_on_b": adapter_a_on_b["test_iid"]["accuracy"], "adapter_b_on_a": adapter_b_on_a["test_iid"]["accuracy"], }, "dedicated_atomic": {"dedicated_on_a": dedicated_atomic_a["test_iid"]["accuracy"], "dedicated_on_b": dedicated_atomic_b["test_iid"]["accuracy"], }, "dedicated_vs_composed_iid": float(dedicated_eval["test_iid"]["accuracy"] - composed_eval["test_iid"]["accuracy"]), "dedicated_vs_composed_latent": float(dedicated_eval["test_latent"]["accuracy"] - composed_eval["test_latent"]["accuracy"]), }
    validity = check_composite_task_validity(base_eval=base_composite_eval, adapter_a_eval=adapter_a_eval, adapter_b_eval=adapter_b_eval, joint_adapter_eval=dedicated_eval, composed_adapter_eval=composed_eval, config=config, )
    atomic_checks = {"a": compare_with_atomic_manifest(config, composition.skill_a, seed, standalone_a_atomic), "b": compare_with_atomic_manifest(config, composition.skill_b, seed, standalone_b_atomic), }
    atomic_available = [v for v in atomic_checks.values() if v.get("available", False)]
    quality_checks = {"endpoints_pass": all(v["passed"] for v in endpoint_results.values()), "adapters_unchanged": mutation_check["unchanged"], "deterministic": determinism_check["passed"], "atomic_endpoints_pass": (all(v["passed"] for v in atomic_available) if atomic_available else None)}
    validity["composition_quality_checks"] = quality_checks
    validity["all_pass"] = bool(validity["all_pass"] and all(value is not False for value in quality_checks.values()))
    result = {"run_id": run_id, "composition": asdict(composition), "seed": seed, "dataset_dir": str(dataset_dir), "dataset_manifest": dataset_manifest, "base": base_composite_eval, "base_atomic_a": base_atomic_a, "base_atomic_b": base_atomic_b, "adapter_a": adapter_a_eval, "adapter_b": adapter_b_eval, "dedicated": dedicated_eval, "dedicated_atomic_a": dedicated_atomic_a, "dedicated_atomic_b": dedicated_atomic_b, "composed": composed_eval, "atomic_standalone_a": standalone_a_atomic, "atomic_standalone_b": standalone_b_atomic, "atomic_composed_a": composed_atomic_a, "atomic_composed_b": composed_atomic_b, "adapter_a_on_b": adapter_a_on_b, "adapter_b_on_a": adapter_b_on_a, "composed_state": composed_state, "determinism": determinism_check, "endpoints": endpoint_results, "atomic_endpoint_checks": atomic_checks, "adapter_mutation": mutation_check, "metrics": metrics, "validity": validity, "completed_at": now_ts(), }
    write_json(run_root / "evaluation_metrics.json", metrics)
    write_json(run_root / "endpoint_checks.json", {"endpoints": endpoint_results, "atomic": atomic_checks, "mutation": mutation_check})
    write_json(run_root / "manifest.json", result)

    # Save deliverables: raw predictions
    raw_pred_dir = Path(config["paths"]["evaluation"]) / "raw_predictions" / composition.composition_id / f"seed_{seed}"
    raw_pred_dir.mkdir(parents=True, exist_ok=True)
    for model_dir in (run_root / "composed", run_root / "dedicated", run_root / "base", run_root / "adapter_a", run_root / "adapter_b"):
        pred_source = model_dir / "predictions"
        if pred_source.exists():
            for pfile in pred_source.glob("*.jsonl"):
                target_p = raw_pred_dir / f"{model_dir.name}_{pfile.name}"
                target_p.write_bytes(pfile.read_bytes())

    return result


def check_skill_qualification(config: Dict[str, Any], summary: Dict[str, Any], ) -> Dict[str, Any]:
    gates = config["atomic_gates"]
    metrics = summary["metrics"]

    def metric_mean(name: str) -> float:
        return float(metrics[name]["mean"])

    def metric_std(name: str) -> float:
        return float(metrics[name]["std"])

    iid_gain = metric_mean("iid_gain_over_base")
    iid_accuracy = metric_mean("adapter_test_iid_accuracy")
    latent_accuracy = metric_mean("adapter_test_latent_accuracy")
    iid_seed_std = metric_std("adapter_test_iid_accuracy")
    latent_seed_std = metric_std("adapter_test_latent_accuracy")
    iid_min = float(metrics["adapter_test_iid_accuracy"]["min"])
    latent_min = float(metrics["adapter_test_latent_accuracy"]["min"])
    checks = {"learns_above_base": {"value": iid_gain, "threshold": float(gates["minimum_iid_gain_over_base"]), "passed": (iid_gain >= float(gates["minimum_iid_gain_over_base"])), }, "minimum_iid_accuracy": {"value": iid_accuracy, "threshold": float(gates["minimum_iid_accuracy"]), "passed": (iid_accuracy >= float(gates["minimum_iid_accuracy"])), }, "generalizes_to_held_out_latents": {"value": latent_accuracy, "threshold": float(gates["minimum_latent_accuracy"]), "passed": (latent_accuracy >= float(gates["minimum_latent_accuracy"])), }, "iid_seed_stability": {"value": iid_seed_std, "threshold": float(gates["maximum_iid_seed_std"]), "passed": (iid_seed_std <= float(gates["maximum_iid_seed_std"])), }, "latent_seed_stability": {"value": latent_seed_std, "threshold": float(gates["maximum_latent_seed_std"]), "passed": (latent_seed_std <= float(gates["maximum_latent_seed_std"])), }, "minimum_iid_accuracy_per_seed": {"value": iid_min, "threshold": float(gates["minimum_per_seed_accuracy"]), "passed": (iid_min >= float(gates["minimum_per_seed_accuracy"])), }, "minimum_latent_accuracy_per_seed": {"value": latent_min, "threshold": float(gates["minimum_per_seed_latent_accuracy"]), "passed": (latent_min >= float(gates["minimum_per_seed_latent_accuracy"])), }, }
    return {"all_pass": all(check["passed"] for check in checks.values()), "checks": checks, }


def check_composite_task_validity(base_eval: Dict[str, Any], adapter_a_eval: Dict[str, Any], adapter_b_eval: Dict[str, Any], joint_adapter_eval: Dict[str, Any], composed_adapter_eval: Dict[str, Any], config: Dict[str, Any], ) -> Dict[str, Any]:
    gates = config["composition"]["task_gates"]
    max_constituent = float(gates["max_constituent_iid_accuracy"])
    minimum_joint = float(gates["minimum_joint_adapter_accuracy"])
    base = float(base_eval["test_iid"]["accuracy"])
    a = float(adapter_a_eval["test_iid"]["accuracy"])
    b = float(adapter_b_eval["test_iid"]["accuracy"])
    joint = float(joint_adapter_eval["test_iid"]["accuracy"])
    latent_a = float(adapter_a_eval["test_latent"]["accuracy"])
    latent_b = float(adapter_b_eval["test_latent"]["accuracy"])
    latent_joint = float(joint_adapter_eval["test_latent"]["accuracy"])
    composed = float(composed_adapter_eval["test_iid"]["accuracy"])
    composed_latent = float(composed_adapter_eval["test_latent"]["accuracy"])
    max_latent = float(gates.get("max_constituent_latent_accuracy", max_constituent))
    # Eligibility is determined by learnability and control quality. Dynamic
    # composition is the outcome under study, so its score must not decide
    # whether a task pair enters the comparison corpus.
    checks = {"adapter_a_cannot_solve_composite": {"value": a, "threshold": max_constituent, "passed": a <= max_constituent, }, "adapter_b_cannot_solve_composite": {"value": b, "threshold": max_constituent, "passed": b <= max_constituent, }, "adapter_a_cannot_solve_composite_latent": {"value": latent_a, "threshold": max_latent, "passed": latent_a <= max_latent, }, "adapter_b_cannot_solve_composite_latent": {"value": latent_b, "threshold": max_latent, "passed": latent_b <= max_latent, }, "joint_adapter_can_solve_composite": {"value": joint, "threshold": minimum_joint, "passed": joint >= minimum_joint, }, "joint_adapter_generalizes_latent": {"value": latent_joint, "threshold": minimum_joint, "passed": latent_joint >= minimum_joint, }, }
    return {"all_pass": all(check["passed"] for check in checks.values()), "base_accuracy": base, "adapter_a_accuracy": a, "adapter_b_accuracy": b, "joint_adapter_accuracy": joint, "joint_adapter_latent_accuracy": latent_joint, "composed_adapter_accuracy": composed, "composed_adapter_latent_accuracy": composed_latent, "adapter_a_latent_accuracy": latent_a, "adapter_b_latent_accuracy": latent_b, "dynamic_outcome": {"iid_accuracy": composed, "latent_accuracy": composed_latent}, "checks": checks, }


def run_atomic(config: Dict[str, Any], config_path: Path, run_name: str, selected_skills: Sequence[str] | None = None, ) -> None:
    configured_skills = [str(skill_id) for skill_id in config["atomic"]["skills"]]
    if selected_skills is None:
        skills = configured_skills
    else:
        skills = [str(skill_id) for skill_id in selected_skills]
        unknown_skills = [skill_id for skill_id in skills if skill_id not in SKILL_REGISTRY]
        if unknown_skills:
            raise ValueError(f"Unknown skill(s): {unknown_skills}. "
                             f"Available skills: {sorted(SKILL_REGISTRY)}")
        not_configured = [skill_id for skill_id in skills if skill_id not in configured_skills]
        if not_configured:
            raise ValueError(f"Skill(s) requested with --skills but not present "
                             f"in atomic.skills: {not_configured}")
    if not skills:
        raise ValueError("No skills selected for atomic.")
    seeds = [int(seed) for seed in config["atomic"]["seeds"]]
    qualification_dir = Path(config["paths"]["skills"])
    qualification_dir.mkdir(parents=True, exist_ok=True, )
    LOG.info("atomic skills selected: %s", ", ".join(skills), )
    all_summaries = []
    for skill_id in skills:
        LOG.info("=" * 72)
        LOG.info("QUALIFYING SKILL: %s", skill_id)
        LOG.info("=" * 72)
        manifests = []
        for seed in seeds:
            manifest = run_single_skill_seed(config=config, skill_id=skill_id, seed=seed, run_name=run_name, )
            manifests.append(manifest)
        summary = aggregate_skill_runs(skill_id=skill_id, manifests=manifests)
        gates = check_skill_qualification(config=config, summary=summary)
        report = {**summary, "gates": gates, "generated_at": now_ts(), }
        report_path = (qualification_dir / f"{skill_id}_qualification.json")
        write_json(report_path, report)
        all_summaries.append(report)
        LOG.info("Skill %s: %s", skill_id, "PASS" if gates["all_pass"] else "FAIL", )
    write_json(qualification_dir / "atomic_summary.json", {"skills": all_summaries, "all_skills_pass": all(s["gates"]["all_pass"] for s in all_summaries), }, )


def run_composition(config: Dict[str, Any], config_path: Path, run_name: str, selected_pairs: Sequence[Tuple[str, str]] | None = None, sweep_weights: bool = False, sweep_methods: bool = False, ) -> None:
    skills = [str(skill_id) for skill_id in config["atomic"]["skills"]]
    for skill_id in skills:
        get_skill(skill_id)
    if selected_pairs is None:
        pairs = generate_ordered_pairs(skills=skills, include_self_pairs=bool(config["composition"].get("include_self_pairs", False, )), )
    else:
        pairs = list(selected_pairs)
    for skill_a, skill_b in pairs:
        if skill_a not in skills: raise ValueError(f"{skill_a} is not configured in atomic")
        if skill_b not in skills: raise ValueError(f"{skill_b} is not configured in atomic")
        if (skill_a, skill_b) not in COMPOSITION_RELATIONSHIPS:
            reverse = (skill_b, skill_a)
            if reverse in COMPOSITION_RELATIONSHIPS:
                raise ValueError(f"Unsupported ordered composition for {skill_a}:{skill_b}: only {skill_b}:{skill_a} is audited. Composition order is significant because {skill_a} does not produce the structured-record schema required by {skill_b}.")
            raise ValueError(f"No audited composition relationship for {skill_a}:{skill_b}. Available: {sorted(COMPOSITION_RELATIONSHIPS)}")
    if not bool(config["composition"].get("evaluate_reverse_order", True)):
        pairs = [pair for pair in pairs if (pair[1], pair[0]) not in pairs or pair[0] < pair[1]]
    default_method = str(config["composition"].get("composition_method", "additive"))
    configured_methods = [str(value) for value in config["composition"].get("composition_methods", [default_method])]
    methods = configured_methods if sweep_methods else [default_method]
    supported_methods = set(PEFT_COMBINATION_TYPES) | SEQUENTIAL_COMPOSITION_METHODS | ROUTED_COMPOSITION_METHODS
    unknown_methods = [method_name for method_name in methods if method_name not in supported_methods]
    if unknown_methods:
        raise ValueError(f"Unknown composition method(s): {unknown_methods}. Supported: {sorted(supported_methods)}")
    configured_weight_pairs = config["composition"].get("weight_pairs")
    if configured_weight_pairs:
        all_weights = [(float(item.get("a", 0.5)), float(item.get("b", 0.5))) for item in configured_weight_pairs]
    else:
        all_weights = [(0.5, float(weight_b)) for weight_b in config["composition"].get("weights", [0.5])]
    default_weight = config["composition"].get("default_weight_pair")
    if default_weight:
        weights = [(float(default_weight.get("a", 0.5)), float(default_weight.get("b", 0.5)))]
    else:
        weights = all_weights[:1]
    if sweep_weights:
        weights = all_weights
    LOG.info("composition sweep selection | methods=%s weights=%s", methods, weights)
    all_results = []
    registry = [{"skill_a": a, "skill_b": b, "order": [a, b], "relationship_type": COMPOSITION_RELATIONSHIPS[(a, b)], "composition_methods": methods} for a, b in pairs]
    registry_path = Path(config["paths"]["compositions"]) / "registry" / "composition_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(registry_path, {"generated_at": now_ts(), "config_path": str(config_path), "relationships": registry, "methods": methods, "weights": [{"a": a, "b": b} for a, b in weights], "sweep_weights": sweep_weights, "sweep_methods": sweep_methods, "composite_sampling": config["composition"].get("composite_sampling", {})})

    eval_root = Path(config["paths"]["evaluation"])
    retention_dir = eval_root / "atomic_retention"
    perf_dir = eval_root / "composite_performance"
    interference_dir = eval_root / "interference"
    gain_dir = eval_root / "composition_gain"
    deficit_dir = eval_root / "composition_deficit"
    qual_dir = Path(config["paths"]["compositions"]) / "qualification_reports"
    specs_dir = Path(config["paths"]["compositions"]) / "specs"

    retention_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)
    interference_dir.mkdir(parents=True, exist_ok=True)
    gain_dir.mkdir(parents=True, exist_ok=True)
    deficit_dir.mkdir(parents=True, exist_ok=True)
    qual_dir.mkdir(parents=True, exist_ok=True)
    specs_dir.mkdir(parents=True, exist_ok=True)

    for skill_a, skill_b in pairs:
        for method in methods:
            for weight_a, weight_b in weights:
                composition = build_composition_spec(skill_a=skill_a, skill_b=skill_b, method=method, weight_a=weight_a, weight_b=weight_b, config=config)
                write_json(specs_dir / f"{composition.composition_id}.json", asdict(composition))
                LOG.info("Running composition: %s", composition.composition_id)
                comp_results = []
                for seed in config["atomic"]["seeds"]:
                    result = run_single_composition(config=config, composition=composition, seed=int(seed))
                    comp_results.append(result)
                    all_results.append(result)

                comp_id = composition.composition_id
                write_json(retention_dir / f"{comp_id}.json", {"composition_id": comp_id, "skill_a": composition.skill_a, "skill_b": composition.skill_b, "relationship_type": composition.relationship_type, "seeds": [res["seed"] for res in comp_results], "retention_iid": [res["metrics"]["atomic"] for res in comp_results], "retention_latent": [res["metrics"]["atomic_latent"] for res in comp_results], })
                write_json(perf_dir / f"{comp_id}.json", {"composition_id": comp_id, "skill_a": composition.skill_a, "skill_b": composition.skill_b, "relationship_type": composition.relationship_type, "seeds": [res["seed"] for res in comp_results], "composite": [res["metrics"]["composite"] for res in comp_results], "semantic": [res["metrics"]["composite_semantic_accuracy"] for res in comp_results], })
                write_json(interference_dir / f"{comp_id}.json", {"composition_id": comp_id, "skill_a": composition.skill_a, "skill_b": composition.skill_b, "relationship_type": composition.relationship_type, "seeds": [res["seed"] for res in comp_results], "interference_a": [res["metrics"]["atomic"]["interference_a"] for res in comp_results], "interference_b": [res["metrics"]["atomic"]["interference_b"] for res in comp_results], "interference_latent_a": [res["metrics"]["atomic_latent"]["interference_a"] for res in comp_results], "interference_latent_b": [res["metrics"]["atomic_latent"]["interference_b"] for res in comp_results], })
                write_json(gain_dir / f"{comp_id}.json", {"composition_id": comp_id, "skill_a": composition.skill_a, "skill_b": composition.skill_b, "relationship_type": composition.relationship_type, "seeds": [res["seed"] for res in comp_results], "gain_iid": [res["metrics"]["composite"]["test_iid"]["composition_gain"] for res in comp_results], "gain_latent": [res["metrics"]["composite"]["test_latent"]["composition_gain"] for res in comp_results], "gain_composition_ood": [res["metrics"]["composite"]["test_composition_ood"]["composition_gain"] for res in comp_results], })
                write_json(deficit_dir / f"{comp_id}.json", {"composition_id": comp_id, "skill_a": composition.skill_a, "skill_b": composition.skill_b, "relationship_type": composition.relationship_type, "seeds": [res["seed"] for res in comp_results], "deficit_iid": [res["metrics"]["composite"]["test_iid"]["composition_deficit"] for res in comp_results], "deficit_latent": [res["metrics"]["composite"]["test_latent"]["composition_deficit"] for res in comp_results], "deficit_composition_ood": [res["metrics"]["composite"]["test_composition_ood"]["composition_deficit"] for res in comp_results], })
                write_json(qual_dir / f"{comp_id}_qualification.json", {"composition_id": comp_id, "skill_a": composition.skill_a, "skill_b": composition.skill_b, "relationship_type": composition.relationship_type, "seeds": [res["seed"] for res in comp_results], "all_seeds_pass": all(res["validity"]["all_pass"] for res in comp_results), "seed_validity": [res["validity"] for res in comp_results], "generated_at": now_ts(), })

    evaluation_dir = Path(config["paths"]["evaluation"]) / "composition"
    evaluation_dir.mkdir(parents=True, exist_ok=True, )
    write_json(evaluation_dir / "composition_summary.json", {"generated_at": now_ts(), "pair_count": len(pairs), "method_count": len(methods), "weight_count": len(weights), "composition_count": len(pairs) * len(methods) * len(weights), "run_count": len(all_results), "methods": methods, "relationships": registry, "results": all_results, }, )


def evaluate_constituent_on_composite(config: Dict[str, Any], adapter_dir: Path, dataset: Dict[str, List[CompositeExample]], output_dir: Path, label: str, ) -> Dict[str, Any]:
    model, tokenizer = load_adapter_for_eval(config["model"]["name"], adapter_dir, )
    result = evaluate_composite_model(model=model, tokenizer=tokenizer, dataset=dataset, eval_cfg=config["composition"]["evaluation"], output_dir=output_dir, label=label, )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, )
    parser.add_argument("--mode", choices=["atomic", "compose"], required=True, help="Run atomic adapter training or composition evaluation.")
    parser.add_argument("--run-name", default="experiment", )
    parser.add_argument("--skills", nargs="+", default=None, )
    parser.add_argument("--composition-pairs", nargs="+", default=None, help=("Ordered pairs such as field_filtering:value_selection format_conversion:field_sorting"), )
    parser.add_argument("--sweep-weights", action="store_true", help="Run every composition weight pair configured under composition.weight_pairs", )
    parser.add_argument("--sweep-methods", action="store_true", help="Run every composition method configured under composition.composition_methods", )
    return parser.parse_args()


def parse_composition_pairs(values: Sequence[str] | None, ) -> List[Tuple[str, str]] | None:
    if values is None:
        return None
    pairs = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid composition pair {value!r}. Expected skill_a:skill_b.")
        pairs.append((parts[0], parts[1]))
    return pairs


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_yaml(config_path)
    if args.mode == "atomic":
        run_atomic(config=config, config_path=config_path, run_name=args.run_name, selected_skills=args.skills, )
        return
    if args.mode == "compose":
        run_composition(config=config, config_path=config_path, run_name=args.run_name, selected_pairs=parse_composition_pairs(args.composition_pairs), sweep_weights=args.sweep_weights, sweep_methods=args.sweep_methods, )
        return


if __name__ == "__main__":
    main()
