"""
Week 4 - HCAG-style Hierarchical Architectural Recovery on HPC
Group 3 | Hadoop MapReduce Client Core
Default model: ibm-granite/granite-34b-code-instruct-8k

This script implements the Week 4 hierarchical abstraction part inspired by HCAG:
  raw source code -> file/chunk summaries -> directory summaries -> cluster summaries

Inputs:
  - clusters_arc.csv, clusters_acdc.csv, and clusters_limbo.csv
    with columns: cluster,class,path
  - Hadoop source tree under WORK_ROOT/hadoop, or the script can clone it.

Main outputs:
  - OUTPUT_ROOT/<ALGORITHM>/file_summaries/cluster_<id>.json
  - OUTPUT_ROOT/<ALGORITHM>/directory_summaries/cluster_<id>/*.json
  - OUTPUT_ROOT/<ALGORITHM>/cluster_<id>_architecture.json
  - OUTPUT_ROOT/<ALGORITHM>/week4_cluster_descriptions.json
  - OUTPUT_ROOT/<ALGORITHM>_hierarchical_summarization_results.csv
  - OUTPUT_ROOT/week4_all_algorithms_manifest.json
  - OUTPUT_ROOT/week4_rq2_zero_shot_report.md
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "ibm-granite/granite-34b-code-instruct-8k")

WORK_ROOT = Path(
    os.environ.get(
        "WORK_ROOT",
        str(Path(os.environ.get("PC2PFS", os.environ.get("SCRATCH", "/tmp"))) / "ds4se_group3"),
    )
)

HADOOP_DIR = Path(os.environ.get("HADOOP_DIR", str(WORK_ROOT / "hadoop")))
HADOOP_REF = os.environ.get("HADOOP_REF", "rel/release-3.4.1")

SOURCE_CODE_DIR = Path(
    os.environ.get(
        "SOURCE_CODE_DIR",
        str(
            HADOOP_DIR
            / "hadoop-mapreduce-project/hadoop-mapreduce-client"
            / "hadoop-mapreduce-client-core/src/main/java/org/apache/hadoop/mapreduce"
        ),
    )
)

CLUSTERS_DIR = Path(os.environ.get("CLUSTERS_DIR", str(WORK_ROOT / "input")))
ARC_CLUSTERS_CSV = Path(os.environ.get("ARC_CLUSTERS_CSV", str(CLUSTERS_DIR / "clusters_arc.csv")))
ACDC_CLUSTERS_CSV = Path(os.environ.get("ACDC_CLUSTERS_CSV", str(CLUSTERS_DIR / "clusters_acdc.csv")))
LIMBO_CLUSTERS_CSV = Path(os.environ.get("LIMBO_CLUSTERS_CSV", str(CLUSTERS_DIR / "clusters_limbo.csv")))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", str(WORK_ROOT / "week4_hcag_output")))

# OUTPUT_DIR is reassigned per algorithm by run_algorithm().
OUTPUT_DIR = OUTPUT_ROOT

# Generation limits. Granite 34B Code has an 8k context window.
MODEL_CONTEXT_LENGTH = int(os.environ.get("MODEL_CONTEXT_LENGTH", "8192"))
SAFETY_MARGIN_TOKENS = int(os.environ.get("SAFETY_MARGIN_TOKENS", "192"))
MAX_NEW_TOKENS_FILE = int(os.environ.get("MAX_NEW_TOKENS_FILE", "512"))
MAX_NEW_TOKENS_DIR = int(os.environ.get("MAX_NEW_TOKENS_DIR", "768"))
MAX_NEW_TOKENS_CLUSTER = int(os.environ.get("MAX_NEW_TOKENS_CLUSTER", "1024"))
MAX_NEW_TOKENS_INTERMEDIATE = int(os.environ.get("MAX_NEW_TOKENS_INTERMEDIATE", "512"))

# Token budgets for chunking raw source and child summaries before prompting.
MAX_CODE_CHUNK_TOKENS = int(os.environ.get("MAX_CODE_CHUNK_TOKENS", "5200"))
MAX_CHILD_SUMMARY_TOKENS = int(os.environ.get("MAX_CHILD_SUMMARY_TOKENS", "5400"))

# Deterministic generation is better for assignment reproducibility.
DO_SAMPLE = os.environ.get("DO_SAMPLE", "false").lower() == "true"
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))

# If true, an existing checkpoint is reused instead of regenerating the summary.
RESUME = os.environ.get("RESUME", "true").lower() == "true"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str) -> None:
    sys.exit(f"[error] {message}")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(text: str) -> str:
    """Return a filesystem-safe name while keeping paths readable."""
    if text in {"", "."}:
        return "root"
    cleaned = text.replace("\\", "/").strip("/").replace("/", "__")
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in cleaned)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    if len(cleaned) > 120:
        cleaned = cleaned[:120]
    return f"{cleaned}__{digest}"


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def format_prompt(tokenizer, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"System: {system_text}\n\nUser: {user_text}\n\nAssistant:"


def input_token_limit(max_new_tokens: int) -> int:
    return max(512, MODEL_CONTEXT_LENGTH - max_new_tokens - SAFETY_MARGIN_TOKENS)


# ---------------------------------------------------------------------------
# Setup and data loading
# ---------------------------------------------------------------------------
def clone_hadoop_if_needed() -> None:
    if SOURCE_CODE_DIR.exists():
        log(f"[setup] Hadoop source found: {SOURCE_CODE_DIR}")
        return

    HADOOP_DIR.parent.mkdir(parents=True, exist_ok=True)
    log(f"[setup] Hadoop source not found. Cloning ref '{HADOOP_REF}' into {HADOOP_DIR}")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            HADOOP_REF,
            "https://github.com/apache/hadoop.git",
            str(HADOOP_DIR),
        ],
        check=True,
    )

    if not SOURCE_CODE_DIR.exists():
        fail(f"Hadoop was cloned, but SOURCE_CODE_DIR is still missing: {SOURCE_CODE_DIR}")


def load_clusters(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        fail(f"clusters.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"cluster", "class", "path"}
    missing = required - set(df.columns)
    if missing:
        fail(f"clusters.csv is missing required columns: {sorted(missing)}")

    df = df[["cluster", "class", "path"]].copy()
    df["cluster"] = df["cluster"].astype(int)
    df["class"] = df["class"].astype(str)
    df["path"] = df["path"].astype(str)
    df = df.sort_values(["cluster", "path", "class"]).reset_index(drop=True)

    log(f"[data] Loaded {len(df)} files in {df['cluster'].nunique()} clusters")
    return df


def validate_source_paths(clusters_df: pd.DataFrame) -> None:
    missing: list[str] = []
    for _, row in clusters_df.iterrows():
        if not (SOURCE_CODE_DIR / row["path"]).exists():
            missing.append(row["path"])

    if not missing:
        log("[data] All source paths exist")
        return

    report_path = OUTPUT_DIR / "missing_source_files.txt"
    save_json(report_path.with_suffix(".json"), missing)
    report_path.write_text("\n".join(missing), encoding="utf-8")
    fail(
        f"{len(missing)} files from clusters.csv were not found under {SOURCE_CODE_DIR}. "
        f"See {report_path}. The Hadoop checkout may not match the Week 3 source version."
    )


def read_source_file(relative_path: str) -> str:
    full_path = SOURCE_CODE_DIR / relative_path
    return full_path.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------
def load_model(hf_token: str):
    if not torch.cuda.is_available():
        fail("CUDA is not available. Run this inside a Slurm GPU job.")

    log(f"[model] CUDA devices visible: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        log(f"[model] GPU{i}: {props.name}, {props.total_memory / 1024**3:.1f} GiB")

    log("[model] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Keep some headroom on 40 GB A100 GPUs.
    max_memory = {i: "37GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "96GiB"

    log(f"[model] Loading {MODEL_NAME} with bfloat16 and device_map=auto")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    model.eval()

    log(
        "[model] Memory allocated: "
        + " | ".join(
            f"GPU{i}: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GiB"
            for i in range(torch.cuda.device_count())
        )
    )
    return tokenizer, model


def generate(prompt: str, tokenizer, model, max_new_tokens: int) -> str:
    first_device = next(model.parameters()).device
    limit = input_token_limit(max_new_tokens)

    original_tokens = token_count(tokenizer, prompt)
    if original_tokens > limit:
        log(f"[warn] Prompt has {original_tokens} tokens; truncating to {limit} tokens")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=limit,
    ).to(first_device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": DO_SAMPLE,
    }
    if DO_SAMPLE:
        gen_kwargs.update({"temperature": TEMPERATURE, "top_p": TOP_P})

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------
def parse_title_description(raw: str) -> tuple[str, str]:
    title = ""
    description = raw.strip()
    if "TITLE:" in raw:
        title = raw.split("TITLE:", 1)[1].splitlines()[0].strip()
    if "DESCRIPTION:" in raw:
        description = raw.split("DESCRIPTION:", 1)[1].strip()
    return title, description


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:[-']\w+)?\b", text))


def trim_to_word_limit(text: str, limit: int = 150) -> str:
    words = re.findall(r"\S+", text.strip())
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).rstrip(" ,;:") + "."


def rewrite_description_under_limit(
    cluster_id: int,
    title: str,
    description: str,
    tokenizer,
    model,
    limit: int = 150,
) -> str:
    system_text = "You are an expert software architect producing concise architecture summaries."
    user_text = f"""Rewrite the following cluster description so it is under {limit} words.

It must explicitly include:
- components and interactions
- quality attributes such as scalability, maintainability, reliability, security, or performance
- technologies used, such as Java, Hadoop MapReduce, APIs, frameworks, or tools visible in the summary

Do not invent new source-code details. Preserve the same meaning.
Return only the rewritten description, without a heading.

Cluster: {cluster_id}
Title: {title}
Current description:
{description}
"""
    prompt = format_prompt(tokenizer, system_text, user_text)
    rewritten = generate(prompt, tokenizer, model, MAX_NEW_TOKENS_INTERMEDIATE).strip()
    return trim_to_word_limit(rewritten, limit)


def split_text_by_token_budget(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split text into line-preserving chunks that fit the token budget."""
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = token_count(tokenizer, line)

        # If one physical line is huge, hard-split by characters.
        if line_tokens > max_tokens:
            if current:
                chunks.append("".join(current))
                current, current_tokens = [], 0
            approx_chars = max(1000, int(len(line) * max_tokens / max(1, line_tokens)))
            for i in range(0, len(line), approx_chars):
                part = line[i : i + approx_chars]
                chunks.append(part)
            continue

        if current and current_tokens + line_tokens > max_tokens:
            chunks.append("".join(current))
            current, current_tokens = [line], line_tokens
        else:
            current.append(line)
            current_tokens += line_tokens

    if current:
        chunks.append("".join(current))
    return chunks or [text]


def split_child_records_by_budget(child_records: list[dict], tokenizer, max_tokens: int) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    for item in child_records:
        text = f"--- {item['type'].upper()}: {item['name']} ---\n{item['summary']}\n"
        item_tokens = token_count(tokenizer, text)
        if current and current_tokens + item_tokens > max_tokens:
            chunks.append(current)
            current, current_tokens = [item], item_tokens
        else:
            current.append(item)
            current_tokens += item_tokens

    if current:
        chunks.append(current)
    return chunks or [child_records]


# ---------------------------------------------------------------------------
# HCAG-style bottom-up summarization
# ---------------------------------------------------------------------------
def summarize_code_chunk(file_name: str, chunk_index: int, total_chunks: int, code_chunk: str, tokenizer, model) -> str:
    system_text = "You are an expert software architect analysing Apache Hadoop MapReduce source code."
    user_text = f"""Summarize this raw Java source-code chunk.

This is chunk {chunk_index + 1} of {total_chunks} from file {file_name}.
Use only the code shown in this chunk.

Return exactly these four sections:

KEY FUNCTIONALITY: main purpose visible in this chunk
CORE LOGIC: key algorithms or processing patterns visible in this chunk
INPUTS/OUTPUTS: data flowing in and out
DEPENDENCIES: key imported, parent, or collaborator classes visible in this chunk

Code chunk:
{code_chunk}
"""
    prompt = format_prompt(tokenizer, system_text, user_text)
    return generate(prompt, tokenizer, model, MAX_NEW_TOKENS_FILE)


def summarize_file_from_chunk_summaries(file_name: str, chunk_summaries: list[str], tokenizer, model) -> str:
    summaries_text = "\n\n".join(
        f"--- chunk {i + 1} ---\n{summary}" for i, summary in enumerate(chunk_summaries)
    )
    system_text = "You are an expert software architect analysing Apache Hadoop MapReduce source code."
    user_text = f"""Combine these chunk summaries into one file-level semantic summary.

File: {file_name}

Return exactly these four sections:

KEY FUNCTIONALITY: main purpose of this class or file
CORE LOGIC: key algorithms or processing patterns
INPUTS/OUTPUTS: data flowing in and out
DEPENDENCIES: key imported, parent, or collaborator classes

Chunk summaries:
{summaries_text}
"""
    prompt = format_prompt(tokenizer, system_text, user_text)
    return generate(prompt, tokenizer, model, MAX_NEW_TOKENS_FILE)


def summarize_file(file_name: str, code: str, tokenizer, model) -> dict:
    code_chunks = split_text_by_token_budget(code, tokenizer, MAX_CODE_CHUNK_TOKENS)

    if len(code_chunks) == 1:
        summary = summarize_code_chunk(file_name, 0, 1, code_chunks[0], tokenizer, model)
        return {
            "summary": summary,
            "chunked": False,
            "number_of_code_chunks": 1,
            "chunk_summaries": [],
        }

    chunk_summaries = []
    log(f"    file {file_name}: splitting raw code into {len(code_chunks)} chunks")
    for i, chunk in enumerate(code_chunks):
        log(f"      summarizing code chunk {i + 1}/{len(code_chunks)}")
        chunk_summaries.append(summarize_code_chunk(file_name, i, len(code_chunks), chunk, tokenizer, model))

    final_summary = summarize_file_from_chunk_summaries(file_name, chunk_summaries, tokenizer, model)
    return {
        "summary": final_summary,
        "chunked": True,
        "number_of_code_chunks": len(code_chunks),
        "chunk_summaries": chunk_summaries,
    }


def summarize_intermediate_children(
    node_type: str,
    node_name: str,
    chunk_index: int,
    child_records: list[dict],
    tokenizer,
    model,
) -> dict:
    summaries_text = "\n\n".join(
        f"--- {item['type'].upper()}: {item['name']} ---\n{item['summary']}" for item in child_records
    )
    system_text = "You are an expert software architect creating hierarchical software architecture summaries."
    user_text = f"""Create an intermediate summary for part of a larger {node_type} node.

Node: {node_name}
Chunk index: {chunk_index}

Use only the child summaries below. Do not invent source-code details.
Explain the shared responsibility, collaboration, and visible architectural role.

Child summaries:
{summaries_text}

Return 4 to 6 concise sentences.
"""
    prompt = format_prompt(tokenizer, system_text, user_text)
    raw = generate(prompt, tokenizer, model, MAX_NEW_TOKENS_INTERMEDIATE)
    return {
        "type": "intermediate",
        "name": f"{node_name} chunk {chunk_index}",
        "summary": raw,
        "number_of_children": len(child_records),
    }


def compress_children_if_needed(
    node_type: str,
    node_name: str,
    child_records: list[dict],
    tokenizer,
    model,
) -> tuple[list[dict], list[dict]]:
    """Compress a large child list into intermediate summaries if needed."""
    chunks = split_child_records_by_budget(child_records, tokenizer, MAX_CHILD_SUMMARY_TOKENS)
    if len(chunks) == 1:
        return child_records, []

    log(f"    {node_type} {node_name}: compressing {len(child_records)} children into {len(chunks)} intermediate summaries")
    intermediate_records = []
    for i, chunk in enumerate(chunks):
        intermediate_records.append(summarize_intermediate_children(node_type, node_name, i, chunk, tokenizer, model))
    return intermediate_records, intermediate_records


def summarize_directory_node(
    cluster_id: int,
    directory_path: str,
    child_records: list[dict],
    tokenizer,
    model,
) -> dict:
    prompt_children, intermediate_records = compress_children_if_needed(
        "directory", directory_path, child_records, tokenizer, model
    )

    summaries_text = "\n\n".join(
        f"--- {item['type'].upper()}: {item['name']} ---\n{item['summary']}" for item in prompt_children
    )

    system_text = "You are an expert software architect analysing Hadoop MapReduce architecture."
    user_text = f"""Generate a directory-level architectural abstraction.

Cluster: {cluster_id}
Directory: {directory_path}

Use only the child summaries below. These children may be file summaries or already-summarized subdirectories.
Do not use raw source code.

Explain:
- the responsibility of this directory
- how the files or child directories collaborate
- the architectural role or pattern of this directory

Child summaries:
{summaries_text}

Output format exactly as:
TITLE: <maximum 10 words>

DESCRIPTION:
<4 to 6 concise sentences>
"""
    prompt = format_prompt(tokenizer, system_text, user_text)
    raw = generate(prompt, tokenizer, model, MAX_NEW_TOKENS_DIR)
    title, description = parse_title_description(raw)

    return {
        "type": "directory",
        "cluster_id": int(cluster_id),
        "directory": directory_path,
        "title": title,
        "description": description,
        "summary": f"TITLE: {title}\nDESCRIPTION: {description}",
        "number_of_children": len(child_records),
        "number_of_intermediate_summaries": len(intermediate_records),
        "children": [
            {"type": item["type"], "name": item["name"]} for item in child_records
        ],
        "intermediate_summaries": intermediate_records,
        "raw_response": raw,
    }


def summarize_cluster_node(
    cluster_id: int,
    top_records: list[dict],
    number_of_files: int,
    number_of_directories: int,
    tokenizer,
    model,
) -> dict:
    prompt_children, intermediate_records = compress_children_if_needed(
        "cluster", str(cluster_id), top_records, tokenizer, model
    )

    summaries_text = "\n\n".join(
        f"--- {item['type'].upper()}: {item['name']} ---\n{item['summary']}" for item in prompt_children
    )

    system_text = "You are an expert software architect analysing Hadoop MapReduce architecture."
    user_text = f"""Generate the final cluster-level architectural description using zero-shot prompting.

Cluster: {cluster_id}

Use only the directory-level summaries below. Do not use raw source code and do not invent details.

The DESCRIPTION must explicitly include all of the following:
- Components and interactions: explain how the distinct parts of the cluster work together.
- Quality attributes: mention relevant non-functional qualities achieved by this architecture, such as scalability, maintainability, reliability, security, or performance.
- Technology used: mention technologies, frameworks, programming languages, APIs, or tools identified in the summaries.
- Conciseness: keep the description under 150 words.

Also explain the architectural role or pattern where visible, such as coordinator, worker, data model, utility, adapter, factory, or committer.

Directory summaries:
{summaries_text}

Output format exactly as:
TITLE: <maximum 10 words>

DESCRIPTION:
<under 150 words; include components/interactions, quality attributes, and technologies used>
"""
    prompt = format_prompt(tokenizer, system_text, user_text)
    raw = generate(prompt, tokenizer, model, MAX_NEW_TOKENS_CLUSTER)
    title, description = parse_title_description(raw)

    if word_count(description) > 150:
        log(f"    cluster {cluster_id}: description has {word_count(description)} words; rewriting under 150 words")
        description = rewrite_description_under_limit(cluster_id, title, description, tokenizer, model, 150)

    description = trim_to_word_limit(description, 150)

    return {
        "cluster_id": int(cluster_id),
        "title": title,
        "description": description,
        "description_word_count": word_count(description),
        "number_of_files": int(number_of_files),
        "number_of_directories": int(number_of_directories),
        "number_of_intermediate_summaries": len(intermediate_records),
        "top_nodes": [{"type": item["type"], "name": item["name"]} for item in top_records],
        "intermediate_summaries": intermediate_records,
        "raw_response": raw,
    }


# ---------------------------------------------------------------------------
# Directory tree construction
# ---------------------------------------------------------------------------
def direct_parent(path_text: str) -> str:
    path = Path(path_text)
    parent = str(path.parent).replace("\\", "/")
    return "." if parent in {"", "."} else parent


def direct_child_dirs(parent: str, all_dirs: set[str]) -> list[str]:
    result = []
    parent_path = Path(parent)
    for directory in all_dirs:
        if directory == parent:
            continue
        if direct_parent(directory) == str(parent_path).replace("\\", "/") or (
            parent == "." and direct_parent(directory) == "."
        ):
            result.append(directory)
    return sorted(result)


def collect_directories(paths: list[str]) -> set[str]:
    dirs: set[str] = {"."}
    for rel_path in paths:
        directory = str(Path(rel_path).parent).replace("\\", "/")
        if directory in {"", "."}:
            dirs.add(".")
            continue
        p = Path(directory)
        parts = p.parts
        for i in range(1, len(parts) + 1):
            dirs.add(str(Path(*parts[:i])).replace("\\", "/"))
    return dirs


def summarize_directories_for_cluster(
    cluster_id: int,
    file_records: list[dict],
    tokenizer,
    model,
) -> tuple[list[dict], dict[str, dict]]:
    directory_to_files: dict[str, list[dict]] = {}
    for item in file_records:
        directory = str(Path(item["path"]).parent).replace("\\", "/")
        if directory in {"", "."}:
            directory = "."
        directory_to_files.setdefault(directory, []).append(item)

    all_dirs = collect_directories([item["path"] for item in file_records])
    ordered_dirs = sorted(all_dirs, key=lambda d: len(Path(d).parts), reverse=True)

    dir_results: dict[str, dict] = {}
    cluster_dir = OUTPUT_DIR / "directory_summaries" / f"cluster_{cluster_id}"

    for directory in ordered_dirs:
        dir_path = cluster_dir / f"{safe_name(directory)}.json"
        if RESUME and dir_path.exists():
            record = load_json(dir_path)
            dir_results[directory] = record
            log(f"    using existing directory summary: {directory}")
            continue

        child_records: list[dict] = []

        for file_item in sorted(directory_to_files.get(directory, []), key=lambda x: x["file_name"]):
            child_records.append(
                {
                    "type": "file",
                    "name": file_item["file_name"],
                    "summary": file_item["summary"],
                }
            )

        for child_dir in direct_child_dirs(directory, all_dirs):
            if child_dir in dir_results:
                child = dir_results[child_dir]
                child_records.append(
                    {
                        "type": "directory",
                        "name": child_dir,
                        "summary": child["summary"],
                    }
                )

        if not child_records:
            continue

        log(f"    summarizing directory: {directory} ({len(child_records)} children)")
        record = summarize_directory_node(cluster_id, directory, child_records, tokenizer, model)
        save_json(dir_path, record)
        dir_results[directory] = record

    # The root directory is the direct architectural parent of the cluster.
    if "." in dir_results:
        top_records = [
            {
                "type": "directory",
                "name": ".",
                "summary": dir_results["."]["summary"],
            }
        ]
    else:
        # Fallback: use all shallowest directories if root was not created.
        shallowest = sorted(all_dirs, key=lambda d: len(Path(d).parts))[:10]
        top_records = [
            {"type": "directory", "name": d, "summary": dir_results[d]["summary"]}
            for d in shallowest
            if d in dir_results
        ]

    return top_records, dir_results


# ---------------------------------------------------------------------------
# Existing file-summary checkpoints
# ---------------------------------------------------------------------------
def load_existing_file_summaries(cluster_id: int) -> list[dict]:
    path = OUTPUT_DIR / "file_summaries" / f"cluster_{cluster_id}.json"
    if not (RESUME and path.exists()):
        return []
    return load_json(path)


def save_file_summaries(cluster_id: int, records: list[dict]) -> None:
    save_json(OUTPUT_DIR / "file_summaries" / f"cluster_{cluster_id}.json", records)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def write_required_algorithm_csv(algorithm: str, clusters_df: pd.DataFrame, cluster_results: list[dict]) -> Path:
    """Write the exact Week 4 submission CSV for one clustering algorithm."""
    result_by_cluster = {int(item["cluster_id"]): item for item in cluster_results}
    csv_path = OUTPUT_ROOT / f"{algorithm}_hierarchical_summarization_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster_ID", "files", "title", "description"])
        writer.writeheader()

        for cluster_id in sorted(clusters_df["cluster"].unique()):
            files = sorted(clusters_df.loc[clusters_df["cluster"] == cluster_id, "path"].astype(str).unique())
            result = result_by_cluster.get(int(cluster_id), {})
            writer.writerow(
                {
                    "cluster_ID": int(cluster_id),
                    "files": "; ".join(files),
                    "title": result.get("title", ""),
                    "description": result.get("description", ""),
                }
            )

    return csv_path


def write_rq2_report() -> Path:
    report_path = OUTPUT_ROOT / "week4_rq2_zero_shot_report.md"
    report_path.write_text(
        """# RQ2 Methodology: Zero-Shot Hierarchical Summarization

## Research Question

**RQ2:** How do prompting techniques impact the LLM's ability to accurately describe architectural components based on source code?

## Prompting Method Used

This Week 4 pipeline uses a **zero-shot prompting methodology**. No examples are included in the prompt. The LLM is instructed directly to summarize source-code files and then aggregate those summaries into higher-level architectural descriptions.

## Hierarchical Process

1. **Leaf/file level:** each Java file is summarized from raw source code using a prompt requesting key functionality, core logic, inputs/outputs, and dependencies.
2. **Directory level:** summaries from files and already-summarized subdirectories are aggregated into directory-level architectural abstractions.
3. **Cluster level:** directory-level summaries are aggregated into the final cluster title and high-level description.

## Required Cluster Description Constraints

Each final cluster description is instructed to include:

- **Components and interactions:** how the distinct parts of the cluster work together.
- **Quality attributes:** non-functional qualities such as scalability, maintainability, reliability, security, or performance.
- **Technology used:** Java, Hadoop MapReduce, APIs, frameworks, or other visible technologies.
- **Conciseness:** fewer than 150 words.

## Clustering Algorithms Covered

The same zero-shot hierarchical summarization process is applied independently to all three clustering algorithms:

- ARC
- ACDC
- LIMBO

The input CSV files for ACDC and LIMBO were prepared as file-level clusters (one row per `.java` file) prior to running this pipeline, as required by the hierarchical summarization methodology. Raw ACDC and LIMBO output can produce inner-class based entries; those were resolved to their enclosing source files before the CSVs were finalised.

## Output Format

The final submission contains one CSV file per clustering algorithm:

- `ARC_hierarchical_summarization_results.csv`
- `ACDC_hierarchical_summarization_results.csv`
- `LIMBO_hierarchical_summarization_results.csv`

Each CSV contains exactly:

```csv
cluster_ID,files,title,description
```
""",
        encoding="utf-8",
    )
    return report_path


def run_algorithm(algorithm: str, csv_path: Path, tokenizer, model) -> dict[str, Any]:
    global OUTPUT_DIR
    OUTPUT_DIR = OUTPUT_ROOT / algorithm
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "file_summaries").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "directory_summaries").mkdir(parents=True, exist_ok=True)

    log("\n" + "=" * 72)
    log(f"[algorithm] {algorithm}")
    log(f"[algorithm] CSV: {csv_path}")
    log(f"[algorithm] Output: {OUTPUT_DIR}")
    log("=" * 72)

    clusters_df = load_clusters(csv_path)
    validate_source_paths(clusters_df)

    unique_clusters = sorted(clusters_df["cluster"].unique())
    summaries_by_cluster: dict[int, list[dict]] = {}

    log("\n[phase 1] Leaf-node file summaries")
    for cluster_id in unique_clusters:
        existing = load_existing_file_summaries(cluster_id)
        done_classes = {item["class"] for item in existing}
        summaries_by_cluster[cluster_id] = existing

        cluster_files = clusters_df[clusters_df["cluster"] == cluster_id]
        log(f"[phase 1] {algorithm} cluster {cluster_id}: {len(done_classes)}/{len(cluster_files)} already summarized")

        for _, row in cluster_files.iterrows():
            if row["class"] in done_classes:
                continue

            rel_path = row["path"]
            file_name = Path(rel_path).name
            log(f"  summarizing file: algorithm={algorithm} cluster={cluster_id} path={rel_path}")

            code = read_source_file(rel_path)
            file_result = summarize_file(file_name, code, tokenizer, model)

            record = {
                "algorithm": algorithm,
                "cluster": int(cluster_id),
                "class": row["class"],
                "path": rel_path,
                "file_name": file_name,
                "directory": str(Path(rel_path).parent).replace("\\", "/"),
                "summary": file_result["summary"],
                "chunked": file_result["chunked"],
                "number_of_code_chunks": file_result["number_of_code_chunks"],
                "chunk_summaries": file_result["chunk_summaries"],
            }
            summaries_by_cluster[cluster_id].append(record)
            save_file_summaries(cluster_id, summaries_by_cluster[cluster_id])

    log("\n[phase 2] Bottom-up directory summaries")
    cluster_results: list[dict] = []
    hierarchy_manifest: dict[str, Any] = {"algorithm": algorithm, "clusters": []}

    for cluster_id in unique_clusters:
        file_records = summaries_by_cluster[cluster_id]
        if not file_records:
            log(f"  skipping cluster {cluster_id}: no file summaries")
            continue

        log(f"[phase 2] {algorithm} cluster {cluster_id}: {len(file_records)} files")
        top_records, dir_results = summarize_directories_for_cluster(
            cluster_id, file_records, tokenizer, model
        )

        log(f"[phase 3] {algorithm} cluster {cluster_id}: final cluster summary from {len(top_records)} top node(s)")
        result = summarize_cluster_node(
            cluster_id=cluster_id,
            top_records=top_records,
            number_of_files=len(file_records),
            number_of_directories=len(dir_results),
            tokenizer=tokenizer,
            model=model,
        )
        result["algorithm"] = algorithm
        result["files"] = sorted(item["path"] for item in file_records)
        cluster_results.append(result)
        save_json(OUTPUT_DIR / f"cluster_{cluster_id}_architecture.json", result)
        log(f"    title: {result['title']}")
        log(f"    description words: {result['description_word_count']}")

        hierarchy_manifest["clusters"].append(
            {
                "cluster_id": int(cluster_id),
                "number_of_files": len(file_records),
                "number_of_directories": len(dir_results),
                "directories": sorted(dir_results.keys()),
            }
        )

    log("\n[phase 4] Writing final outputs")
    save_json(OUTPUT_DIR / "week4_cluster_descriptions.json", cluster_results)
    save_json(OUTPUT_DIR / "hierarchy_manifest.json", hierarchy_manifest)

    # Internal detailed CSV, useful for debugging. The required submission CSV is written separately.
    detailed_csv_path = OUTPUT_DIR / "week4_cluster_descriptions.csv"
    with detailed_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "cluster_id",
                "title",
                "description",
                "description_word_count",
                "number_of_files",
                "number_of_directories",
            ],
        )
        writer.writeheader()
        for item in cluster_results:
            writer.writerow(
                {
                    "algorithm": algorithm,
                    "cluster_id": item["cluster_id"],
                    "title": item["title"],
                    "description": item["description"],
                    "description_word_count": item["description_word_count"],
                    "number_of_files": item["number_of_files"],
                    "number_of_directories": item["number_of_directories"],
                }
            )

    required_csv_path = write_required_algorithm_csv(algorithm, clusters_df, cluster_results)

    report_path = OUTPUT_DIR / "week4_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Week 4 - HCAG-style LLM-Based Architectural Recovery\n")
        f.write(f"Algorithm: {algorithm}\n")
        f.write(f"Group 3 - Hadoop MapReduce Client Core\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Files analyzed: {len(clusters_df)}\n")
        f.write(f"Clusters: {len(cluster_results)}\n")
        f.write("Prompting method: zero-shot\n")
        f.write("=" * 72 + "\n\n")
        for item in cluster_results:
            f.write(f"Cluster {item['cluster_id']} - {item['title']}\n")
            f.write(f"Files: {item['number_of_files']}\n")
            f.write(f"Directories: {item['number_of_directories']}\n")
            f.write(f"Description words: {item['description_word_count']}\n")
            f.write(f"Description: {item['description']}\n\n")

    log(f"\n[done] {algorithm} results written")
    log(f"  Required CSV : {required_csv_path}")
    log(f"  Detailed CSV : {detailed_csv_path}")
    log(f"  JSON         : {OUTPUT_DIR / 'week4_cluster_descriptions.json'}")
    log(f"  TXT          : {report_path}")
    log(f"  Hierarchy    : {OUTPUT_DIR / 'hierarchy_manifest.json'}")

    return {
        "algorithm": algorithm,
        "input_csv": str(csv_path),
        "output_dir": str(OUTPUT_DIR),
        "required_csv": str(required_csv_path),
        "detailed_csv": str(detailed_csv_path),
        "number_of_files": int(len(clusters_df)),
        "number_of_clusters": int(len(cluster_results)),
    }


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        fail("HF_TOKEN is not set. Submit with: HF_TOKEN=hf_... sbatch week4_hcag_all_algorithms_run.sh")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    log(f"[setup] Work root: {WORK_ROOT}")
    log(f"[setup] Output root: {OUTPUT_ROOT}")
    log(f"[setup] Source code dir: {SOURCE_CODE_DIR}")
    log(f"[setup] ARC CSV: {ARC_CLUSTERS_CSV}")
    log(f"[setup] ACDC CSV: {ACDC_CLUSTERS_CSV}")
    log(f"[setup] LIMBO CSV: {LIMBO_CLUSTERS_CSV}")

    clone_hadoop_if_needed()

    tokenizer, model = load_model(hf_token)

    algorithm_inputs = [
        ("ARC", ARC_CLUSTERS_CSV),
        ("ACDC", ACDC_CLUSTERS_CSV),
        ("LIMBO", LIMBO_CLUSTERS_CSV),
    ]

    all_results = []
    for algorithm, csv_path in algorithm_inputs:
        if not csv_path.exists():
            fail(f"Missing {algorithm} clusters CSV: {csv_path}")
        all_results.append(run_algorithm(algorithm, csv_path, tokenizer, model))

    manifest_path = OUTPUT_ROOT / "week4_all_algorithms_manifest.json"
    save_json(manifest_path, {"model": MODEL_NAME, "prompting_method": "zero-shot", "algorithms": all_results})
    rq2_path = write_rq2_report()

    log("\n" + "=" * 72)
    log("[done] All Week 4 hierarchical summarization outputs written")
    log(f"  Manifest  : {manifest_path}")
    log(f"  RQ2 report: {rq2_path}")
    for item in all_results:
        log(f"  {item['algorithm']} CSV: {item['required_csv']}")
    log("=" * 72)


if __name__ == "__main__":
    main()
