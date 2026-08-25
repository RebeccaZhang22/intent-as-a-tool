#!/usr/bin/env python3
"""Collect full reasoning traces and truncated tool-call probabilities."""

import argparse
import json
import os
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Any, Dict, List

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.reasoning_splitter import split_reasoning_steps_by_ids
from pipeline.truncation_engine import TruncationEngine

MISALIGNMENT_SCENARIOS = {"blackmail", "leaking", "murder", "all", "clean"}


def load_misalignment_prompts(prompts_dir: str) -> List[Dict]:
    items = []
    for cdir in sorted(Path(prompts_dir).iterdir()):
        if not cdir.is_dir():
            continue
        if cdir.name == "template":
            continue
        if not all((cdir / f).exists() for f in
                   ["system_prompt.txt", "user_prompt.txt",
                    "email_content.txt", "metadata.json"]):
            print(f"Warning: missing files in {cdir.name}, skipping")
            continue
        metadata = json.loads((cdir / "metadata.json").read_text())
        user_prompt   = (cdir / "user_prompt.txt").read_text()
        email_content   = (cdir / "email_content.txt").read_text()
        condition_id  = metadata.get("condition_id", cdir.name)
        full_user_prompt = user_prompt + "\n\n" + email_content
        items.append({
            "item_id":              condition_id,
            "scenario":             metadata.get("scenario", condition_id.split("_")[0]),
            "goal_type":            metadata.get("goal_type", "none"),
            "goal_value":           metadata.get("goal_value", "none"),
            "urgency_type":         metadata.get("urgency_type", "none"),
            "system_prompt":        (cdir / "system_prompt.txt").read_text(),
            "user_prompt":          user_prompt,
            "email_content":        email_content,
            "full_user_prompt":     full_user_prompt,
        })
    return items


def filter_items_for_scenario(items: List[Dict], scenario: str) -> List[Dict]:
    if scenario not in {"all", "clean"}:
        return [it for it in items if it["scenario"] == scenario]
    return [{**it, "source_scenario": it["scenario"], "scenario": scenario} for it in items]


def filter_records_for_scenario(records: List[Dict], scenario: str) -> List[Dict]:
    if scenario not in {"all", "clean"}:
        return [r for r in records if r["scenario"] == scenario]
    return [{**r, "source_scenario": r.get("source_scenario", r.get("scenario")), "scenario": scenario} for r in records]


def load_stage1_records(path: str) -> List[Dict]:
    """Load all JSONL records from a file or a directory of JSONL files."""
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    records = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
    return records


def stage1_sample(item: dict, engine: TruncationEngine,
                  sample_index: int, max_retries: int = 3) -> dict:
    """Run one item/sample pair and retain token IDs for Stage 2.

    Retry up to ``max_retries`` times on failure.
    """
    scenario = item["scenario"]
    base = {
        "item_id":       item["item_id"],
        "sample_index":  sample_index,
        "scenario":      scenario,
        "model":         engine.model,
    }

    system_prompt = item["system_prompt"]

    base.update({
        "condition_id":         item["item_id"],
        "goal_type":            item.get("goal_type", "none"),
        "goal_value":           item.get("goal_value", "none"),
        "urgency_type":         item.get("urgency_type", "none"),
        "system_prompt":        system_prompt,
        "user_prompt":          item.get("user_prompt", ""),
        "email_content":        item.get("email_content", ""),
    })

    if engine.scenario != scenario:
        engine.set_scenario(scenario)

    res = None
    for attempt in range(max_retries):
        res = engine.run_complete_inference(system_prompt=system_prompt, full_user_prompt=item["full_user_prompt"])

        if not res.get("error"):
            break

        if attempt < max_retries - 1:
            continue

    if res.get("error"):
        return {**base, "error": res["error"], "num_attempts": max_retries,
                "metadata": {"timestamp": datetime.now().isoformat()}}

    reasoning_ids = res.get("reasoning_ids", [])
    sentence_split_indices = []
    paragraph_split_indices = []

    if engine.tokenizer is not None and reasoning_ids:
        try:
            sentence_split_indices = split_reasoning_steps_by_ids(
                reasoning_ids, engine.tokenizer, granularity="sentence"
            )
            paragraph_split_indices = split_reasoning_steps_by_ids(
                reasoning_ids, engine.tokenizer, granularity="paragraph"
            )
        except Exception:
            sentence_split_indices = []
            paragraph_split_indices = []

    return {
        **base,
        "complete_inference": {
            "reasoning":   res.get("reasoning", ""),
            "tool_calls":  res.get("tool_calls", []),
            "termination": res.get("termination", "unknown"),
        },
        "input_token_ids": res.get("input_token_ids", []),
        "reasoning_ids":   reasoning_ids,
        "sentence_split_indices": sentence_split_indices,
        "paragraph_split_indices": paragraph_split_indices,
        "metadata":        {"timestamp": datetime.now().isoformat()},
        "num_attempts":    res.get("num_attempts", 1),
    }


def stage2_sample(record: dict, engine: TruncationEngine,
                  granularity: str,
                  verbose: bool = False, worker_id: int = 0, batch_size: int = 32) -> dict:
    """
    Truncated decoding for one stage-1 record.

    At each truncation position, score the first-token probability of every
    candidate tool name.
    """
    scenario        = record["scenario"]
    input_token_ids = record.get("input_token_ids", [])
    reasoning_ids   = record.get("reasoning_ids", [])

    out = {k: v for k, v in record.items()}

    if "paragraph_split_indices" not in out and reasoning_ids and engine.tokenizer:
        try:
            out["paragraph_split_indices"] = split_reasoning_steps_by_ids(
                reasoning_ids, engine.tokenizer, granularity="paragraph"
            )
        except Exception:
            out["paragraph_split_indices"] = []

    if engine.scenario != scenario:
        engine.set_scenario(scenario)

    split_indices = split_reasoning_steps_by_ids(
        reasoning_ids, engine.tokenizer, granularity=granularity)

    if not split_indices:
        out["truncated_predictions"] = []
        out["error"] = "No reasoning steps found"
        out["metadata"] = {
            **out.get("metadata", {}),
            "num_truncation_points":  0,
            "truncation_granularity": granularity,
        }
        return out

    with tqdm(
        total=len(split_indices),
        desc=f"  [trace {record['item_id']}]",
        unit="pos",
        position=worker_id + 1,
        leave=False,
        disable=not verbose,
    ) as pbar:
        preds = engine.run_truncated_trace_inference_by_indices(
            input_token_ids=input_token_ids,
            reasoning_ids=reasoning_ids,
            split_indices=split_indices,
            batch_size=batch_size,
            progress_bar=pbar,
        )

    out["truncated_predictions"] = preds
    out["metadata"] = {
        **out.get("metadata", {}),
        "num_truncation_points":  len(split_indices),
        "truncation_granularity": granularity,
    }
    return out


def build_engine_pool(args):
    load_tokenizer = getattr(args, "load_tokenizer", True)
    print("\nInitializing engine" + (" + tokenizer..." if load_tokenizer else " (tokenizer disabled)..."))
    engine_kw = dict(
        model=args.model,
        scenario=args.scenario,
        inference_type=args.inference_type,
        temperature=args.temperature,
        tool_mode=getattr(args, "tool_mode", "neutral"),
        verbose=args.verbose,
        load_tokenizer=load_tokenizer,
    )
    primary = TruncationEngine(**engine_kw)
    pool: Dict[int, TruncationEngine] = {0: primary}
    lock = threading.Lock()

    def get(wid: int) -> TruncationEngine:
        if wid not in pool:
            with lock:
                if wid not in pool:
                    kw = dict(engine_kw, verbose=False)
                    pool[wid] = TruncationEngine(**kw)
        return pool[wid]

    return get


def load_processed_keys(output_dir: str, scenarios: List[str] = None) -> set:
    """Load processed keys from output directory.

    If scenarios specified, only read {scenario}.jsonl files.
    Otherwise, read every JSONL file in the directory.
    """
    keys: set = set()
    p = Path(output_dir)

    if scenarios:
        # Read only the requested scenario files.
        for st in scenarios:
            f = p / f"{st}.jsonl"
            if f.exists():
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            d = json.loads(line)
                            keys.add((d.get("item_id", ""), d.get("sample_index", 0)))
    else:
        files = sorted(p.glob("*.jsonl")) if p.exists() else []
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        d = json.loads(line)
                        keys.add((d.get("item_id", ""), d.get("sample_index", 0)))
    return keys


def open_output_files(output_dir: str, scenarios: List[str], resume: bool):
    """Open per-scenario JSONL files inside one output directory."""
    mode = "a" if resume else "w"
    file_locks: Dict[str, threading.Lock] = {}
    open_files: Dict[str, Any] = {}

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for st in scenarios:
        if st not in file_locks:
            file_locks[st] = threading.Lock()
            open_files[st] = open(
                Path(output_dir) / f"{st}.jsonl", mode, encoding="utf-8")

    return file_locks, open_files


def make_writer(file_locks, open_files):
    def write(result: dict):
        key = result.get("scenario")
        with file_locks[key]:
            open_files[key].write(json.dumps(result, ensure_ascii=False) + "\n")
            open_files[key].flush()
    return write


def run_stage1(args, items: List[Dict], get_engine, output_dir: str):
    scenarios = list({it["scenario"] for it in items})
    processed_keys = load_processed_keys(output_dir, scenarios) if args.resume else set()
    if processed_keys:
        print(f"Resume: {len(processed_keys)} records already in output")

    # Group pending sample indices by item ID.
    groups: OrderedDict = OrderedDict()
    for item in items:
        pending = [i for i in range(args.num_samples)
                   if (item["item_id"], i) not in processed_keys]
        if pending:
            groups[item["item_id"]] = (item, pending)

    total = sum(len(s) for _, s in groups.values())
    if total == 0:
        print("All work already done (resume).")
        return

    print(f"Items: {len(items)}  |  Samples/item: {args.num_samples}  |  "
          f"Remaining: {total}  |  Workers: {args.max_workers}")

    scenarios = list({item["scenario"] for item in items})
    file_locks, open_files = open_output_files(output_dir, scenarios, args.resume)
    write = make_writer(file_locks, open_files)

    total_ok = total_fail = 0
    ctr_lock  = threading.Lock()

    def process_group(wid, item, sample_indices, pbar):
        nonlocal total_ok, total_fail
        engine = get_engine(wid)
        for sidx in sample_indices:
            res = stage1_sample(item, engine, sidx)
            write(res)
            ok = not res.get("error")
            with ctr_lock:
                if ok: total_ok += 1
                else:  total_fail += 1
            if args.verbose:
                num_attempts = res.get("num_attempts", 1)
                status = 'OK' if ok else 'FAIL'
                if num_attempts > 1:
                    status += f" ({num_attempts} attempts)"
                tqdm.write(f"  [{status}] "
                           f"{str(item['item_id'])[:60]} #{sidx}")
            pbar.update(1)

    try:
        with tqdm(total=total, desc="Stage-1", unit="trace") as pbar:
            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                futs = {
                    ex.submit(process_group,
                              i % args.max_workers, item, sidxs, pbar): iid
                    for i, (iid, (item, sidxs)) in enumerate(groups.items())
                }
                for fut in as_completed(futs):
                    try:
                        fut.result()
                    except Exception as e:
                        tqdm.write(f"  [ERROR] {str(futs[fut])[:60]}: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        for fh in open_files.values():
            fh.close()

    print(f"\nDone!  Success: {total_ok}  Failed: {total_fail}")
    print(f"Output dir: {output_dir}")


def run_stage2(args, records: List[Dict], get_engine, output_dir: str):
    # Resume against the current scenario output only.
    processed_keys = load_processed_keys(output_dir, [args.scenario]) if args.resume else set()

    if processed_keys:
        print(f"Resume: {len(processed_keys)} {args.scenario} records already in output")

    # Build (item_id, sample_index, record) work items.
    pending_samples: List[tuple] = []
    for rec in records:
        iid, sidx = rec["item_id"], rec["sample_index"]
        if (iid, sidx) in processed_keys:
            continue
        pending_samples.append((iid, sidx, rec))

    total = len(pending_samples)
    if total == 0:
        print("All work already done.")
        return

    print(f"Stage-1 records: {len(records)}  |  Remaining samples: {total}  |  "
          f"Workers: {args.max_workers}")

    scenarios = list({r["scenario"] for r in records})
    file_locks, open_files = open_output_files(output_dir, scenarios, args.resume)
    write = make_writer(file_locks, open_files)

    total_ok = total_fail = 0
    ctr_lock  = threading.Lock()

    def process_sample(wid, rec: dict, pbar):
        nonlocal total_ok, total_fail
        engine = get_engine(wid)
        res = stage2_sample(rec, engine,
                            args.truncation_granularity,
                            verbose=True,
                            worker_id=wid,
                            batch_size=args.batch_size)
        write(res)
        ok = not res.get("error")
        with ctr_lock:
            if ok: total_ok += 1
            else:  total_fail += 1
        if args.verbose:
            tqdm.write(f"  [{'OK' if ok else 'FAIL'}] "
                       f"{str(rec.get('item_id','?'))[:60]} #{rec.get('sample_index','?')}")
        pbar.update(1)

    try:
        with tqdm(total=total, desc="Stage-2", unit="sample") as pbar:
            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                futs = {
                    ex.submit(process_sample, i % args.max_workers, rec, pbar): (iid, sidx)
                    for i, (iid, sidx, rec) in enumerate(pending_samples)
                }
                for fut in as_completed(futs):
                    try:
                        fut.result()
                    except Exception as e:
                        key = futs[fut]
                        tqdm.write(f"  [ERROR] {key[0][:60]} #{key[1]}: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        for fh in open_files.values():
            fh.close()

    print(f"\nDone!  Success: {total_ok}  Failed: {total_fail}")
    print(f"Output dir: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage decoded-action collection (misalignment scenarios).")

    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=1,
        help="1 = complete inference; 2 = truncated decoding",
    )
    parser.add_argument("--scenario", required=True)

    # Stage 1 input
    parser.add_argument("--prompts-dir", default=None, help="Prompts directory")

    # Stage output
    parser.add_argument("--output-dir", default=None, required=True)

    # Stage 2 input (stage-1 results)
    parser.add_argument("--stage1-input", default=None, help="Stage-1 input path for stage 2. Can be a JSONL file or a directory of JSONL files.")

    # Model
    parser.add_argument("--model", default="qwen3-8b", help="Model name")
    parser.add_argument("--inference-type", default="vllm")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (stage 1 complete inference)")

    # Stage 1 sampling
    parser.add_argument("--num-samples", type=int, default=1, help="Reasoning traces to sample per item (stage 1)")

    # Stage 2 truncation
    parser.add_argument("--truncation-granularity", choices=["token", "paragraph", "sentence"], default="token")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for truncated inference (stage 2)")

    # Misalignment tool set: neutral, signal, or all scenario tools.
    parser.add_argument("--tool-mode", choices=["neutral", "signal", "all"], default="neutral", help="neutral tools only, neutral plus the scenario intent tool, or all scenario tools")

    # Common
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--load-tokenizer", action="store_true", default=False, help="Load tokenizer and save token IDs.")

    args = parser.parse_args()

    get_engine = build_engine_pool(args)

    # Stage 1
    if args.stage == 1:
        if not args.prompts_dir:
            print(f"ERROR: --prompts-dir is required for stage {args.stage}")
            sys.exit(1)
        print(f"\nLoading prompts from {args.prompts_dir}...")
        items = load_misalignment_prompts(args.prompts_dir)
        items = filter_items_for_scenario(items, args.scenario)
        print(f"Filtered to {len(items)} '{args.scenario}' items")

        run_stage1(args, items, get_engine, args.output_dir)

    # Stage 2
    else:
        if not Path(args.stage1_input).exists():
            print(f"ERROR: stage-1 input not found: {args.stage1_input}")
            sys.exit(1)

        print(f"\nLoading stage-1 records from {args.stage1_input}...")
        records = load_stage1_records(args.stage1_input)
        records = filter_records_for_scenario(records, args.scenario)
        print(f"Filtered to {len(records)} '{args.scenario}' stage-1 records")

        # Limit Stage 2 to the selected sample indices for each item.
        records = [r for r in records if r.get("sample_index", 0) < args.num_samples]
        print(f"Limited to first {args.num_samples} sample(s) per item → {len(records)} records")

        # Stage 2 requires tokenizer
        if not args.load_tokenizer:
            print("ERROR: Stage 2 requires tokenizer. Please run with --load-tokenizer")
            sys.exit(1)

        run_stage2(args, records, get_engine, args.output_dir)


if __name__ == "__main__":
    main()
