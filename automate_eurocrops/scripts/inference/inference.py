#!/usr/bin/env python3
"""
Mistral Inference CLI
# code adapted from https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407 function calling with transformers


Modes:
  - basic    : prompt uses only the original name
  - ctx      : prompt uses original name + context string
  - ctx-dpl  : prompt uses original name + context + per-row DeepL translation hint

Inputs:
  - country CSV with original names (default column: original_name)
  - HCAT Excel with list of allowed HCAT names (default sheet first, column: HCAT4_name)
  - Optional: contexts pickle (list[str], aligned to country rows)
  - Optional: DeepL Excel with a column 'dpl_trans' (aligned to country rows)

Output:
  - JSON list of validated objects: {original_name, translated_name, HCAT_name}
"""

import re
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Optional

import torch
import pandas as pd
from pydantic import BaseModel, ValidationError, validator, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter, defaultdict


# ---------- Models / Validation ----------
# hallucination control
class HCATRecord(BaseModel):
    original_name: str
    HCAT_name: str
    translated_name: str

    # will be set at runtime
    _allowed_hcat_names: List[str] = []

    @validator("HCAT_name")
    def validate_hcat(cls, v):
        if cls._allowed_hcat_names and v not in cls._allowed_hcat_names:
            raise ValueError({v}," is not a valid HCAT mapping.")
        return v


# ---------- Prompt builder ----------

BASE_SYSTEM = (
    "You are an assistant that answers ONLY with the required json output to the user's question. "
    "No explanations."
)

PROMPT_TEMPLATE = """[INST]
{system}

{ctx_block}

Given a description below:

{orig_block}

First translate every part of the description to English{dpl_tail}.
This is "translated_name".

Next use the "translated_name" to match based on the closest semantic meaning to an entry from this list of HCAT_names:

{hcat_block}

The "HCAT_name" should be as specific as possible, prioritize species level semantic matches to the HCAT_names from the "translated_name"
Consider the entire "translated_name" and look for the HCAT_name that is the closest match to the entire description. 
ALso map to the more informative HCAT_name.
E.g for root chicory, chicory is more informative than root so map to chicory_chicories
Summer crops should be the spring equivalent in "HCAT_name", but keep the translated_name as the original English translation
If the exact "translated_name" is not in the HCAT_names, find the upper class of crops that includes the crop in HCAT_names and map to its other class.
For example for prickly pear, the upper crop class is fruit, "HCAT_name" is other_orchards_fruits.
If the "translated_name" is not a crop or argriculture product, for example rocks or landscape features, "HCAT_name" is not_known_and_other
If the "translated_name" is a mix of crops, find the upper class of crops that includes all the crops in HCAT_names and map to the upper class
For example if "translated_name" is mixed alfafa and clover, the common upper crop class is legumes, "HCAT_name" is legumes.
Or if "translated_name" is mixed cultures, the common upper crop class is arable crops, HCAT_name" is arable_crops
Trees and flowers are also agriculture products in HCAT.

Return only in json format defined below:
'''
"original_name": "{orig_value}", "translated_name": "english translation of {orig_value}", "HCAT_name": "closest semantic match of translated name in HCAT_names"
'''
Only use a "HCAT_name" from the provided HCAT_names.
No other detail.

**Example Json Output:**
"original_name": "Erdbeeren", "translated_name": "strawberries", "HCAT_name": "strawberries"
[/INST]
"""

def build_prompt(
    mode: str,
    original_value: str,
    hcat_list: List[str],
    context_str: Optional[str],
    dpl_trans: Optional[str],
) -> str:
    if mode not in {"basic", "ctx", "ctx-dpl"}:
        raise ValueError("mode must be one of: basic, ctx, ctx-dpl")

    # if with RAG provided context
    ctx_block = ""
    if mode in {"ctx", "ctx-dpl"} and context_str:
        ctx_block = f"Based on the following agricultural information:\n{context_str}\n"
    # if with Deepl provided translation in addition to RAG provided context
    dpl_tail = ""
    if mode == "ctx-dpl" and dpl_trans is not None:
        dpl_tail = f', while taking into consideration that deepl\'s translation is "{dpl_trans}"'
    else:
        dpl_tail = ""

    prompt = PROMPT_TEMPLATE.format(
        system=BASE_SYSTEM,
        ctx_block=ctx_block.strip(),
        orig_block=original_value,
        dpl_tail=dpl_tail,
        hcat_block=hcat_list,
        orig_value=original_value,
    )
    return prompt


# ---------- Inference ----------

def generate_hcat(
    tokenizer,
    model,
    prompt: str,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
    max_new_tokens: int = 120,
    temperature: float = 0.35,
    top_p: float = 0.9,
):
    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id

        )
    # decode only the newly generated portion
    gen = tokenizer.decode(out[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return gen


def extract_json(raw_text: str, original_value: str) -> dict:
    """
    Sometimes llm outputs additional info in addition to the required stuff,
    this tries to extract it
    Try direct JSON parse; if it fails, attempt to extract a JSON-ish object.
    Default output here to HCAT_name: not_known_and_other
    """
    def _try_json(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    # direct
    obj = _try_json(raw_text)
    if obj is None:
        # embedded {...}
        m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if m:
            obj = _try_json(m.group(0))

    # loose key:value pairs
    if obj is None:
        kv = {}
        for key in ("original_name", "translated_name", "HCAT_name"):
            p = re.search(rf'"?{key}"?\s*:\s*"([^"]*)"', raw_text)
            if p:
                kv[key] = p.group(1)
        obj = kv or {}

    
    original = obj.get("original_name") or original_value
    translated = obj.get("translated_name")
    hcat = obj.get("HCAT_name")

    if not translated or not str(translated).strip():
        translated = ""

    if hcat is None or str(hcat).strip() == "":
        hcat = "not_known_and_other"

    return {
        "original_name": original,
        "translated_name": translated,
        "HCAT_name": hcat,
    }


def majority_generation(all_runs_outputs):
    """
    all_runs_outputs: List[List[dict]] with shape [runs][rows]
    Returns: List[dict] consensus list length == rows
    Rule:
      - Majority vote on HCAT_name (non-empty). Ties broken by earliest-run winner.
      - translated_name picked as the most common among rows where HCAT_name == winner;
        tie-break by first non-empty seen in earliest run.
      - original_name taken from any run (they all share the same original).
    """


    num_runs = len(all_runs_outputs)
    num_rows = len(all_runs_outputs[0])


    merged = []
    for row_idx in range(num_rows):
        # gather candidates for this row across runs
        rows = [all_runs_outputs[run_idx][row_idx] for run_idx in range(num_runs)]

        # vote on HCAT_name
        labels = [r.get("HCAT_name") for r in rows]
        label_counts = Counter(labels)
        max_count = max(label_counts.values())
        # candidates with max votes
        winners = [lab for lab, c in label_counts.items() if c == max_count]

        if len(winners) == 1:
            hcat_winner = winners[0]
        else:
            # tie-break by earliest appearance across runs
            for lab in labels:
                if lab in winners:
                    hcat_winner = lab
                    break

        # choose translated_name conditioned on the winning HCAT_name
        tn_candidates = [r.get("translated_name", "") for r in rows if r.get("HCAT_name") == hcat_winner]
        tn_counts = Counter([t for t in tn_candidates if t])
        if tn_counts:
            tn_max = max(tn_counts.values())
            tn_winners = [t for t, c in tn_counts.items() if c == tn_max]
            if len(tn_winners) == 1:
                translated_winner = tn_winners[0]
            else:
                # tie-break by earliest non-empty with winning label
                translated_winner = ""
                for r in rows:
                    if r.get("HCAT_name") == hcat_winner:
                        t = r.get("translated_name", "")
                        if t:
                            translated_winner = t
                            break
        else:
            # fallback: first non-empty translated_name in earliest run
            translated_winner = ""
            for r in rows:
                t = r.get("translated_name", "")
                if t:
                    translated_winner = t
                    break

        original_name = rows[0].get("original_name", "")  # same across runs
        merged.append({
            "original_name": original_name,
            "translated_name": translated_winner,
            "HCAT_name": hcat_winner,
        })
    return merged


# ---------- Main CLI ----------

def main():
    ap = argparse.ArgumentParser(description="HCAT mapper CLI with selectable prompting modes.")
    # Required model
    ap.add_argument("--model-id", default="mistralai/Mistral-Nemo-Instruct-2407",
                    help="HF model id for generation.")
    # Inputs
    ap.add_argument("--country-csv", required=True, type=Path, help="Path to country CSV with original names.")
    ap.add_argument("--country-col", default="original_name", help="Column in country CSV with original names.")
    ap.add_argument("--hcat-xlsx", required=True, type=Path, help="Path to HCAT Excel.")
    ap.add_argument("--hcat-col", default="HCAT4_name", help="Column in HCAT Excel that lists valid HCAT names.")
    ap.add_argument("--contexts-pkl", type=Path, default=None, help="Optional: pickle with list[str] contexts.")
    ap.add_argument("--deepl-xlsx", type=Path, default=None, help="Optional: Excel with 'dpl_trans' column.")
    ap.add_argument("--deepl-col", default="dpl_trans", help="Column name for DeepL translation text.")
    
    # Mode
    ap.add_argument("--mode", choices=["basic", "ctx", "ctx-dpl"], required=True,
                    help="Select prompting mode.")
    ap.add_argument("--majority-result", action="store_true",
                help="After N runs, compute a majority-vote consensus per row.")
    ap.add_argument("--majority-out-path", type=Path, default=None,
                    help="Path to save consensus JSON (defaults to <out-json-path>_consensus.json).")

    # Output
    ap.add_argument("--out-json-path", required=True, type=Path, help="Where to write results JSON.")
    ap.add_argument("--runs", type=int, default=1, help="Repeat full pass N times with different sampling.")
    ap.add_argument("--max-retries", type=int, default=1, help="Repeat generation N times when HCAT failed validation.")    
    # Gen params
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--top-p", type=float, default=0.9)
    # Device 
    ap.add_argument("--device-map", default="auto", help='Transformers device_map, default "auto".')


    args = ap.parse_args()




    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    eos_id = tokenizer.eos_token_id

    # load data
    country_df = pd.read_csv(args.country_csv)
    original_names = country_df[args.country_col].astype(str).tolist()

    hcat_df = pd.read_excel(args.hcat_xlsx)
    hcat_list = hcat_df[args.hcat_col].astype(str).tolist()
    HCATRecord._allowed_hcat_names = hcat_list  # bind validator list

    contexts = None
    if args.mode in {"ctx", "ctx-dpl"}:
        with open(args.contexts_pkl, "rb") as f:
            contexts = pickle.load(f)
        if len(contexts) != len(original_names):
            raise ValueError("Length mismatch: contexts list must align with country rows.")

    dpl_list = None
    if args.mode == "ctx-dpl":
        dpl_df = pd.read_excel(args.deepl_xlsx)
        dpl_list = dpl_df[args.deepl_col].astype(str).tolist()
        if len(dpl_list) != len(original_names):
            raise ValueError("Length mismatch: DeepL list must align with country rows.")

    # runs loop 
    all_runs_outputs = []
    for run_idx in range(1, args.runs + 1):
        run_outputs = []
        
        for i, orig in enumerate(original_names):    
            for attempt in range(args.max_retries):    
                ctx = contexts[i] if contexts is not None else None
                dpl = dpl_list[i] if dpl_list is not None else None

                prompt = build_prompt(
                    mode=args.mode,
                    original_value=orig,
                    hcat_list=hcat_list,
                    context_str=ctx,
                    dpl_trans=dpl,
                )

                # generation
                raw =generate_hcat(
                    tokenizer,
                    model,
                    prompt,
                    eos_token_id=eos_id,
                    pad_token_id=eos_id,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                # parse / validate
                parsed = extract_json(raw, orig)

                try:
                    validated = HCATRecord(**parsed).model_dump()
                    print("[Validated]", parsed)
                    break
                except ValidationError as e:
                    print("[ERROR]", e.errors())
                    print("[RAW]", parsed)  
                    validated = HCATRecord(
                        original_name=orig,
                        translated_name="",
                        HCAT_name="not_known_and_other",
                    ).model_dump()
  

            run_outputs.append(validated)

                
                

        all_runs_outputs.append(run_outputs)

    # If multiple runs, save as an object with per-run arrays. If single run, save the list directly.
    out_path = args.out_json_path

    if args.runs == 1:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_runs_outputs[0], f, ensure_ascii=False, indent=2)
        print("Saved results to ", out_path)
    else:
        outputs = {f"run_{i+1}": out for i, out in enumerate(all_runs_outputs)}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        print("Saved results to ", out_path)

        if args.majority_result:
            consensus = majority_generation(all_runs_outputs)
            consensus_path = args.majority_out_path or out_path.with_name(out_path.stem + "_consensus.json")
            with open(consensus_path, "w", encoding="utf-8") as f:
                json.dump(consensus, f, ensure_ascii=False, indent=2)
            print(f"Saved majority generation to ",consensus_path )

    

if __name__ == "__main__":
    main()
