"""
Fine-tune BioBERT for clinical NER on MACCROBAT2018 (Stage 5, approach B).

    dmis-lab/biobert-base-cased-v1.2   (base BioBERT: encoder only, NO NER ability)
            +
    new token-classification layer    (Linear 768 -> 15 labels, randomly initialised)
            ↓  trained on MACCROBAT2018 clinical case reports
    models/biobert-maccrobat-ner/      (fine-tuned BioBERT that predicts our labels)

Data: data/maccrobat/data.jsonl  (ktgiahieu/maccrobat2018_2020 on Hugging Face, CC-BY-4.0,
      from Caufield 2019, MACCROBAT, figshare doi:10.6084/m9.figshare.9764942)
      The file contains every document twice -> duplicates are removed before splitting,
      otherwise test documents would also be in training (data leakage).

Usage:
    python train_biobert_ner.py                  # train with defaults (~15-30 min on Apple M4)
    python train_biobert_ner.py --epochs 1       # quick run
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

BASE_MODEL = "dmis-lab/biobert-base-cased-v1.2"
# MACCROBAT entity types we keep. All other types (Lab_value, Age, ...) become "O".
KEEP = ["Sign_symptom", "Disease_disorder", "Medication", "Duration", "Severity", "Dosage", "Frequency"]
LABELS = ["O"] + [f"{p}-{t}" for t in KEEP for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
IGNORE = -100          # PyTorch's cross-entropy skips positions with this label
MAX_WORDS = 120        # long sentences are cut into chunks of at most this many words


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- data
def load_documents(path: Path) -> list[dict]:
    docs = [json.loads(line) for line in path.open()]
    unique, seen = [], set()
    for d in docs:
        key = " ".join(d["tokens"])
        if key not in seen:                      # remove exact duplicate documents
            seen.add(key)
            unique.append(d)
    # The dataset keeps line breaks and invisible Unicode spaces ('\n', ' ', ...) as separate
    # "words". The tokenizer turns them into NOTHING, so they are formatting leftovers, not words.
    removed = 0
    for d in unique:
        keep = [i for i, w in enumerate(d["tokens"]) if w.strip()]
        removed += len(d["tokens"]) - len(keep)
        d["tokens"] = [d["tokens"][i] for i in keep]
        d["tags"] = [d["tags"][i] for i in keep]
    print(f"Documents in file: {len(docs)}  ->  unique: {len(unique)}  |  whitespace-only words removed: {removed}")
    return unique


def filter_tags(tags: list[str]) -> list[str]:
    """Keep only the entity types in KEEP; everything else becomes O."""
    return [t if t != "O" and t[2:] in KEEP else "O" for t in tags]


def split_sentences(doc: dict) -> list[tuple[list[str], list[str]]]:
    """Cut a whole case report into sentences (at '.'), then into chunks of <= MAX_WORDS words."""
    tokens, tags = doc["tokens"], filter_tags(doc["tags"])
    sentences, start = [], 0
    for i, tok in enumerate(tokens):
        if tok == "." or i == len(tokens) - 1:
            sentences.append((tokens[start:i + 1], tags[start:i + 1]))
            start = i + 1
    chunks = []
    for toks, tgs in sentences:
        for s in range(0, len(toks), MAX_WORDS):
            t, g = toks[s:s + MAX_WORDS], tgs[s:s + MAX_WORDS]
            if g and g[0].startswith("I-"):     # a chunk can't start inside an entity
                g = ["B-" + g[0][2:]] + g[1:]
            chunks.append((t, g))
    return chunks


def encode(tokenizer, sentences):
    """Words -> WordPiece ids. Only the FIRST sub-token of each word gets the word's label;
    later sub-tokens and [CLS]/[SEP] get IGNORE so they don't count in the loss."""
    examples = []
    for words, tags in sentences:
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=512)
        labels, previous = [], None
        for wid in enc.word_ids():
            if wid is None or wid == previous:
                labels.append(IGNORE)
            else:
                labels.append(LABEL2ID[tags[wid]])
            previous = wid
        examples.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                         "labels": labels, "words": words, "tags": tags})
    return examples


def collate(batch, pad_id):
    """Pad every example in the batch to the same length."""
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda seq, value: seq + [value] * (n - len(seq))
    return {
        "input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
        "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
        "labels": torch.tensor([pad(b["labels"], IGNORE) for b in batch]),
    }


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, examples, tokenizer, device, batch_size):
    """Predict labels for every word and compare with the gold labels at ENTITY level (seqeval)."""
    model.eval()
    gold, pred = [], []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        batch = {k: v.to(device) for k, v in collate(chunk, tokenizer.pad_token_id).items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        ids = logits.argmax(-1).cpu()
        for j, ex in enumerate(chunk):
            keep = [k for k, lab in enumerate(ex["labels"]) if lab != IGNORE]   # first sub-token of each word
            # Gold and predicted labels are read at the SAME token positions, so they can never drift apart.
            pred.append([LABELS[int(ids[j][k])] for k in keep])
            gold.append([LABELS[ex["labels"][k]] for k in keep])
    return gold, pred


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BioBERT on MACCROBAT2018.")
    parser.add_argument("--data", default="data/maccrobat/data.jsonl")
    parser.add_argument("--output", default="models/biobert-maccrobat-ner")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    set_seed(args.seed)

    # 1. Data: dedupe, split BY DOCUMENT (sentences of one report never end up in two splits).
    docs = load_documents(Path(args.data))
    random.Random(args.seed).shuffle(docs)
    n = len(docs)
    splits = {"train": docs[:int(0.8 * n)], "validation": docs[int(0.8 * n):int(0.9 * n)], "test": docs[int(0.9 * n):]}
    # BioBERT v1.2 is a CASED model, but its tokenizer config lower-cases by default ("MRI" -> "m ##ri").
    # do_lower_case=False keeps capitalisation, which matters for abbreviations like MRI, COVID, IgA.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, do_lower_case=False)
    data = {}
    for name, ds in splits.items():
        sentences = [s for d in ds for s in split_sentences(d)]
        data[name] = encode(tokenizer, sentences)
        n_entities = sum(t.startswith("B-") for _, tags in sentences for t in tags)
        print(f"{name:<10} {len(ds):>3} documents  {len(sentences):>5} sentences  {n_entities:>5} entities")

    # 2. Model: base BioBERT + a NEW classification layer with our 15 labels.
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
    ).to(args.device)
    print(f"\nBase model: {BASE_MODEL}  |  labels: {len(LABELS)}  |  device: {args.device}")

    # 3. Optimiser: AdamW with weight decay, learning rate warms up for 10% of steps then decays linearly.
    loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    # 4. Training loop.
    best_f1, history, out_dir = -1.0, [], Path(args.output)
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = model(**batch).loss            # cross-entropy over first sub-tokens only
            loss.backward()                        # gradients for classifier AND all BioBERT layers
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0 or step == len(loader):
                print(f"  epoch {epoch}  step {step:>4}/{len(loader)}  loss {running / step:.4f}", flush=True)

        gold, pred = evaluate(model, data["validation"], tokenizer, args.device, args.batch_size)
        val_f1 = f1_score(gold, pred)
        history.append({"epoch": epoch, "train_loss": round(running / len(loader), 4),
                        "val_f1": round(val_f1, 4), "minutes": round((time.time() - start) / 60, 1)})
        print(f"epoch {epoch}: train loss {running / len(loader):.4f}  |  validation F1 {val_f1:.4f}"
              f"  |  {history[-1]['minutes']} min")
        if val_f1 > best_f1:                       # keep the best epoch (early-stopping style)
            best_f1 = val_f1
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"  -> best so far, saved to {out_dir}")

    # 5. Final, honest score: reload the best checkpoint and evaluate ONCE on the unseen test documents.
    model = AutoModelForTokenClassification.from_pretrained(out_dir).to(args.device)
    gold, pred = evaluate(model, data["test"], tokenizer, args.device, args.batch_size)
    report = classification_report(gold, pred, digits=3)
    print("\nTEST SET (20 unseen case reports):\n")
    print(report)

    metrics = {
        "base_model": BASE_MODEL, "dataset": "MACCROBAT2018 (ktgiahieu/maccrobat2018_2020, deduplicated)",
        "labels": LABELS, "split_documents": {k: len(v) for k, v in splits.items()},
        "hyperparameters": {k: getattr(args, k) for k in ("epochs", "batch_size", "lr", "seed")},
        "history": history, "best_val_f1": round(best_f1, 4),
        "test_report": classification_report(gold, pred, output_dict=True),
    }
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"Saved model and training_metrics.json to {out_dir}")


if __name__ == "__main__":
    main()
