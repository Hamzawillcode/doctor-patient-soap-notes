"""
Step 5 of the pipeline: Clinical information extraction with BioBERT (Named Entity Recognition).

    outputs/<name>_roles.json  (DOCTOR / PATIENT utterances)
            ↓
    BioBERT tokenizer   -> WordPiece tokens + character offsets
            ↓
    BioBERT encoder     -> one 768-number contextual vector per token (12 Transformer layers)
            ↓
    token-classification layer (Linear 768 -> 3)  -> B- / I- / O label per token
            ↓
    BIO decoding        -> entities with character offsets and confidence
            ↓
    outputs/<name>_clinical_entities.json

Models (both fine-tuned BioBERT, BertForTokenClassification):
    alvaroalon2/biobert_diseases_ner   labels: B-DISEASE, I-DISEASE, O   (BC5CDR-disease + NCBI-disease)
    alvaroalon2/biobert_chemical_ner   labels: B-CHEMICAL, I-CHEMICAL, O (BC5CDR-chemical + BC4CHEMD)

Labels are reported EXACTLY as the models predict them (DISEASE, CHEMICAL).
    - Symptoms (fever, headache, nausea) appear under DISEASE, because BC5CDR annotates signs and
      symptoms as diseases. The model cannot tell a symptom from a disease.
    - Drugs appear under CHEMICAL, which also covers non-drug chemicals.
    - DURATION and SEVERITY are NOT supported by these models and are NOT extracted.

Usage:
    python clinical_extraction.py audio/sample_consultation_2.wav
    python clinical_extraction.py --transcript examples/example_transcript.txt
    python clinical_extraction.py audio/sample_consultation_2.wav --raw     # show token-level model output
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODELS = {
    "DISEASE": "alvaroalon2/biobert_diseases_ner",
    "CHEMICAL": "alvaroalon2/biobert_chemical_ner",
}
UNSUPPORTED = {
    "DURATION": "No BioBERT NER checkpoint is trained with a duration label.",
    "SEVERITY": "No BioBERT NER checkpoint is trained with a severity label.",
}
OUTSIDE_LABELS = {"O", "0"}   # the disease model spells "outside" as the digit '0'


class BioBertNER:
    """One fine-tuned BioBERT token-classification model, run step by step."""

    def __init__(self, model_name: str):
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).eval()
        self.id2label = self.model.config.id2label

    def predict_tokens(self, text: str) -> list[dict]:
        """Run tokenizer -> encoder -> classifier. Returns one prediction per WordPiece token."""
        # 1. Tokenizer: text -> WordPiece token ids. offset_mapping gives each token's
        #    character span in the original text; word_ids() says which word it belongs to.
        enc = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        word_ids = enc.word_ids()

        with torch.no_grad():
            # 2. Encoder: 12 Transformer layers -> contextual vector per token, shape (1, tokens, 768).
            hidden = self.model.bert(**enc).last_hidden_state
            # 3. Token-classification layer: Linear(768 -> number of labels) -> scores per token.
            logits = self.model.classifier(self.model.dropout(hidden))
            # 4. Softmax turns scores into probabilities that sum to 1 for each token.
            probs = logits.softmax(dim=-1)[0]

        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        out = []
        for i, token in enumerate(tokens):
            best = int(probs[i].argmax())
            out.append({
                "token": token,
                "word_id": word_ids[i],          # None for special tokens [CLS] / [SEP]
                "start": offsets[i][0],
                "end": offsets[i][1],
                "label": self.id2label[best],
                "probability": round(float(probs[i][best]), 4),
                "hidden_size": hidden.shape[-1],
            })
        return out

    def extract(self, text: str) -> tuple[list[dict], list[dict]]:
        """Token predictions -> entities. Returns (entities, raw token predictions)."""
        tokens = self.predict_tokens(text)

        # Word-level labels: a word takes the label of its FIRST sub-token. This is the
        # standard convention used when BERT NER models are trained.
        words = []
        for t in tokens:
            if t["word_id"] is None:
                continue
            if words and words[-1]["word_id"] == t["word_id"]:
                words[-1]["end"] = t["end"]               # extend the word over later sub-tokens
            else:
                words.append(dict(t))

        # BIO decoding: B- starts an entity, I- continues it, O ends it.
        # An I- without a preceding entity of the same type is treated as a start (lenient).
        entities, current = [], None
        for w in words:
            label = w["label"]
            if label in OUTSIDE_LABELS:
                current = None
                continue
            prefix, etype = label.split("-", 1)
            if prefix == "I" and current and current["label"] == etype:
                current["end"] = w["end"]
                current["_probs"].append(w["probability"])
            else:
                current = {"label": etype, "start": w["start"], "end": w["end"], "_probs": [w["probability"]]}
                entities.append(current)

        for e in entities:
            e["text"] = text[e["start"]:e["end"]]
            # confidence = average probability of the predicted label over the entity's words
            e["confidence"] = round(sum(e["_probs"]) / len(e["_probs"]), 4)
            e["model"] = self.name
            del e["_probs"]
        return [{k: e[k] for k in ("text", "label", "start", "end", "confidence", "model")} for e in entities], tokens


def load_conversation(args) -> tuple[list[dict], str, str]:
    """Return (utterances, source path, output stem). Each utterance has role, speaker, text."""
    if args.transcript:
        path = Path(args.transcript)
        utterances = []
        for line in path.read_text().splitlines():
            if ":" in line:
                role, text = line.split(":", 1)
                utterances.append({"role": role.strip().upper(), "speaker": None, "text": text.strip(),
                                   "start": None, "end": None})
        return utterances, str(path), path.stem

    stem = Path(args.audio).stem
    path = Path(args.roles or f"outputs/{stem}_roles.json")
    if not path.exists():
        raise SystemExit(f"Missing {path} - run `python identify_roles.py <audio>` first.")
    data = json.loads(path.read_text())
    if data.get("needs_review"):
        print("WARNING: roles have LOW confidence in the input - check them before trusting speaker context.\n")
    return data["utterances"], str(path), stem


def print_raw(model_name: str, text: str, tokens: list[dict]) -> None:
    print(f"   [{model_name}] {len(tokens)} tokens -> vectors of size {tokens[0]['hidden_size']}")
    for t in tokens:
        mark = "" if t["label"] in OUTSIDE_LABELS else "   <==="
        print(f"      {t['token']:<14} {t['label']:<11} p={t['probability']:.2f}{mark}")


def main():
    parser = argparse.ArgumentParser(description="BioBERT clinical entity extraction.")
    parser.add_argument("audio", nargs="?", help="Audio file (used to find outputs/<name>_roles.json)")
    parser.add_argument("--roles", help="Roles JSON from identify_roles.py")
    parser.add_argument("--transcript", help="Plain text file with lines like 'PATIENT: text'")
    parser.add_argument("--raw", action="store_true", help="Print token-level model output")
    parser.add_argument("--output", help="Where to save JSON (default: outputs/<name>_clinical_entities.json)")
    args = parser.parse_args()
    if not (args.audio or args.roles or args.transcript):
        parser.error("give an audio file, --roles or --transcript")
    if args.roles and not args.audio:
        args.audio = args.roles.replace("_roles.json", "")

    utterances, source, stem = load_conversation(args)

    print("Loading BioBERT NER models ...")
    models = [BioBertNER(name) for name in MODELS.values()]

    conversation = []
    print("\n" + "=" * 60)
    print("CLINICAL INFORMATION EXTRACTION\n")
    for u in utterances:
        entities = []
        role = u.get("role") or u.get("speaker")
        print(f"{role}:")
        print(f"Text: {u['text']}")
        for m in models:
            found, tokens = m.extract(u["text"])
            entities.extend(found)
            if args.raw:
                print_raw(m.name, u["text"], tokens)
        entities.sort(key=lambda e: e["start"])
        print("Extracted Entities:" if entities else "Extracted Entities: none")
        for e in entities:
            print(f"- {e['text']} → {e['label']}  (confidence {e['confidence']:.2f}, chars {e['start']}-{e['end']})")
        print()
        conversation.append({"speaker": role, "speaker_label": u.get("speaker"),
                             "start": u.get("start"), "end": u.get("end"),
                             "text": u["text"], "entities": entities})

    # Summary: every entity, grouped by label, with who said it.
    summary = {}
    for turn in conversation:
        for e in turn["entities"]:
            summary.setdefault(e["label"], []).append({"text": e["text"], "speaker": turn["speaker"]})

    print("=" * 60)
    print("SUMMARY OF EXTRACTED ENTITIES\n")
    for label, items in summary.items():
        print(f"{label} ({len(items)}):")
        for it in items:
            print(f"   {it['text']:<20} said by {it['speaker']}")
    print("\nNOT EXTRACTED (unsupported by the BioBERT models):")
    for label, why in UNSUPPORTED.items():
        print(f"   {label}: {why}")

    result = {
        "source": source,
        "models": MODELS,
        "label_notes": {
            "DISEASE": "Diseases AND signs/symptoms (BC5CDR annotation scheme); symptom vs disease is not distinguished.",
            "CHEMICAL": "Chemicals including drugs; not limited to medications.",
        },
        "unsupported_labels": UNSUPPORTED,
        "conversation": conversation,
        "summary": summary,
    }
    output_path = Path(args.output) if args.output else Path("outputs") / f"{stem}_clinical_entities.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved JSON to: {output_path}")


if __name__ == "__main__":
    main()
