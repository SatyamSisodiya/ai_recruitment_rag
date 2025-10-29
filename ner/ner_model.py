# ner/ner_model.py
from transformers import pipeline
from typing import Dict, List
import re

class ResumeNER:
    """
    Simple wrapper over Hugging Face NER pipeline.
    For better results replace model_name with a resume-specific model.
    """
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        # you can swap to a resume-specific HF model or fine-tune your own
        self.nlp = pipeline("ner", model=model_name, grouped_entities=True)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        ner_out = self.nlp(text)
        results = {}
        # grouped_entities returns dicts with entity_group
        for ent in ner_out:
            label = ent.get("entity_group", ent.get("entity"))
            val = ent.get("word") or ent.get("entity")
            results.setdefault(label, set()).add(val.strip())
        # convert sets to lists and do minor normalization
        return {k: list(v) for k, v in results.items()}
