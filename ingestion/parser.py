# ingestion/parser.py
from typing import Dict, List
import re
from utils import extract_contacts, normalize_text
from ner.ner_model import ResumeNER
import os

ner_model = ResumeNER()

SECTION_HEADERS = [
    "education", "experience", "work experience", "skills", "projects", "summary", "objective",
    "certifications", "achievements", "publications", "languages"
]

# Minimal domain skills lexicon (extend as needed)
COMMON_SKILLS = [
    "python","java","c++","c#","javascript","typescript","sql","html","css","react","node",
    "django","flask","fastapi","pytorch","tensorflow","keras","scikit-learn","sklearn","pandas",
    "numpy","matplotlib","seaborn","nlp","cv","computer vision","machine learning","deep learning",
    "data science","docker","kubernetes","linux","git","aws","azure","gcp","tableau","power bi",
]

def split_by_sections(text: str) -> Dict[str, str]:
    """
    Try to split resume text into sections by heuristics (headers)
    """
    text = text.replace("\r", "\n")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sections = {}
    current_header = "header"
    buffer = []
    for line in lines:
        low = line.lower().strip(": ")
        if any(low.startswith(h) for h in SECTION_HEADERS) or re.match(r"^[A-Z ]{3,}$", line):
            # treat as header
            if buffer:
                sections[current_header] = "\n".join(buffer).strip()
            current_header = low
            buffer = []
        else:
            buffer.append(line)
    if buffer:
        sections[current_header] = "\n".join(buffer).strip()
    return sections

def parse_resume(text: str) -> Dict:
    """
    Combine NER + heuristics + simple rules to produce structured resume JSON.
    """
    parsed = {}
    parsed['raw_text'] = text
    parsed['contacts'] = extract_contacts(text)
    sections = split_by_sections(text)
    parsed['sections'] = sections

    # run NER to extract entities
    entities = ner_model.extract_entities(text)
    parsed['entities'] = entities

    # extract skills heuristically if skills section exists
    skills = []
    for key in sections:
        if "skill" in key:
            # split comma or bullets
            s = sections[key]
            parts = re.split(r"[,\n•\-;]+", s)
            parts = [p.strip() for p in parts if p.strip()]
            skills.extend(parts)
    # fallback: scan full text for common skills if none found
    if not skills:
        low = text.lower()
        for sk in COMMON_SKILLS:
            if sk in low:
                skills.append(sk)
    parsed['skills'] = list(dict.fromkeys([s for s in skills if len(s) > 1]))  # dedupe

    # unify experiences: try to find experience paragraphs
    experiences = []
    for key in sections:
        if "experience" in key or "work" in key:
            ex_text = sections[key]
            # split by blank lines or bullets
            items = re.split(r"\n{1,2}", ex_text)
            experiences.extend([i.strip() for i in items if len(i.strip()) > 20])
    parsed['experiences'] = experiences

    # estimate years of experience from date ranges in full text as fallback
    years = 0.0
    for m in re.finditer(r"(19\d{2}|20\d{2})\s*[-–to]{1,3}\s*(19\d{2}|20\d{2}|present|Present)", text):
        try:
            start = int(m.group(1))
            endg = m.group(2)
            end = int(endg) if endg.isdigit() else 2025
            if end >= start:
                years += (end - start)
        except Exception:
            continue
    parsed['years_of_experience'] = years

    # extract degrees from text (simple heuristic)
    degree_patterns = [
        (r"\b(phd|doctorate|doctor of philosophy)\b", "phd"),
        (r"\b(master(?:'s)?|msc|m\.sc|ms|mtech|m\.tech|mca|mba|m\.ba)\b", "master"),
        (r"\b(bachelor(?:'s)?|bsc|b\.sc|bs|b\.tech|btech|be|b\.e|ba|b\.a)\b", "bachelor"),
        (r"\b(high\s*school|hsc|ssc|12th|10th)\b", "highschool"),
    ]
    degrees = []
    lowt = text.lower()
    for pat, norm in degree_patterns:
        if re.search(pat, lowt):
            degrees.append(norm)
    parsed['degrees'] = list(dict.fromkeys(degrees))
    return parsed
