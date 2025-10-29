# app.py
import streamlit as st
import os
from pathlib import Path
from ingestion.ocr import extract_text
from ingestion.parser import parse_resume
from embeddings.indexer import FAISSIndexer
from retriever.retriever import HybridRetriever
from rag.rag_chain import GeminiRAG
from scoring.scoring import keyword_match_score, semantic_similarity_score, combined_global_score, experience_score, education_score
from ranking.topsis_ranking import topsis_rank
from config import RESUME_DIR, JD_DIR, EMBEDDING_MODEL, TOP_K
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from utils import ensure_dir, timestamp, write_json
import re

st.set_page_config(page_title="AI Recruitment RAG", layout="wide")
st.title("AI Recruitment Assistance System (RAG)")

# Optional: load Gemini API key from Streamlit secrets if provided
try:
    if "gemini_api_key" in st.secrets and st.secrets["gemini_api_key"]:
        os.environ["GEMINI_API_KEY"] = st.secrets["gemini_api_key"]
except Exception:
    # Secrets may not be configured; ignore silently
    pass

# Sidebar: index building and settings
with st.sidebar:
    st.header("Index & Models")
    # Model & API key controls (session-only)
    model_current = os.environ.get("GEMINI_MODEL", "")
    model_input = st.text_input("Gemini model", value=model_current or "gemini-2.0-flash", help="e.g., gemini-pro, gemini-1.5-flash-001")
    if model_input and model_input != model_current:
        os.environ["GEMINI_MODEL"] = model_input
        st.caption(f"Using model: {model_input}")

    api_key_current = os.environ.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input("Gemini API key", value=api_key_current, type="password")
    if api_key_input and api_key_input != api_key_current:
        os.environ["GEMINI_API_KEY"] = api_key_input
        st.caption("API key set for this session")
    if "indexer" not in st.session_state:
        st.session_state.indexer = None
    if st.button("Build / Rebuild Index from data folder"):
        st.info("Building index...")
        indexer = FAISSIndexer()
        # scan resumes and jds
        docs = []
        # resumes
        resume_paths = list(Path(RESUME_DIR).glob("*"))
        for p in resume_paths:
            try:
                text = extract_text(str(p))
                # chunk by naive split for now (split paragraphs)
                paras = [seg.strip() for seg in text.split("\n\n") if seg.strip()]
                for i, seg in enumerate(paras):
                    docs.append({"doc_id": f"{p.name}", "text": seg, "section": "resume_paragraph", "type": "resume"})
            except Exception as e:
                st.error(f"Error reading {p}: {e}")
        # jds
        jd_paths = list(Path(JD_DIR).glob("*"))
        for p in jd_paths:
            try:
                text = extract_text(str(p))
                paras = [seg.strip() for seg in text.split("\n\n") if seg.strip()]
                for i, seg in enumerate(paras):
                    docs.append({"doc_id": f"{p.name}", "text": seg, "section": "jd_paragraph", "type": "jd"})
            except Exception as e:
                st.error(f"Error reading {p}: {e}")
        if len(docs) == 0:
            st.warning("No documents found in data/resumes or data/jds.")
        else:
            indexer.add_documents(docs)
            indexer.save()
            st.session_state.indexer = indexer
            st.success(f"Index built with {len(docs)} chunks.")

    if st.session_state.indexer is None:
        if os.path.exists("data/faiss_index.bin"):
            load_btn = st.button("Load existing index")
            if load_btn:
                idx = FAISSIndexer()
                try:
                    idx.load()
                    st.session_state.indexer = idx
                    st.success("Index loaded from disk.")
                except Exception as e:
                    st.error(f"Failed to load index: {e}")

def resume_to_requirements(parsed: dict) -> dict:
    """Convert parsed resume dict into JD-like requirements array with importances."""
    reqs = []
    idx = 1

    # Contacts
    contacts = parsed.get("contacts", {}) or {}
    emails = contacts.get("emails", []) or []
    phones = contacts.get("phones", []) or []
    if emails:
        reqs.append({"id": str(idx), "type": "contact", "text": f"emails: {', '.join(emails)}", "importance": "low"}); idx += 1
    if phones:
        # phones may be tuples; stringify
        ptxt = ", ".join([str(p) for p in phones])
        reqs.append({"id": str(idx), "type": "contact", "text": f"phones: {ptxt}", "importance": "low"}); idx += 1

    # Education / degrees
    degrees = parsed.get("degrees", []) or []
    for d in degrees:
        imp = "high" if d in ("phd", "master") else ("medium" if d == "bachelor" else "low")
        reqs.append({"id": str(idx), "type": "education", "text": d, "importance": imp}); idx += 1

    # Skills
    for sk in parsed.get("skills", []) or []:
        reqs.append({"id": str(idx), "type": "skill", "text": sk, "importance": "high"}); idx += 1

    # Experiences
    for ex in parsed.get("experiences", []) or []:
        reqs.append({"id": str(idx), "type": "experience", "text": ex, "importance": "high"}); idx += 1

    # Years of experience summary
    y = parsed.get("years_of_experience")
    if isinstance(y, (int, float)) and y > 0:
        reqs.append({"id": str(idx), "type": "experience", "text": f"estimated_years: {y:.1f}", "importance": "medium"}); idx += 1

    # Other sections
    sec = parsed.get("sections", {}) or {}
    for name, text_block in sec.items():
        lname = (name or "").lower()
        if "skill" in lname or "experience" in lname:
            # already represented
            continue
        t = lname if lname in {"summary","projects","certifications","achievements","languages","objective"} else "other"
        imp = "medium" if t in {"summary","projects","certifications","achievements"} else "low"
        if text_block and len(text_block.strip()) > 0:
            reqs.append({"id": str(idx), "type": t, "text": text_block.strip(), "importance": imp}); idx += 1

    return {"requirements": reqs}

# Main UI - Evaluate JD against resumes
st.header("Evaluate resumes for a Job Description")

col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Select Job Description")
    jd_files = [f.name for f in Path(JD_DIR).glob("*")]
    selected_jd = st.selectbox("JD file", options=[""] + jd_files)
    jd_text_area = st.text_area("Or paste JD text (overrides file)", height=200)
    if selected_jd and jd_text_area.strip() == "":
        jd_text = extract_text(str(Path(JD_DIR)/selected_jd))
    else:
        jd_text = jd_text_area

    st.subheader("Settings")
    top_k = st.number_input("Top K candidates to retrieve", min_value=1, max_value=50, value=5)
    compute_btn = st.button("Compute Rankings")

with col2:
    st.subheader("Resumes")
    uploaded = st.file_uploader("Upload additional resumes (pdf/docx/txt)", accept_multiple_files=True)
    if uploaded:
        for up in uploaded:
            save_path = Path("data/resumes") / up.name
            ensure_dir(str(save_path))
            with open(save_path, "wb") as f:
                f.write(up.getbuffer())
        st.success("Uploaded files saved to data/resumes. Rebuild index to include them.")

if st.button("Show the Parsed Resumes"):
    resume_paths = list(Path(RESUME_DIR).glob("*"))
    if not resume_paths:
        st.info("No resumes found in data/resumes.")
    else:
        for p in resume_paths:
            try:
                text = extract_text(str(p))
                parsed = parse_resume(text)
                reqs_obj = resume_to_requirements(parsed)
                with st.expander(f"Parsed resume: {p.name}", expanded=False):
                    st.json(reqs_obj)
            except Exception as e:
                st.error(f"Error parsing {p.name}: {e}")

if compute_btn:
    if jd_text.strip() == "":
        st.warning("Please provide a job description text.")
    elif st.session_state.indexer is None:
        st.warning("Index not built or loaded. Build index first from sidebar.")
    else:
        # naive approach: retrieve top candidates by searching JD text against resume chunks and aggregate candidates
        retriever = HybridRetriever(st.session_state.indexer)
        hits = retriever.retrieve(jd_text, top_k=200)
        # aggregate scores per doc
        candidate_map = {}
        for h in hits:
            doc = h["doc_id"]
            candidate_map.setdefault(doc, []).append(h)
        # compute features per candidate
        # Try to derive JD skills once using RAG; fallback to heuristic
        jd_skills = []
        try:
            rag = GeminiRAG(retriever)
            jd_parsed = rag.extract_requirements_from_jd(jd_text)
            reqs = jd_parsed.get("requirements", []) if isinstance(jd_parsed, dict) else []
            jd_skills = [r.get("text","") for r in reqs if isinstance(r, dict) and r.get("type") == "skill" and r.get("text")]
            # Split JD phrases into atomic items
            expanded = []
            for it in jd_skills:
                tmp = it.replace(" and ", ", ")
                tmp = tmp.replace("/", ", ").replace("•", ", ").replace("|", ", ")
                parts = re.split(r"[,;\n]+", tmp)
                expanded.extend([p.strip() for p in parts if p and p.strip()])
            jd_skills = list(dict.fromkeys(expanded)) or jd_skills
        except Exception:
            pass
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
        rows = []
        for doc_id, chunks in candidate_map.items():
            # assemble resume-level text
            texts = [c["text"] for c in chunks]
            # parse resume and extract skill list (best-effort)
            resume_full_text = " ".join(texts)
            # use parser to get skills
            from ingestion.parser import parse_resume
            parsed = parse_resume(resume_full_text)
            resume_skills = parsed.get("skills", [])
            # fallback JD skills heuristic if LLM parse failed or empty
            if not jd_skills and "skill" in jd_text.lower():
                lines = [l.strip() for l in jd_text.split("\n") if l.strip()]
                for i, line in enumerate(lines):
                    if "skill" in line.lower():
                        jd_skills.extend([tok.strip() for tok in re.split(r"[,\n•;]+", " ".join(lines[i:i+5])) if tok.strip()])
                        break
            keyword_s = keyword_match_score(jd_skills, resume_skills)
            semantic_s = semantic_similarity_score([jd_text], texts, embed_model)
            global_s = combined_global_score(keyword_s, semantic_s)
            # experience from parser estimate (fallback already included)
            candidate_years = float(parsed.get("years_of_experience", 0.0))
            exp_score = experience_score(3.0, candidate_years)  # assume required 3 years as example
            # education from parsed degrees
            edu_score = education_score("bachelor", parsed.get("degrees", []))
            rows.append({
                "candidate": doc_id,
                "keyword_score": keyword_s,
                "semantic_score": semantic_s,
                "global_score": global_s,
                "experience_score": exp_score,
                "education_score": edu_score
            })
        if len(rows) == 0:
            st.info("No candidate chunks matched the JD.")
        else:
            df = pd.DataFrame(rows)
            st.subheader("Candidate scores")
            st.dataframe(df)
            # run TOPSIS on criteria: education_score, experience_score, global_score
            matrix = df[["education_score", "experience_score", "global_score"]].to_numpy()
            weights = np.array([0.2, 0.3, 0.5])
            closeness, ranks = topsis_rank(matrix, weights)
            df["closeness"] = closeness
            df["rank"] = ranks
            df = df.sort_values("rank")
            st.subheader("Ranked candidates (TOPSIS)")
            st.table(df[["candidate","rank","closeness","global_score","experience_score","education_score"]].head(20))
            # export
            out_path = f"data/output/ranking_{timestamp()}.json"
            ensure_dir(out_path)
            write_json(out_path, df.to_dict(orient="records"))
            st.success(f"Ranking exported to {out_path}")
