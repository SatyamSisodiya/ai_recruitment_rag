# api_server.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ingestion.ocr import extract_text
from ingestion.parser import parse_resume
from embeddings.indexer import FAISSIndexer
from retriever.retriever import HybridRetriever
from rag.rag_chain import GeminiRAG
from scoring.scoring import *
from ranking.topsis_ranking import topsis_rank
from config import RESUME_DIR, JD_DIR, EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer
import shutil, os, json
import pandas as pd
import numpy as np

app = FastAPI(title="AI Recruitment Assistant (RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

indexer = FAISSIndexer()
if os.path.exists("data/faiss_index.bin"):
    indexer.load()
retriever = HybridRetriever(indexer)
rag = GeminiRAG(retriever)
embed_model = SentenceTransformer(EMBEDDING_MODEL)

@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    os.makedirs(RESUME_DIR, exist_ok=True)
    file_path = os.path.join(RESUME_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = extract_text(file_path)
    parsed = parse_resume(text)
    return {"filename": file.filename, "parsed": parsed}

@app.post("/parse_jd/")
async def parse_jd(jd_text: str = Form(...)):
    parsed = rag.extract_requirements_from_jd(jd_text)
    return {"requirements": parsed}

@app.post("/rank_resumes/")
async def rank_resumes(jd_text: str = Form(...)):
    hits = retriever.retrieve(jd_text, top_k=50)
    candidates = {}
    for h in hits:
        candidates.setdefault(h["doc_id"], []).append(h)
    rows = []
    for doc_id, chs in candidates.items():
        resume_text = " ".join([c["text"] for c in chs])
        parsed = parse_resume(resume_text)
        jd_skills = [r["text"] for r in rag.extract_requirements_from_jd(jd_text).get("requirements", []) if r["type"]=="skill"]
        resume_skills = parsed.get("skills", [])
        k_s = keyword_match_score(jd_skills, resume_skills)
        s_s = semantic_similarity_score([jd_text], [resume_text], embed_model)
        g_s = combined_global_score(k_s, s_s)
        exp_s = experience_score(3.0, 4.0)
        edu_s = education_score("bachelor", parsed.get("education", []))
        rows.append([doc_id, k_s, s_s, g_s, exp_s, edu_s])
    if not rows:
        return {"message": "No candidates found."}
    df = pd.DataFrame(rows, columns=["candidate","keyword","semantic","global","experience","education"])
    mat = df[["education","experience","global"]].to_numpy()
    weights = np.array([0.2,0.3,0.5])
    closeness, ranks = topsis_rank(mat, weights)
    df["closeness"], df["rank"] = closeness, ranks
    df = df.sort_values("rank")
    return json.loads(df.to_json(orient="records"))
