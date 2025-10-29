# rag/rag_chain.py
import json
import re
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever.retriever import HybridRetriever
from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE

if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

class GeminiRAG:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        # Prefer explicit API key; fall back to env var GOOGLE_API_KEY
        api_key = GEMINI_API_KEY or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "Google Gemini API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment."
            )

        # Sanitize model name (library expects names like 'gemini-1.5-flash')
        model_name = GEMINI_MODEL
        if model_name.startswith("models/"):
            model_name = model_name.split("/", 1)[-1]

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=GEMINI_TEMPERATURE,
            api_key=api_key,
            convert_system_message_to_human=True,
        )

        # Fallback list for common Gemini model names if the configured one is unavailable
        self._model_candidates = [
            model_name,
            
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
        ]

    def _compose_prompt(self, instruction: str, contexts: list, json_schema: dict = None) -> str:
        ctx_text = "\n\n".join([f"[section: {c.get('section')}] {c.get('text')}" for c in contexts])
        base = f"""
You are an AI assistant specialized in parsing resumes and job descriptions.

INSTRUCTION:
{instruction}

CONTEXT:
{ctx_text}

Output the result strictly as valid JSON following this schema:
{json.dumps(json_schema, indent=2) if json_schema else 'JSON output required'}
Rules:
- Return ONLY raw JSON. Do not include markdown fences, backticks, or any prose.
- If information is missing, return an empty array or nulls per schema, but keep valid JSON.
"""
        return base

    def extract_requirements_from_jd(self, jd_text: str):
        schema = {
            "type": "object",
            "properties": {
                "requirements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string", "enum": ["skill","education","experience","certification","other"]},
                            "text": {"type": "string"},
                            "importance": {"type": "string", "enum": ["low","medium","high"]}
                        },
                        "required": ["id", "type", "text"]
                    }
                }
            }
        }
        # Always include the raw JD text itself so the model has a reliable source
        contexts = [{"section": "job_description", "text": jd_text}]
        # Optionally enrich with retrieved chunks (best-effort)
        try:
            retrieved = self.retriever.retrieve(jd_text)
            contexts.extend(retrieved or [])
        except Exception:
            pass
        prompt = self._compose_prompt("Extract all requirements from the job description.", contexts, schema)
        resp = self._invoke_with_fallback(prompt)
        parsed = self._parse_json_output(resp)
        if parsed is None:
            return {"raw": resp}
        return parsed

    def parse_resume(self, resume_text: str):
        schema = {
            "type": "object",
            "properties": {
                "contact": {"type": "object"},
                "skills": {"type": "array", "items": {"type": "string"}},
                "education": {"type": "array", "items": {"type": "string"}},
                "experiences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "company": {"type": "string"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"},
                            "bullets": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            }
        }
        # Include the full resume text as primary context
        contexts = [{"section": "resume_full", "text": resume_text}]
        try:
            retrieved = self.retriever.retrieve(resume_text)
            contexts.extend(retrieved or [])
        except Exception:
            pass
        prompt = self._compose_prompt("Parse this resume into a structured JSON format.", contexts, schema)
        resp = self._invoke_with_fallback(prompt)
        parsed = self._parse_json_output(resp)
        return parsed if parsed is not None else {"raw": resp}

    def _invoke_with_fallback(self, prompt_text: str) -> str:
        """Invoke the LLM; on NotFound model errors, try fallback model names automatically."""
        # Try current config first, then fallbacks
        last_err = None
        tried = []
        for i, m in enumerate(self._model_candidates):
            try:
                if i > 0:
                    # Switch model and retry
                    self.llm = ChatGoogleGenerativeAI(
                        model=m,
                        temperature=GEMINI_TEMPERATURE,
                        api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                        convert_system_message_to_human=True,
                    )
                chain = ChatPromptTemplate.from_template("{text}") | self.llm | StrOutputParser()
                return chain.invoke({"text": prompt_text})
            except Exception as e:
                msg = str(e)
                last_err = e
                tried.append(m)
                if ("NotFound" in msg) or ("is not found" in msg):
                    # try next candidate
                    continue
                # Not a model-not-found error; re-raise
                raise
        # Exhausted candidates
        raise RuntimeError(
            f"All candidate Gemini models failed with NotFound. Tried: {tried}. "
            f"Set a supported model in GEMINI_MODEL env var or Streamlit sidebar. Last error: {last_err}"
        )

    def _parse_json_output(self, text: str):
        """Best-effort parse: strip code fences, extract JSON blob, then json.loads."""
        if text is None:
            return None
        s = text.strip()
        # Strip markdown code fences ```json ... ``` or ``` ... ```
        fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
        if fence:
            s = fence.group(1).strip()
        # If still not valid, try to locate the first JSON object/array
        if not (s.startswith("{") or s.startswith("[")):
            start_obj = s.find("{")
            start_arr = s.find("[")
            starts = [x for x in [start_obj, start_arr] if x != -1]
            if starts:
                s = s[min(starts):]
        # Try to trim trailing non-JSON
        # Heuristic: balance braces/brackets
        def balanced_substring(t):
            stack = []
            end = 0
            for i, ch in enumerate(t):
                if ch in "[{":
                    stack.append(ch)
                elif ch in "]}":
                    if not stack:
                        break
                    stack.pop()
                    if not stack:
                        end = i + 1
                        break
            return t[:end] if end else t
        if s:
            s = balanced_substring(s)
        try:
            return json.loads(s)
        except Exception:
            return None
