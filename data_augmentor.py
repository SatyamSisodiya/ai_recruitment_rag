# data_augmentor.py
import os, json, random
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY, GEMINI_MODEL
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

def synthesize_resume(role: str, level: str = "mid") -> str:
    prompt = f"""
Generate a realistic resume for a {level}-level professional applying for a {role} role.
Include sections: Contact, Summary, Skills, Education, Experience (3 entries), Projects, Certifications.
Output as plain text resume.
"""
    return llm.invoke(prompt).content

def synthesize_jd(role: str, skills: list) -> str:
    prompt = f"""
Generate a realistic job description for a {role} position requiring skills: {', '.join(skills)}.
Include sections: Overview, Responsibilities, Required Skills, Education, Experience.
Output as plain text JD.
"""
    return llm.invoke(prompt).content

if __name__ == "__main__":
    os.makedirs("data/generated", exist_ok=True)
    roles = ["Data Scientist", "Backend Developer", "AI Engineer"]
    skills_bank = ["Python", "SQL", "TensorFlow", "Docker", "AWS", "Machine Learning"]
    for i, role in enumerate(roles):
        jd = synthesize_jd(role, random.sample(skills_bank, 4))
        with open(f"data/generated/jd_{i}.txt", "w") as f: f.write(jd)
        res = synthesize_resume(role)
        with open(f"data/generated/resume_{i}.txt", "w") as f: f.write(res)
    print("✅ Synthetic resumes & JDs created in data/generated/")
