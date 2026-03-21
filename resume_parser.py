import re
import spacy
import pdfplumber
import docx2txt
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")

# Load embedding model for better skill/context extraction
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
    elif file_path.endswith(".docx"):
        return docx2txt.process(file_path)
    else:
        with open(file_path, 'r', errors="ignore") as f:
            return f.read()

def extract_skills(text, skills_file="skills.txt"):
    with open(skills_file, "r") as f:
        skills_list = [s.strip().lower() for s in f.readlines()]

    text_lower = text.lower()
    found = []

    for skill in skills_list:
        if skill in text_lower:
            found.append(skill)

    return list(set(found))

def extract_experience(text):
    years = re.findall(r"(\d+)\+?\s+years?", text.lower())
    max_years = max([int(y) for y in years], default=0)

    titles = re.findall(r"(intern|engineer|developer|manager|analyst)", text.lower())

    return {
        "years_of_experience": max_years,
        "roles": list(set(titles))
    }

def extract_education(text):
    degrees = re.findall(r"(bachelor|master|phd|b\.tech|m\.tech|be|bs|ms)", text.lower())
    return list(set(degrees))

def extract_projects(text):
    doc = nlp(text)
    lines = text.split("\n")
    project_lines = []

    for i, line in enumerate(lines):
        if "project" in line.lower():
            for j in range(i, min(i+5, len(lines))):
                project_lines.append(lines[j])

    return list(set(project_lines))

def parse_resume(file_path):
    text = extract_text(file_path)

    return {
        "skills": extract_skills(text),
        "experience": extract_experience(text),
        "education": extract_education(text),
        "projects": extract_projects(text),
        "raw_text": text
    }
    
if __name__ == "__main__":
    file_path = input("Enter resume file path: ")  #<-- Provide the path to the resume file here

    parsed = parse_resume(file_path)

    print("\n=== Extracted Resume Information ===")
    for key, value in parsed.items():
        print(f"\n{key.upper()}:")
        print(value)

    # Load job description
    with open("job.txt", "r") as f:
        job_text = f.read()

    # Embedding similarity
    resume_embedding = embed_model.encode(parsed["raw_text"], convert_to_tensor=True)
    job_embedding = embed_model.encode(job_text, convert_to_tensor=True)

    similarity = util.cos_sim(resume_embedding, job_embedding)

    print("\nMatch Score (0 to 1):", float(similarity))
