import re
import json
import fitz

def split_sections(text):
    sections = {}
    current_section = None
    valid_sections = ['skills', 'education', 'work experience', 'project']

    for line in text.splitlines():
        line_strip = line.strip().lower()
        if line_strip in valid_sections:
            current_section = line_strip
            sections[current_section] = []
        elif current_section:
            sections[current_section].append(line.strip())

    # Join lines back into a single string per section
    for sec in sections:
        sections[sec] = '\n'.join(sections[sec]).strip()
    return sections

def clean_list(text):
    # Remove common bullet characters before splitting
    text = re.sub(r'[•\u2022\u2023\u25E6\u2043\u2219]', '', text)
    lines = re.split(r'[\n]+', text)
    cleaned = [line.strip().lower() for line in lines if line.strip()]
    return cleaned

def extract_skills(skills_text):
    return clean_list(skills_text)

def extract_emails(text):
    return re.findall(r'[\w\.-]+@[\w\.-]+', text)

def extract_phone(text):
    phone_pattern = re.compile(r'(\+?\d[\d\s\-\(\)]{5,14}\d)')
    phones = phone_pattern.findall(text)
    phones = [re.sub(r'\D', '', p) for p in phones]
    return list(set(phones))

def extract_name(text):
    match = re.search(r'Name:\s*(.+)', text)
    if match:
        return match.group(1).strip()
    return ""

def extract_list_section(section_text):
    # Simply split lines, ignore empty lines or all-caps lines (section headers)
    lines = [line.strip() for line in section_text.split('\n')]
    filtered = [line for line in lines if line and not re.match(r'^[A-Z\s]+$', line)]
    return filtered

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def parse_resume(resume_text):
    sections = split_sections(resume_text)

    data = {
        "name": extract_name(resume_text),
        "email": extract_emails(resume_text),
        "phone": extract_phone(resume_text),
        "skills": extract_skills(sections.get('skills', '')),
        "education": extract_list_section(sections.get('education', '')),
        "employment": extract_list_section(sections.get('work experience', '')),
        "projects": extract_list_section(sections.get('project', '')),
    }
    return data


if __name__ == "__main__":
    resume_path ="test.pdf" 
    resume_text = extract_text_from_pdf(resume_path)
    extracted = parse_resume(resume_text)
    with open("resume_data.json", "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=4, ensure_ascii=False)

    print("Extracted data saved to resume_data.json")