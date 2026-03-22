import re
import json
import pymupdf 
import fitz  # PyMuPDF

class ResumeParser:
    def __init__(self):
        # Common variations of section headers
        self.patterns = {
            'skills': r'(skills|technical skills|competencies|technologies|tools)',
            'education': r'(education|academic background|qualification|academic profile)',
            'experience': r'(work experience|employment|experience|professional history|internships)',
            'projects': r'(projects|academic projects|personal projects|key projects)'
        }

    def extract_text(self, pdf_path):
        """Extracts and cleans raw text from PDF."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            # Clean hidden PDF artifacts
            text = text.replace('\xa0', ' ').replace('\x0c', ' ')
            text = re.sub(r'[•\u2022\u2023\u25E6\u2043\u2219\uf0b7]', '-', text)
            return text
        except Exception as e:
            return f"Error reading PDF: {e}"

    def split_sections(self, text):
        """Divides resume text into logical blocks."""
        sections = {key: [] for key in self.patterns.keys()}
        current_section = None
        
        lines = text.splitlines()
        for line in lines:
            line_strip = line.strip()
            if not line_strip: continue
            
            # Check if line is a header
            is_header = False
            for section_key, pattern in self.patterns.items():
                if re.search(r'^' + pattern + r'$', line_strip.lower()):
                    current_section = section_key
                    is_header = True
                    break
            
            if not is_header and current_section:
                sections[current_section].append(line_strip)
        
        return {k: '\n'.join(v) for k, v in sections.items()}

    def get_contact_info(self, text):
        email = re.findall(r'[\w\.-]+@[\w\.-]+', text)
        phone = re.findall(r'(\+?\d[\d\s\-\(\)]{8,14}\d)', text)
        # Name heuristic: First non-empty line usually contains the name
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        name = lines[0] if lines else "Unknown"
        return {
            "name": name,
            "email": email[0] if email else "N/A",
            "phone": phone[0] if phone else "N/A"
        }

    def parse(self, pdf_path):
        raw_text = self.extract_text(pdf_path)
        contact = self.get_contact_info(raw_text)
        sections = self.split_sections(raw_text)

        # Final structured data
        return {
            "admin_view": {
                "candidate_name": contact['name'],
                "contact_info": f"{contact['email']} | {contact['phone']}",
                "top_skills": [s.strip() for s in sections['skills'].split('\n') if s.strip()][:8],
                "latest_education": sections['education'].split('\n')[0] if sections['education'] else "N/A",
                "experience_summary": sections['experience'].split('\n')[:3], # First 3 lines
                "project_count": len([p for p in sections['projects'].split('\n') if len(p) > 10])
            },
            "raw_sections": sections
        }

# --- Execution ---
if __name__ == "__main__":
    parser = ResumeParser()
    
    # Path to your test file
    FILE_PATH = "test.pdf" 
    
    result = parser.parse(FILE_PATH)

    # 1. Save to JSON for system use
    with open("processed_resume.json", "w") as f:
        json.dump(result, f, indent=4)

    # 2. Display "Easy" format for Admin
    view = result["admin_view"]
    print("\n" + "="*40)
    print(f" ADMIN DASHBOARD: {view['candidate_name'].upper()} ")
    print("="*40)
    print(f"📧 Contact: {view['contact_info']}")
    print(f"🎓 Education: {view['latest_education']}")
    print(f"🛠 Skills: {', '.join(view['top_skills'])}")
    print(f"📂 Projects: Found {view['project_count']} major items")
    print("-" * 40)
    print("Recent Experience Preview:")
    for line in view['experience_summary']:
        print(f" • {line}")
    print("="*40)