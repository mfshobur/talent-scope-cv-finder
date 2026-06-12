"""
Generate 20 synthetic employee folders (emp_002–emp_021).
Each folder contains: cv.pdf + assessments/{psychotest,technical,behavioral,english}.pdf

LLM generates content; reportlab renders to PDF with varied layouts.
Run: python scripts/generate_synthetic_data.py
"""

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import litellm
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

from config.settings import Settings

DATA_DIR = Path(__file__).parent.parent / "data" / "employees"

EMPLOYEES = [
    {"id": "emp_002", "name": "Arini Kusumawati",      "dept": "IT",         "role": "Data Engineer",                   "lang": "bilingual",  "layout": "two_col",    "skills": ["Apache Spark", "dbt", "Apache Airflow", "PostgreSQL", "Python", "Kafka"],             "exp_years": 4},
    {"id": "emp_003", "name": "Bagas Prasetyo",         "dept": "IT",         "role": "Backend Developer",               "lang": "english",    "layout": "single_col", "skills": ["Java", "Spring Boot", "Kubernetes", "PostgreSQL", "Redis", "Docker"],                "exp_years": 5},
    {"id": "emp_004", "name": "Citra Dewi Lestari",     "dept": "IT",         "role": "Cloud Infrastructure Engineer",   "lang": "english",    "layout": "table",      "skills": ["AWS", "Terraform", "Kubernetes", "CI/CD", "Linux", "Python"],                       "exp_years": 3},
    {"id": "emp_005", "name": "Dimas Wahyu Nugroho",    "dept": "IT",         "role": "DevOps Engineer",                 "lang": "bilingual",  "layout": "two_col",    "skills": ["Docker", "Kubernetes", "Jenkins", "Ansible", "Prometheus", "Python"],              "exp_years": 4},
    {"id": "emp_006", "name": "Eka Fitriani Putri",     "dept": "IT",         "role": "Cybersecurity Analyst",           "lang": "english",    "layout": "single_col", "skills": ["Penetration Testing", "SIEM", "Firewall", "ISO 27001", "Python", "Wireshark"],     "exp_years": 3},
    {"id": "emp_007", "name": "Fajar Nugroho Santoso",  "dept": "Finance",    "role": "Financial Controller",            "lang": "indonesian", "layout": "table",      "skills": ["PSAK", "SAP", "Financial Reporting", "Audit", "Excel", "Power BI"],               "exp_years": 8},
    {"id": "emp_008", "name": "Gita Rahayu Wibowo",     "dept": "Finance",    "role": "Treasury Analyst",                "lang": "bilingual",  "layout": "single_col", "skills": ["Cash Flow Management", "FX Hedging", "Bloomberg", "Excel", "SAP"],                "exp_years": 4},
    {"id": "emp_009", "name": "Hendra Kusuma Adi",      "dept": "Finance",    "role": "Tax Specialist",                  "lang": "indonesian", "layout": "two_col",    "skills": ["Perpajakan Indonesia", "e-SPT", "Transfer Pricing", "SAP", "Excel"],              "exp_years": 5},
    {"id": "emp_010", "name": "Indah Permata Sari",     "dept": "Finance",    "role": "Internal Auditor",                "lang": "english",    "layout": "single_col", "skills": ["Risk Assessment", "SOX", "ACL Analytics", "CISA", "Excel", "SQL"],               "exp_years": 6},
    {"id": "emp_011", "name": "Joko Santoso Budiman",   "dept": "HR",         "role": "HR Business Partner",             "lang": "indonesian", "layout": "table",      "skills": ["HRIS", "Performance Management", "Talent Development", "Labour Law", "SAP HCM"], "exp_years": 7},
    {"id": "emp_012", "name": "Kartika Sari Dewi",      "dept": "HR",         "role": "Talent Acquisition Specialist",   "lang": "bilingual",  "layout": "two_col",    "skills": ["Sourcing", "ATS", "Behavioral Interviewing", "LinkedIn Recruiter", "Excel"],     "exp_years": 3},
    {"id": "emp_013", "name": "Lukman Hakim Pratama",   "dept": "HR",         "role": "Learning & Development Coordinator", "lang": "bilingual", "layout": "single_col", "skills": ["Instructional Design", "LMS", "Facilitating", "Kirkpatrick Model", "PowerPoint"], "exp_years": 4},
    {"id": "emp_014", "name": "Mega Wulandari Agung",   "dept": "Marketing",  "role": "Brand Manager",                   "lang": "english",    "layout": "two_col",    "skills": ["Brand Strategy", "Market Research", "ATL/BTL", "Adobe Creative", "Nielsen"],     "exp_years": 6},
    {"id": "emp_015", "name": "Nanda Putra Wijaya",     "dept": "Marketing",  "role": "Digital Marketing Specialist",    "lang": "bilingual",  "layout": "single_col", "skills": ["SEO/SEM", "Google Ads", "Meta Ads", "Analytics", "Content Creation"],           "exp_years": 3},
    {"id": "emp_016", "name": "Oktavia Dewi Anggraini", "dept": "Marketing",  "role": "Market Research Analyst",         "lang": "english",    "layout": "table",      "skills": ["Quantitative Research", "SPSS", "Focus Groups", "Competitive Analysis", "Excel"], "exp_years": 4},
    {"id": "emp_017", "name": "Pratama Setiawan Hadi",  "dept": "Operations", "role": "Supply Chain Coordinator",        "lang": "indonesian", "layout": "two_col",    "skills": ["SAP MM", "Inventory Management", "Procurement", "Logistics", "Excel"],          "exp_years": 5},
    {"id": "emp_018", "name": "Qori Amelia Putri",      "dept": "Operations", "role": "Process Improvement Specialist",  "lang": "bilingual",  "layout": "single_col", "skills": ["Lean Six Sigma", "DMAIC", "Value Stream Mapping", "Minitab", "Project Mgmt"],   "exp_years": 4},
    {"id": "emp_019", "name": "Rizky Firmansyah",       "dept": "Operations", "role": "Logistics Manager",               "lang": "indonesian", "layout": "table",      "skills": ["Fleet Management", "WMS", "3PL Management", "SAP TM", "Cost Optimization"],     "exp_years": 7},
    {"id": "emp_020", "name": "Siti Nurhaliza Rahmat",  "dept": "Legal",      "role": "Legal Counsel",                   "lang": "bilingual",  "layout": "two_col",    "skills": ["Hukum Perdata", "Contract Drafting", "Regulatory Compliance", "Litigation", "M&A"], "exp_years": 6},
    {"id": "emp_021", "name": "Tommy Hartono Susilo",   "dept": "IT",         "role": "Data Analyst",                    "lang": "bilingual",  "layout": "single_col", "skills": ["SQL", "Python", "Power BI", "Tableau", "Excel", "Statistical Analysis"],         "exp_years": 3},
]

ASSESSMENT_SCORES = {
    # (psychotest_label, technical_score, behavioral_label, english_cefr)
    "emp_002": ("Above Average", 82, "Proficient", "B2"),
    "emp_003": ("Average",       75, "Developing", "B1"),
    "emp_004": ("Above Average", 79, "Proficient", "C1"),
    "emp_005": ("Average",       71, "Developing", "B2"),
    "emp_006": ("High",          90, "Proficient", "C1"),
    "emp_007": ("Average",       68, "Proficient", "B1"),
    "emp_008": ("Above Average", 74, "Proficient", "B2"),
    "emp_009": ("Average",       70, "Average",    "A2"),
    "emp_010": ("High",          85, "Proficient", "C1"),
    "emp_011": ("Average",       65, "High",       "B2"),
    "emp_012": ("Above Average", 72, "High",       "B2"),
    "emp_013": ("Average",       68, "High",       "B1"),
    "emp_014": ("Above Average", 74, "Proficient", "C1"),
    "emp_015": ("Average",       77, "Proficient", "B2"),
    "emp_016": ("Above Average", 80, "Proficient", "C1"),
    "emp_017": ("Average",       66, "Average",    "B1"),
    "emp_018": ("Above Average", 83, "Proficient", "B2"),
    "emp_019": ("Average",       69, "Average",    "A2"),
    "emp_020": ("High",          78, "High",       "C1"),
    "emp_021": ("Average",       76, "Proficient", "B2"),
}


def _llm_call(prompt: str, s: Settings, max_tokens: int = 900) -> str:
    resp = litellm.completion(
        model=s.llm_model,
        messages=[{"role": "user", "content": prompt}],
        api_key=s.effective_llm_api_key(),
        base_url=s.llm_base_url or None,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# ── PDF helpers ────────────────────────────────────────────────────────────────

def _styles():
    ss = getSampleStyleSheet()
    custom = [
        ParagraphStyle("CVName",  fontSize=18, fontName="Helvetica-Bold",  spaceAfter=2),
        ParagraphStyle("CVTitle", fontSize=11, fontName="Helvetica",       spaceAfter=4, textColor=colors.HexColor("#555555")),
        ParagraphStyle("Section", fontSize=11, fontName="Helvetica-Bold",  spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#1a1a1a")),
        ParagraphStyle("Body",    fontSize=9,  fontName="Helvetica",       spaceAfter=2, leading=13),
        ParagraphStyle("Bullet",  fontSize=9,  fontName="Helvetica",       spaceAfter=1, leftIndent=12, leading=13),
        ParagraphStyle("Small",   fontSize=8,  fontName="Helvetica",       textColor=colors.HexColor("#666666")),
        ParagraphStyle("Center",  fontSize=9,  fontName="Helvetica",       alignment=TA_CENTER),
        ParagraphStyle("Header",  fontSize=14, fontName="Helvetica-Bold",  alignment=TA_CENTER, spaceAfter=4),
    ]
    for style in custom:
        if style.name not in ss:
            ss.add(style)
    return ss


def _render_single_col(path: Path, content: dict):
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    ss = _styles()
    story = []
    story.append(Paragraph(content["name"], ss["CVName"]))
    story.append(Paragraph(content["role"] + " · " + content["dept"], ss["CVTitle"]))
    story.append(Paragraph(content.get("contact", ""), ss["Small"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc"), spaceAfter=6))

    for section, items in content["sections"].items():
        story.append(Paragraph(section.upper(), ss["Section"]))
        for item in items:
            if item.startswith("•"):
                story.append(Paragraph(item, ss["Bullet"]))
            else:
                story.append(Paragraph(item, ss["Body"]))
        story.append(Spacer(1, 4))

    doc.build(story)


def _render_two_col(path: Path, content: dict):
    from reportlab.platypus import KeepInFrame
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    ss = _styles()
    story = []
    story.append(Paragraph(content["name"], ss["CVName"]))
    story.append(Paragraph(content["role"] + " · " + content["dept"], ss["CVTitle"]))
    story.append(Paragraph(content.get("contact", ""), ss["Small"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2c5f8a"), spaceAfter=8))

    left_sections = []
    right_sections = []
    items = list(content["sections"].items())
    mid = len(items) // 2
    for section, lines in items[:mid]:
        left_sections.append(Paragraph(section.upper(), ss["Section"]))
        for line in lines:
            left_sections.append(Paragraph(line, ss["Bullet"] if line.startswith("•") else ss["Body"]))
        left_sections.append(Spacer(1, 6))
    for section, lines in items[mid:]:
        right_sections.append(Paragraph(section.upper(), ss["Section"]))
        for line in lines:
            right_sections.append(Paragraph(line, ss["Bullet"] if line.startswith("•") else ss["Body"]))
        right_sections.append(Spacer(1, 6))

    col_w = (A4[0] - 3*cm) / 2
    table = Table([[left_sections, right_sections]], colWidths=[col_w, col_w])
    table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(table)
    doc.build(story)


def _render_table_style(path: Path, content: dict):
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    ss = _styles()
    story = []

    header_table = Table([[
        Paragraph(content["name"], ss["CVName"]),
        Paragraph(content.get("contact", ""), ss["Small"]),
    ]], colWidths=["65%", "35%"])
    header_table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "BOTTOM")]))
    story.append(header_table)
    story.append(Paragraph(content["role"] + " · " + content["dept"], ss["CVTitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a3c6b"), spaceAfter=8))

    for section, items in content["sections"].items():
        section_row = [[Paragraph(section.upper(), ss["Section"]), ""]]
        t = Table(section_row, colWidths=["30%", "70%"])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,0), colors.HexColor("#eef2f7")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(t)
        for item in items:
            story.append(Paragraph(item, ss["Bullet"] if item.startswith("•") else ss["Body"]))
        story.append(Spacer(1, 4))

    doc.build(story)


def render_cv_pdf(path: Path, content: dict, layout: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if layout == "two_col":
        _render_two_col(path, content)
    elif layout == "table":
        _render_table_style(path, content)
    else:
        _render_single_col(path, content)


def render_assessment_pdf(path: Path, title: str, body_paragraphs: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    ss = _styles()
    story = [
        Paragraph("LAPORAN ASESMEN", ss["Header"]),
        Paragraph(title, ss["CVTitle"]),
        HRFlowable(width="100%", thickness=1, color=colors.grey, spaceAfter=8),
    ]
    for para in body_paragraphs:
        if para.startswith("##"):
            story.append(Paragraph(para[2:].strip(), ss["Section"]))
        elif para.startswith("•"):
            story.append(Paragraph(para, ss["Bullet"]))
        else:
            story.append(Paragraph(para, ss["Body"]))
        story.append(Spacer(1, 3))
    doc.build(story)


# ── Content generators ─────────────────────────────────────────────────────────

def generate_cv_content(emp: dict, s: Settings) -> dict:
    skills_str = ", ".join(emp["skills"])
    lang_instruction = {
        "bilingual": "Campurkan Bahasa Indonesia dan Inggris secara natural (seperti profesional Indonesia yang terbiasa dengan kedua bahasa).",
        "english": "Tulis sepenuhnya dalam Bahasa Inggris.",
        "indonesian": "Tulis sepenuhnya dalam Bahasa Indonesia.",
    }[emp["lang"]]

    prompt = f"""{lang_instruction}

Buat konten CV realistis untuk:
- Nama: {emp["name"]}
- Jabatan Saat Ini: {emp["role"]}
- Departemen: {emp["dept"]}
- Pengalaman Kerja: ~{emp["exp_years"]} tahun
- Keahlian Utama: {skills_str}

Kembalikan HANYA JSON dengan format berikut (tanpa markdown, langsung JSON):
{{
  "contact": "email@company.com | +62 8xx xxxx xxxx | LinkedIn: linkedin.com/in/...",
  "sections": {{
    "Professional Summary": ["2-3 kalimat ringkasan profesional"],
    "Work Experience": [
      "PT [Nama Perusahaan] — {emp["role"]} (20XX–Sekarang)",
      "• Bullet point pencapaian 1",
      "• Bullet point pencapaian 2",
      "• Bullet point pencapaian 3",
      "PT [Nama Perusahaan Sebelumnya] — [Jabatan] (20XX–20XX)",
      "• Bullet point pencapaian 1",
      "• Bullet point pencapaian 2"
    ],
    "Education": [
      "S1 [Jurusan], [Nama Universitas] — 20XX",
      "IPK: 3.XX / 4.00"
    ],
    "Technical Skills": ["• {skills_str}", "• [Tambahkan 2-3 skill relevan lainnya]"],
    "Certifications": ["• [Sertifikasi relevan jika ada]"],
    "Languages": ["• Bahasa Indonesia: Native", "• English: [Level sesuai konteks]"]
  }}
}}

Buat konten yang spesifik dan realistis. Gunakan nama perusahaan Indonesia yang masuk akal."""

    raw = _llm_call(prompt, s, max_tokens=800)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("```").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Simple fallback
        data = {
            "contact": f"employee@company.com | +62 812 xxxx xxxx",
            "sections": {
                "Professional Summary": [f"Profesional berpengalaman sebagai {emp['role']} dengan {emp['exp_years']} tahun pengalaman."],
                "Technical Skills": [f"• {', '.join(emp['skills'])}"],
                "Languages": ["• Bahasa Indonesia: Native", "• English: Professional"],
            }
        }

    data["name"] = emp["name"]
    data["role"] = emp["role"]
    data["dept"] = emp["dept"]
    return data


def generate_psychotest_content(emp: dict, score_label: str, s: Settings) -> list[str]:
    prompt = f"""Buat laporan psikotes singkat dalam Bahasa Indonesia untuk:
Nama: {emp["name"]} | Posisi: {emp["role"]}
Hasil keseluruhan: {score_label}

Sertakan:
1. Profil DISC (D/I/S/C dengan deskripsi singkat)
2. Big Five traits (3-4 traits dominan)
3. Ringkasan kepribadian (2-3 kalimat)
4. Rekomendasi pengembangan (2 poin)

Kembalikan sebagai teks paragraf biasa dengan section headers diawali ##."""

    text = _llm_call(prompt, s, max_tokens=400)
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return [f"## Nama: {emp['name']} | Jabatan: {emp['role']}",
            f"## Hasil Keseluruhan: {score_label}"] + paras


def generate_technical_content(emp: dict, score: int, s: Settings) -> list[str]:
    skills_str = ", ".join(emp["skills"][:3])
    lang = "Bahasa Indonesia" if emp["lang"] == "indonesian" else "English"
    prompt = f"""Create a technical assessment report in {lang} for:
Name: {emp["name"]} | Role: {emp["role"]}
Score: {score}/100 | Key skills tested: {skills_str}

Include:
1. Score breakdown table (3-4 categories)
2. Strengths observed (2-3 bullets)
3. Areas for improvement (2 bullets)
4. Overall recommendation

Use ## for section headers."""

    text = _llm_call(prompt, s, max_tokens=400)
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return [f"## Nama: {emp['name']} | Jabatan: {emp['role']}",
            f"## Skor Total: {score}/100"] + paras


def generate_behavioral_content(emp: dict, label: str, s: Settings) -> list[str]:
    prompt = f"""Buat laporan asesmen behavioral (kompetensi) dalam Bahasa Indonesia untuk:
Nama: {emp["name"]} | Posisi: {emp["role"]}
Hasil keseluruhan: {label}

Nilai kompetensi berikut (skala 1-5) dengan deskripsi singkat:
- Kepemimpinan
- Kolaborasi & Teamwork
- Problem Solving
- Komunikasi
- Integritas & Etika Kerja

Gunakan ## untuk section headers."""

    text = _llm_call(prompt, s, max_tokens=400)
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return [f"## Nama: {emp['name']} | Jabatan: {emp['role']}",
            f"## Hasil Keseluruhan: {label}"] + paras


def generate_english_content(emp: dict, cefr: str, s: Settings) -> list[str]:
    descriptors = {
        "A2": "Elementary — dapat memahami kalimat sederhana",
        "B1": "Intermediate — dapat berkomunikasi dalam situasi umum",
        "B2": "Upper Intermediate — dapat berkomunikasi dengan lancar tentang topik familiar",
        "C1": "Advanced — dapat menggunakan bahasa secara fleksibel dan efektif",
    }
    desc = descriptors.get(cefr, "")
    prompt = f"""Create an English proficiency assessment report for:
Name: {emp["name"]} | Role: {emp["role"]}
Overall CEFR Level: {cefr} — {desc}

Include:
1. Sub-scores: Reading, Writing, Listening, Speaking (all near {cefr} level with slight variation)
2. Brief descriptor for each skill
3. Overall recommendation

Use ## for section headers. Write in English."""

    text = _llm_call(prompt, s, max_tokens=350)
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return [f"## Name: {emp['name']} | Position: {emp['role']}",
            f"## Overall CEFR Level: {cefr} — {desc}"] + paras


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_employee(emp: dict, s: Settings):
    emp_dir = DATA_DIR / emp["id"]
    asm_dir = emp_dir / "assessments"
    cv_path = emp_dir / "cv.pdf"

    if cv_path.exists():
        print(f"  [skip] {emp['id']} already exists")
        return

    print(f"  Generating {emp['id']} — {emp['name']} ({emp['role']}) ...", flush=True)

    # CV
    content = generate_cv_content(emp, s)
    render_cv_pdf(cv_path, content, emp["layout"])

    scores = ASSESSMENT_SCORES[emp["id"]]
    psycho_label, tech_score, beh_label, eng_cefr = scores

    # Psychotest
    paras = generate_psychotest_content(emp, psycho_label, s)
    render_assessment_pdf(asm_dir / "psychotest.pdf", f"Tes Psikologi — {emp['name']}", paras)

    # Technical
    paras = generate_technical_content(emp, tech_score, s)
    render_assessment_pdf(asm_dir / "technical.pdf", f"Technical Assessment — {emp['name']}", paras)

    # Behavioral
    paras = generate_behavioral_content(emp, beh_label, s)
    render_assessment_pdf(asm_dir / "behavioral.pdf", f"Behavioral Assessment — {emp['name']}", paras)

    # English proficiency
    paras = generate_english_content(emp, eng_cefr, s)
    render_assessment_pdf(asm_dir / "english.pdf", f"English Proficiency — {emp['name']}", paras)

    print(f"    Done: cv.pdf + 4 assessment PDFs")


def main():
    s = Settings()
    if not s.effective_llm_api_key():
        print("ERROR: No API key found. Set LLM_API_KEY or OPENAI_API_KEY in .env")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(EMPLOYEES)} synthetic employees in {DATA_DIR}\n")

    for emp in EMPLOYEES:
        generate_employee(emp, s)

    total = sum(1 for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("emp_"))
    print(f"\nDone. {total} employee folders in {DATA_DIR}")


if __name__ == "__main__":
    main()
