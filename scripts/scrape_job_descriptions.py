"""
Scrape 7 job descriptions from Indeed.co.id.
Falls back to LLM-generated JDs if scraping fails (JS-rendered / rate-limited).
"""

import os
import sys
import time
import json
import re
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "job_descriptions"

JD_SPECS = [
    {
        "filename": "ai_ml_engineer.txt",
        "title": "AI/ML Engineer",
        "indeed_query": "AI ML Engineer",
        "skills": ["Python", "PyTorch", "LangChain", "FastAPI", "Machine Learning"],
        "experience": "2-5 tahun",
        "education": "S1 Teknik Informatika / Ilmu Komputer",
    },
    {
        "filename": "data_analyst.txt",
        "title": "Data Analyst",
        "indeed_query": "Data Analyst",
        "skills": ["SQL", "Python", "Power BI", "Tableau", "Excel"],
        "experience": "1-3 tahun",
        "education": "S1 Teknik Informatika / Statistika / Matematika",
    },
    {
        "filename": "finance_controller.txt",
        "title": "Financial Controller",
        "indeed_query": "Financial Controller",
        "skills": ["PSAK", "SAP", "Financial Reporting", "Audit", "Excel"],
        "experience": "5+ tahun",
        "education": "S1 Akuntansi / Keuangan",
    },
    {
        "filename": "hr_business_partner.txt",
        "title": "HR Business Partner",
        "indeed_query": "HR Business Partner",
        "skills": ["HRIS", "Performance Management", "Talent Development", "Rekrutmen"],
        "experience": "3-5 tahun",
        "education": "S1 Psikologi / Manajemen SDM",
    },
    {
        "filename": "marketing_manager.txt",
        "title": "Marketing Manager",
        "indeed_query": "Marketing Manager",
        "skills": ["Brand Strategy", "Digital Marketing", "Market Research", "Campaign Management"],
        "experience": "4-6 tahun",
        "education": "S1 Marketing / Komunikasi / Bisnis",
    },
    {
        "filename": "operations_supervisor.txt",
        "title": "Operations Supervisor",
        "indeed_query": "Operations Supervisor Supply Chain",
        "skills": ["Supply Chain", "Lean Six Sigma", "ERP", "Logistics", "Procurement"],
        "experience": "3-5 tahun",
        "education": "S1 Teknik Industri / Manajemen Operasi",
    },
    {
        "filename": "legal_counsel.txt",
        "title": "Legal Counsel",
        "indeed_query": "Legal Counsel",
        "skills": ["Hukum Perdata", "Contract Review", "Regulatory Compliance", "Litigasi"],
        "experience": "4+ tahun",
        "education": "S1/S2 Hukum",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_indeed(query: str) -> str | None:
    url = f"https://id.indeed.com/jobs?q={requests.utils.quote(query)}&l=Indonesia"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")

        # Try to find a job card and get the detail link
        cards = soup.select("a.jcs-JobTitle")
        if not cards:
            return None

        detail_url = "https://id.indeed.com" + cards[0].get("href", "")
        time.sleep(1)
        dr = requests.get(detail_url, headers=HEADERS, timeout=10)
        if dr.status_code != 200:
            return None

        dsoup = BeautifulSoup(dr.text, "html.parser")
        desc = dsoup.select_one("#jobDescriptionText")
        if not desc:
            return None

        text = desc.get_text(separator="\n").strip()
        return text if len(text) > 200 else None

    except Exception:
        return None


def generate_jd(spec: dict) -> str:
    try:
        import litellm
        from config.settings import Settings
        s = Settings()
        api_key = s.effective_llm_api_key()
        if not api_key:
            raise ValueError("No API key")

        skills_str = ", ".join(spec["skills"])
        prompt = f"""Buatkan deskripsi pekerjaan (job description) dalam Bahasa Indonesia untuk posisi:

Posisi: {spec["title"]}
Keahlian yang dibutuhkan: {skills_str}
Pengalaman: {spec["experience"]}
Pendidikan: {spec["education"]}

Format:
- Tentang Posisi (2-3 kalimat)
- Tanggung Jawab Utama (5-7 bullet points)
- Kualifikasi yang Dibutuhkan (5-6 bullet points)
- Kualifikasi Tambahan (2-3 bullet points)

Gunakan tone korporat profesional seperti perusahaan holding Indonesia besar. \
Sertakan istilah teknis dalam bahasa Inggris di mana sesuai."""

        resp = litellm.completion(
            model=s.llm_model,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            base_url=s.llm_base_url or None,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        print(f"  LLM fallback failed: {e}")
        return _static_fallback(spec)


def _static_fallback(spec: dict) -> str:
    skills_str = "\n".join(f"- {s}" for s in spec["skills"])
    return f"""DESKRIPSI POSISI: {spec["title"]}

Tentang Posisi:
Kami membuka kesempatan bagi kandidat internal yang berpengalaman dan memiliki semangat tinggi \
untuk bergabung sebagai {spec["title"]} di lingkungan holding kami.

Tanggung Jawab Utama:
- Mengelola dan mengembangkan area kerja sesuai fungsi {spec["title"]}
- Berkolaborasi lintas departemen untuk pencapaian target organisasi
- Menyusun laporan dan analisis berkala kepada manajemen
- Memastikan kepatuhan terhadap SOP dan regulasi yang berlaku
- Mengidentifikasi peluang perbaikan proses dan efisiensi

Kualifikasi yang Dibutuhkan:
- Pendidikan minimal {spec["education"]}
- Pengalaman relevan minimal {spec["experience"]}
{skills_str}
- Kemampuan komunikasi yang baik dalam Bahasa Indonesia dan Inggris

Kualifikasi Tambahan:
- Pengalaman bekerja di lingkungan perusahaan holding atau multinasional
- Kemampuan bekerja dalam tim lintas fungsi
"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving JDs to: {OUTPUT_DIR}\n")

    for spec in JD_SPECS:
        out_path = OUTPUT_DIR / spec["filename"]
        if out_path.exists():
            print(f"  [skip] {spec['filename']} already exists")
            continue

        print(f"  Scraping: {spec['title']} ... ", end="", flush=True)
        text = scrape_indeed(spec["indeed_query"])

        if text:
            print(f"scraped ({len(text)} chars)")
        else:
            print("scrape failed → generating with LLM")
            text = generate_jd(spec)

        out_path.write_text(text, encoding="utf-8")
        print(f"  Saved: {spec['filename']}")
        time.sleep(1.5)

    print(f"\nDone. {len(list(OUTPUT_DIR.glob('*.txt')))} JD files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
