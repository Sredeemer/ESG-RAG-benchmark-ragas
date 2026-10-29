"""
Download curated 2025 and 2026 Corporate Sustainability / CSRD / Integrated Annual
Report PDFs (extending existing longitudinal companies in data/raw/ plus 2025-2026
CSRD/TCFD benchmarks) and convert them to Markdown via 00a_ingest_pdfs.py.
"""

from pathlib import Path
import requests

PDF_DIR = Path("data/raw_pdfs_2025_2026")

# Format: (filename: "{companyID}_{YEAR}_{slug}.pdf", url, company_name)
CURATED_REPORTS = [
    # --- 2025 Reports (Longitudinal Panel Extensions + CSRD Statements) ---
    (
        "6_2025_bayer.pdf",
        "https://reports.bayer.com/annual-report-2025/en/_assets/downloads/entire-bayer-ar25.pdf",
        "Bayer AG (Annual & CSRD Sustainability Report 2025)",
    ),
    (
        "56_2025_omv.pdf",
        "https://www.omv.com/downloads/2026/04/a4b668cf-7243-ffad-4a26-6f7f40b70374/OMV%20Combined%20Annual%20Report%202025.pdf",
        "OMV Group (Combined Annual & ESRS Sustainability Report 2025)",
    ),
    (
        "87_2025_deutsche_bank.pdf",
        "https://investor-relations.db.com/files/documents/other-presentations-and-events/2025/Annual-Report-2025.pdf?language_id=1",
        "Deutsche Bank (Annual & Non-Financial Report 2025)",
    ),
    (
        "27_2025_arcelormittal.pdf",
        "https://cdn.arcelormittal.com/media/kyqeahax/2025-sd-report-22-april-2026-final-upload.pdf",
        "ArcelorMittal (Sustainability Report 2025)",
    ),
    (
        "62_2025_novo_nordisk.pdf",
        "https://annualreport.novonordisk.com/2025/_assets/downloads/novo-nordisk-annual-report-2025.pdf",
        "Novo Nordisk (Annual Report & Sustainability Statement 2025)",
    ),
    (
        "55_2025_deutsche_telekom.pdf",
        "https://report.telekom.com/annual-report-2025/_assets/downloads/entire-dtag-ar25.pdf",
        "Deutsche Telekom (Annual & CSRD Sustainability Report 2025)",
    ),
    (
        "52_2025_continental.pdf",
        "https://cdn.continental.com/fileadmin/__imported/sites/corporate/_international/english/hubpages/30_20investors/30_20reports/annual_20reports/downloads/annual-report-2025.pdf",
        "Continental AG (Annual & Sustainability Report 2025)",
    ),
    (
        "18_2025_engie.pdf",
        "https://www.engie.com/sites/default/files/assets/documents/2026-03/ESG%20%40%20ENGIE%202025%20VDEF_1.pdf",
        "ENGIE (ESG & Sustainability Report 2025)",
    ),
    (
        "901_2025_abb.pdf",
        "https://library.e.abb.com/public/ea4348e21dce4e8490929414bb7c9ef3/ABB%20Sustainability%20Statement%202025.pdf?x-sign=OtNFIxww0hgnlXEVGqscxDqLEAjEieED3PFDikIfNyRR0dc2ZPNnmMz7CF19B9dN",
        "ABB (CSRD Sustainability Statement 2025)",
    ),
    (
        "903_2025_sanofi.pdf",
        "https://www.sanofi.com/assets/dotcom/content-app/publications/esg-reports/sanofi-sustainability-statement-2025.pdf",
        "Sanofi (CSRD Sustainability Statement 2025)",
    ),
    (
        "910_2025_ryanair.pdf",
        "https://corporate.ryanair.com/wp-content/uploads/2025/09/Ryanair-Sustainability-Report-2025.pdf",
        "Ryanair (Sustainability Report 2025)",
    ),
    # --- 2026 Reports (FY2025/2026 Published in 2026) ---
    (
        "79_2026_sse.pdf",
        "https://www.sse.com/media/gugb5kyl/sse_ar26_interactive_18mb.pdf",
        "SSE plc (Annual & Sustainability Report 2026)",
    ),
    (
        "905_2026_rsgroup.pdf",
        "https://www.rsgroup.com/media/xwtnpfvs/rs_group_esg_2026_interactive-and-accessible.pdf",
        "RS Group plc (ESG Report 2026)",
    ),
    (
        "906_2026_smulders.pdf",
        "https://www.smulders.com/swfiles/files/Sustainability-Report-2026-over-the-year-2025.pdf",
        "Smulders (Sustainability Report 2026)",
    ),
    (
        "908_2026_tullow.pdf",
        "https://www.tullowoil.com/media/utlliqwg/tullow-oil-plc-annual-report-and-accounts-2025.pdf",
        "Tullow Oil plc (Annual & TCFD Report 2026)",
    ),
    (
        "912_2026_nextplc.pdf",
        "https://www.nextplc.co.uk/~/media/Files/N/next-plc-v4/documents/corporate-responsibility/corporate-responsibility-report-2026.pdf",
        "NEXT plc (Corporate Responsibility Report 2026)",
    ),
    (
        "913_2026_esgpro.pdf",
        "https://esgpro.co.uk/wp-content/uploads/2026/04/ESG-Pro-Annual-Sustainability-and-ESG-Impact-Report-2026-rev-1.pdf",
        "ESG Pro (Annual Sustainability & ESG Impact Report 2026)",
    ),
    (
        "914_2026_acca.pdf",
        "https://www.accaglobal.com/content/dam/ACCA_Global/disc/agm-annual-review/2026/acca-annual-integrated-report-2026.pdf",
        "ACCA (Annual Integrated Report 2026)",
    ),
    (
        "915_2026_mcbgroup.pdf",
        "https://mcbgroup.com/docs/default-source/sustainability/mcbg-sr-2026.pdf",
        "MCB Group (Sustainability Report 2026)",
    ),
    (
        "916_2026_icg.pdf",
        "https://www.icgam.com/wp-content/uploads/2026/06/7jjrt92iiq5v-ICG_AR2026_TCFD.pdf",
        "ICG plc (Annual & TCFD Report 2026)",
    ),
    (
        "917_2026_cae.pdf",
        "https://www.cae.com/content/docs/FY26_Global_Annual_Activity_and_Sustainability_Report_EN.pdf",
        "CAE Inc. (Global Annual Activity and Sustainability Report FY2026)",
    ),
]


def download_all():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    downloaded = 0

    for filename, url, label in CURATED_REPORTS:
        dest = PDF_DIR / filename
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[SKIP] {filename} already exists ({dest.stat().st_size:,} bytes)")
            downloaded += 1
            continue

        print(f"Downloading {label} -> {filename} ...")
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  [OK] Saved {filename} ({size_mb:.2f} MB)")
            downloaded += 1
        except Exception as e:
            print(f"  [ERR] Failed {filename}: {e}")
            if dest.exists():
                dest.unlink()

    print(f"\nDownloaded {downloaded}/{len(CURATED_REPORTS)} PDFs to {PDF_DIR}")


if __name__ == "__main__":
    download_all()

