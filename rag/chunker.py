"""
Chunker: Converts each CSV row into a rich contextual narrative chunk.

Strategy: One chunk per college×branch with labeled sections.
This lets sentence-transformers embed full context including fees,
cutoffs, eligibility, deadlines — enabling semantic retrieval.
"""
import csv
from pathlib import Path
from typing import List, Tuple, Dict

CSV_PATH = Path(__file__).parent.parent / "tn_engineering_colleges.csv"


def _safe_fee(val) -> str:
    try:
        return f"₹{int(float(val)):,}"
    except Exception:
        return str(val) if val else "N/A"


def format_chunk(row: Dict) -> str:
    """Build a structured narrative chunk from a single CSV row."""
    return f"""[COLLEGE PROFILE]
College: {row.get('college_name', 'N/A')} | City: {row.get('city', 'N/A')}, Tamil Nadu | Ownership: {row.get('ownership', 'N/A')} | Tier: {row.get('tier', 'N/A')}
Full Name: {row.get('college_full_name', row.get('college_name', 'N/A'))}
NAAC: {row.get('accreditation', 'N/A')} | NIRF Rank: {row.get('nirf_rank', 'N/A')} | Established: {row.get('established_year', 'N/A')}
Website: {row.get('website', 'N/A')} | Email: {row.get('contact_email', 'N/A')} | Phone: {row.get('contact_phone', 'N/A')}
Campus: {row.get('campus_size_acres', 'N/A')} acres | Total Seats: {row.get('total_seats', 'N/A')}

[BRANCH & COURSE]
Branch: {row.get('course_specialization', 'N/A')} | Degree: {row.get('degree_type', 'B.Tech')} ({row.get('duration_years', '4')} years)
Available Seats: {row.get('course_seats', 'N/A')} | Entrance Exam: {row.get('entrance_exam', 'N/A')}
Lateral Entry: {row.get('lateral_entry_available', 'No')}

[CUTOFFS & ELIGIBILITY]
Cutoff Type: {row.get('cutoff_type', 'N/A')}
Cutoff General/OC: {row.get('cutoff_value_general', 'N/A')} | OBC/BC/MBC: {row.get('cutoff_value_obc', 'N/A')}
Cutoff SC: {row.get('cutoff_value_sc', 'N/A')} | ST: {row.get('cutoff_value_st', 'N/A')} | PWD: {row.get('cutoff_value_pwd', 'N/A')}
Min 12th% General: {row.get('min_12th_percent_general', '60')}% | OBC: {row.get('min_12th_percent_obc', '55')}% | SC: {row.get('min_12th_percent_sc', '50')}% | ST: {row.get('min_12th_percent_st', '50')}%
Subjects Required: {row.get('subjects_required_12th', 'Physics, Chemistry, Mathematics (PCM)')}
Eligibility: {row.get('eligibility', 'N/A')}

[FEES & SCHOLARSHIPS]
Annual Fee: {_safe_fee(row.get('fees_annual_inr'))} | Total 4-Year Fee: {_safe_fee(row.get('fees_total_inr'))}
Scholarship Available: {row.get('scholarship_available', 'No')}
Scholarship Details: {row.get('scholarship_details', 'N/A')}
Capitation Fee: {row.get('capitation_fee', 'No')} | Management Quota: {row.get('management_quota_available', 'No')} | NRI Quota: {row.get('nri_quota_available', 'No')}

[SEAT RESERVATION]
General/OC: {row.get('reservation_general_percent', '50')}% | OBC/BC/MBC: {row.get('reservation_obc_percent', '27')}%
SC: {row.get('reservation_sc_percent', '15')}% | ST: {row.get('reservation_st_percent', '7.5')}%
PWD: {row.get('reservation_pwd_percent', '3')}% | EWS: {row.get('reservation_ews_percent', '10')}%

[ADMISSION PROCESS & DEADLINES]
Application Mode: {row.get('application_mode', 'N/A')}
Application Open: {row.get('application_start_date', 'N/A')} | Application Deadline: {row.get('application_end_date', 'N/A')}
Counselling Date: {row.get('counselling_date', 'N/A')} | Admission Date: {row.get('admission_date', 'N/A')}
Admission Steps: {row.get('admission_steps', 'N/A')}

[DOCUMENTS REQUIRED]
{row.get('document_checklist', 'N/A')}

[PLACEMENTS]
Average Package: {row.get('placement_avg_lpa', 'N/A')} LPA | Highest Package: {row.get('placement_highest_lpa', 'N/A')} LPA
Top Recruiters: {row.get('top_recruiters', 'N/A')}
Placement Cell: {row.get('placement_cell', 'Yes')} | International Tie-ups: {row.get('international_tie_ups', 'No')}

[FACILITIES]
Hostel: {row.get('hostel_available', 'No')} | Girls Hostel: {row.get('girls_hostel', 'No')} | Boys Hostel: {row.get('boys_hostel', 'No')}
Sports: {row.get('sports_facilities', 'No')} | Library: {row.get('library', 'Yes')}

[LATEST NEWS]
{row.get('news_updates', 'No recent updates.')}
"""


def create_chunks() -> List[Tuple[str, Dict]]:
    """
    Read CSV and produce (chunk_text, metadata) pairs.
    One chunk per row (college × branch combination).
    """
    chunks = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            text = format_chunk(row)
            metadata = {
                "college_id": str(row.get("college_id", "")),
                "college_name": row.get("college_name", ""),
                "city": row.get("city", ""),
                "ownership": row.get("ownership", ""),
                "tier": str(row.get("tier", "3")),
                "branch": row.get("course_specialization", ""),
                "entrance_exam": row.get("entrance_exam", ""),
                "nirf_rank": str(row.get("nirf_rank", "999")),
                "fees_annual": str(row.get("fees_annual_inr", "0")),
                "accreditation": row.get("accreditation", ""),
                "hostel": row.get("hostel_available", "No"),
            }
            chunks.append((text, metadata))
    return chunks


def get_all_college_names() -> List[str]:
    names = set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.add(row["college_name"])
    return sorted(names)
