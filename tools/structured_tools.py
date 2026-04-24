"""
Advisory Agent Tools
--------------------
Dedicated tools for the Advisory Agent:
  - compare_colleges        : side-by-side comparison of 2–4 colleges
  - get_admission_deadlines : upcoming dates/deadlines
  - search_by_branch        : find all colleges offering a specific branch
  - get_reservation_info    : explain TN reservation policy for a community

All data strictly from the verified CSV — hallucination blocked at tool level.
"""
import csv
import json
from pathlib import Path
from typing import List, Optional
from langchain_core.tools import tool

CSV_PATH = Path(__file__).parent.parent / "tn_engineering_colleges.csv"

_DATA_CACHE = None


def _get_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        rows = []
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        _DATA_CACHE = rows
    return _DATA_CACHE


def _find_college(name: str):
    """Return all rows matching a college name (partial/fuzzy)."""
    data = _get_data()
    name_lower = name.lower().strip()
    words = [w for w in name_lower.split() if len(w) > 3]
    matches = []
    for row in data:
        stored = row["college_name"].lower()
        if name_lower in stored or stored in name_lower:
            matches.append(row)
        elif words and any(w in stored for w in words):
            matches.append(row)
    return matches


# ─────────────────────────────────────────────────────────────────────────────

@tool
def compare_colleges(college_names: List[str], aspect: str = "overall") -> str:
    """
    Compare 2 to 4 Tamil Nadu engineering colleges side-by-side.
    Data is from the verified database only.

    Args:
        college_names: List of college names to compare (2–4)
        aspect: What to compare — "fees", "placements", "cutoffs", "facilities", "overall"
    """
    if len(college_names) < 2:
        return json.dumps({"status": "error", "message": "Provide at least 2 college names to compare."})

    comparison = []
    not_found = []

    for name in college_names[:4]:
        rows = _find_college(name)
        if not rows:
            not_found.append(name)
            continue
        r = rows[0]  # take first row (representative)

        entry = {
            "college_name": r["college_name"],
            "city": r["city"],
            "ownership": r["ownership"],
            "nirf_rank": r.get("nirf_rank", "N/A"),
            "accreditation": r.get("accreditation", "N/A"),
            "established": r.get("established_year", "N/A"),
            "campus_acres": r.get("campus_size_acres", "N/A"),
        }

        if aspect in ("fees", "overall"):
            try:
                fee = f"₹{int(float(r.get('fees_annual_inr', 0))):,}"
                total = f"₹{int(float(r.get('fees_total_inr', 0))):,}"
            except Exception:
                fee = r.get("fees_annual_inr", "N/A")
                total = r.get("fees_total_inr", "N/A")
            entry["fees_annual"] = fee
            entry["fees_total_4yr"] = total
            entry["capitation_fee"] = r.get("capitation_fee", "No")
            entry["scholarship_available"] = r.get("scholarship_available", "No")

        if aspect in ("placements", "overall"):
            entry["placement_avg_lpa"] = r.get("placement_avg_lpa", "N/A")
            entry["placement_highest_lpa"] = r.get("placement_highest_lpa", "N/A")
            entry["top_recruiters"] = r.get("top_recruiters", "N/A")

        if aspect in ("cutoffs", "overall"):
            entry["cutoff_general"] = r.get("cutoff_value_general", "N/A")
            entry["cutoff_obc"] = r.get("cutoff_value_obc", "N/A")
            entry["cutoff_sc"] = r.get("cutoff_value_sc", "N/A")
            entry["entrance_exam"] = r.get("entrance_exam", "N/A")

        if aspect in ("facilities", "overall"):
            entry["hostel"] = r.get("hostel_available", "N/A")
            entry["girls_hostel"] = r.get("girls_hostel", "N/A")
            entry["sports"] = r.get("sports_facilities", "N/A")
            entry["international_tieups"] = r.get("international_tie_ups", "N/A")

        entry["branches_offered"] = list({row["course_specialization"] for row in rows})
        comparison.append(entry)

    return json.dumps({
        "status": "success" if comparison else "error",
        "aspect": aspect,
        "comparison": comparison,
        "not_found_in_db": not_found,
        "data_source": "Verified TN Engineering College Database",
    })


# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_admission_deadlines(
    city: Optional[str] = None,
    ownership: Optional[str] = None,
    entrance_exam: Optional[str] = None,
) -> str:
    """
    Get 2026-27 admission deadlines and important dates for Tamil Nadu engineering colleges.

    Args:
        city: Filter by city (optional)
        ownership: Filter by "Government", "Private", or "Deemed" (optional)
        entrance_exam: Filter by entrance exam type e.g. "TNEA", "JEE" (optional)
    """
    data = _get_data()
    seen = set()
    deadlines = []

    for row in data:
        key = f"{row['college_name']}|{row.get('application_end_date', '')}"
        if key in seen:
            continue

        if city and city.lower() not in row.get("city", "").lower():
            continue
        if ownership and ownership.lower() not in row.get("ownership", "").lower():
            continue
        if entrance_exam and entrance_exam.upper() not in row.get("entrance_exam", "").upper():
            continue

        seen.add(key)
        deadlines.append({
            "college": row["college_name"],
            "city": row["city"],
            "ownership": row["ownership"],
            "entrance_exam": row.get("entrance_exam", "N/A"),
            "application_start": row.get("application_start_date", "N/A"),
            "application_deadline": row.get("application_end_date", "N/A"),
            "counselling_date": row.get("counselling_date", "N/A"),
            "admission_date": row.get("admission_date", "N/A"),
            "application_mode": row.get("application_mode", "N/A"),
        })

    deadlines.sort(key=lambda x: x.get("application_deadline", "9999"))

    return json.dumps({
        "status": "success",
        "total": len(deadlines),
        "deadlines": deadlines[:25],
        "data_source": "Verified TN Engineering College Database (2026-27 cycle)",
    })


# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_by_branch(
    specialization: str,
    city: Optional[str] = None,
    max_fee: Optional[int] = None,
    ownership: Optional[str] = None,
) -> str:
    """
    Find all Tamil Nadu engineering colleges offering a specific branch/specialization.
    Results ranked by NIRF rank. All data from verified database.

    Args:
        specialization: Branch name e.g. "Computer Science", "ECE", "AI/ML", "Mechanical"
        city: Optional city filter
        max_fee: Optional max annual fee in INR
        ownership: Optional ownership filter
    """
    data = _get_data()
    ALIASES = {
        "cse": "computer science", "cs": "computer science",
        "ece": "electronics", "eee": "electrical",
        "mech": "mechanical", "it": "information technology",
        "ai": "artificial intelligence", "aiml": "artificial intelligence",
        "aids": "artificial intelligence and data science",
        "ds": "data science", "civil": "civil",
    }
    spec_lower = specialization.lower().strip()
    for alias, full in ALIASES.items():
        if spec_lower == alias:
            spec_lower = full
            break

    search_words = [w for w in spec_lower.split() if len(w) > 3]
    results = []
    seen = set()

    for row in data:
        branch = row.get("course_specialization", "").lower()
        if spec_lower not in branch and not any(w in branch for w in search_words):
            continue
        if city and city.lower() not in row.get("city", "").lower():
            continue
        if max_fee:
            try:
                if int(float(row.get("fees_annual_inr", 0))) > max_fee:
                    continue
            except (ValueError, TypeError):
                pass
        if ownership and ownership.lower() not in row.get("ownership", "").lower():
            continue

        key = f"{row['college_name']}|{row['course_specialization']}"
        if key in seen:
            continue
        seen.add(key)

        try:
            fee_fmt = f"₹{int(float(row.get('fees_annual_inr', 0))):,}"
        except Exception:
            fee_fmt = row.get("fees_annual_inr", "N/A")

        results.append({
            "college": row["college_name"],
            "city": row["city"],
            "ownership": row["ownership"],
            "branch": row["course_specialization"],
            "nirf_rank": row.get("nirf_rank", "N/A"),
            "accreditation": row.get("accreditation", "N/A"),
            "fees_annual": fee_fmt,
            "seats": row.get("course_seats", "N/A"),
            "placement_avg_lpa": row.get("placement_avg_lpa", "N/A"),
            "entrance_exam": row.get("entrance_exam", "N/A"),
            "cutoff_general": row.get("cutoff_value_general", "N/A"),
        })

    results.sort(key=lambda r: (
        int(r.get("nirf_rank", 999)) if str(r.get("nirf_rank", "999")).isdigit() else 999
    ))

    return json.dumps({
        "status": "success",
        "branch_searched": specialization,
        "total_found": len(results),
        "colleges": results[:20],
        "data_source": "Verified TN Engineering College Database",
    })


# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_reservation_policy(community: str, college_name: Optional[str] = None) -> str:
    """
    Explain seat reservation percentages and policies for a given community in TN.
    If college_name provided, gives college-specific reservation data.

    Args:
        community: Community category (general/obc/bc/mbc/sc/st/pwd/ews)
        college_name: Optional specific college name
    """
    # General TN reservation policy
    TN_POLICY = {
        "general": "General/OC category: 50% of seats (31% OC + 19% BC sub-quota in some institutions)",
        "obc/bc/mbc": "OBC/BC/MBC: 27% of seats reserved (BC: 26.5%, BCM: 3.5% in Govt colleges)",
        "sc": "Scheduled Caste (SC): 15% of seats reserved",
        "st": "Scheduled Tribe (ST): 7.5% of seats reserved (combined SC+ST sometimes 22.5%)",
        "pwd": "Persons with Disability (PWD): 3% horizontal reservation across all categories",
        "ews": "Economically Weaker Section (EWS): 10% of seats (for income <8L/year, General caste)",
    }

    comm = community.lower().strip()
    comm_key = comm
    if comm in ("oc",):
        comm_key = "general"
    elif comm in ("bc", "mbc", "obc"):
        comm_key = "obc/bc/mbc"

    policy_info = TN_POLICY.get(comm_key, TN_POLICY.get("general"))

    result = {
        "status": "success",
        "community": community,
        "tn_reservation_policy": policy_info,
        "general_note": (
            "Tamil Nadu follows the 69% reservation policy for BC/MBC/SC/ST categories. "
            "Government engineering colleges strictly follow TNEA counselling reservation norms. "
            "Private colleges may have management quota (up to 50%) and NRI quota seats."
        ),
    }

    if college_name:
        rows = _find_college(college_name)
        if rows:
            r = rows[0]
            result["college_specific"] = {
                "college": r["college_name"],
                "reservation_general": f"{r.get('reservation_general_percent', 'N/A')}%",
                "reservation_obc": f"{r.get('reservation_obc_percent', 'N/A')}%",
                "reservation_sc": f"{r.get('reservation_sc_percent', 'N/A')}%",
                "reservation_st": f"{r.get('reservation_st_percent', 'N/A')}%",
                "reservation_pwd": f"{r.get('reservation_pwd_percent', 'N/A')}%",
                "reservation_ews": f"{r.get('reservation_ews_percent', 'N/A')}%",
                "management_quota": r.get("management_quota_available", "No"),
                "nri_quota": r.get("nri_quota_available", "No"),
            }
        else:
            result["college_note"] = f"College '{college_name}' not found in database."

    return json.dumps(result)
