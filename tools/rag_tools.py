"""
RAG Agent Tools
---------------
Dedicated tools for the RAG Agent:
  - rag_semantic_search    : semantic search via ChromaDB
  - find_eligible_colleges : rank/marks-based structured filter + RAG context
  - get_college_details    : all chunks for a named college

Anti-hallucination:
  - Every response includes raw data from the verified CSV/ChromaDB
  - LLM is instructed to quote only what these tools return
  - "not found" messages are explicit — no guessing
"""
import csv
import json
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from rag.retriever import semantic_search, get_college_chunks, format_context

CSV_PATH = Path(__file__).parent.parent / "tn_engineering_colleges.csv"


def _load_csv():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


_COLLEGE_DATA = None


def _get_data():
    global _COLLEGE_DATA
    if _COLLEGE_DATA is None:
        _COLLEGE_DATA = _load_csv()
    return _COLLEGE_DATA


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: Semantic Search
# ─────────────────────────────────────────────────────────────────────────────

@tool
def rag_semantic_search(
    query: str,
    city: Optional[str] = None,
    ownership: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """
    Semantically search the Tamil Nadu engineering college database.
    Use this for general queries about colleges, facilities, courses, or policies.
    Returns verified facts from the database only — never fabricates.

    Args:
        query: Natural language query (e.g., "colleges with good placement in Chennai")
        city: Optional city filter (e.g., "Chennai", "Coimbatore")
        ownership: Optional ownership filter ("Government", "Private", "Deemed")
        top_k: Number of results to return (default 5)
    """
    hits = semantic_search(query, top_k=top_k, city=city, ownership=ownership)
    context = format_context(hits)

    if not hits:
        return json.dumps({
            "status": "no_results",
            "context": "No matching colleges found in the database for this query.",
            "sources": [],
        })

    return json.dumps({
        "status": "success",
        "num_results": len(hits),
        "context": context,
        "sources": [
            {
                "college": h["metadata"].get("college_name"),
                "branch": h["metadata"].get("branch"),
                "city": h["metadata"].get("city"),
                "relevance_pct": int(h["score"] * 100),
            }
            for h in hits
        ],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: Find Eligible Colleges (structured filter + RAG context)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def find_eligible_colleges(
    twelfth_percent: float = 60.0,
    community: str = "general",
    tnea_rank: Optional[int] = None,
    jee_rank: Optional[int] = None,
    preferred_city: Optional[str] = None,
    preferred_specialization: Optional[str] = None,
    max_annual_fee: Optional[int] = None,
    ownership_preference: Optional[str] = None,
) -> str:
    """
    Find Tamil Nadu engineering colleges where this student is eligible.
    Uses structured cutoff data from the verified CSV database.
    Only returns colleges that genuinely match — never fabricates results.

    Args:
        twelfth_percent: Student's 12th standard percentage
        community: Community category: general/oc/obc/bc/mbc/sc/st/pwd
        tnea_rank: Student's TNEA rank (lower = better)
        jee_rank: Student's JEE Main rank (lower = better)
        preferred_city: City preference (e.g., "Chennai")
        preferred_specialization: Branch name or abbreviation
        max_annual_fee: Max annual fee in INR
        ownership_preference: "Government", "Private", or "Deemed"
    """
    data = _get_data()

    comm = community.lower().strip()
    if comm in ("oc", "general", "gen"):
        cutoff_col, pct_col, label = "cutoff_value_general", "min_12th_percent_general", "General/OC"
    elif comm in ("obc", "bc", "mbc"):
        cutoff_col, pct_col, label = "cutoff_value_obc", "min_12th_percent_obc", "OBC/BC/MBC"
    elif comm == "sc":
        cutoff_col, pct_col, label = "cutoff_value_sc", "min_12th_percent_sc", "SC"
    elif comm == "st":
        cutoff_col, pct_col, label = "cutoff_value_st", "min_12th_percent_st", "ST"
    elif comm == "pwd":
        cutoff_col, pct_col, label = "cutoff_value_pwd", "min_12th_percent_sc", "PWD"
    else:
        cutoff_col, pct_col, label = "cutoff_value_general", "min_12th_percent_general", "General/OC"

    # Normalize branch aliases
    branch_query = (preferred_specialization or "").lower()
    ALIASES = {
        "cse": "computer science", "cs": "computer science",
        "ece": "electronics", "eee": "electrical",
        "mech": "mechanical", "it": "information technology",
        "ai": "artificial intelligence", "aiml": "artificial intelligence",
        "aids": "artificial intelligence", "ds": "data science",
    }
    for alias, expansion in ALIASES.items():
        if branch_query == alias:
            branch_query = expansion
            break

    eligible = []
    seen = set()

    for row in data:
        # Minimum 12th percentage check
        try:
            min_pct = float(row.get(pct_col, 60.0))
        except (ValueError, TypeError):
            min_pct = 60.0
        if twelfth_percent < min_pct:
            continue

        # TNEA rank cutoff
        entrance = row.get("entrance_exam", "")
        if tnea_rank is not None and ("TNEA" in entrance or "Class 12" in entrance):
            try:
                cutoff = int(float(row[cutoff_col]))
                if tnea_rank > cutoff:
                    continue
            except (ValueError, TypeError, KeyError):
                pass

        # JEE rank cutoff
        if jee_rank is not None and "JEE" in entrance:
            try:
                cutoff = int(float(row[cutoff_col]))
                if jee_rank > cutoff:
                    continue
            except (ValueError, TypeError, KeyError):
                pass

        # City filter
        if preferred_city and preferred_city.lower() not in row.get("city", "").lower():
            continue

        # Branch filter (semantic keyword match)
        if branch_query:
            branch_col = row.get("course_specialization", "").lower()
            if not any(w in branch_col for w in branch_query.split() if len(w) > 3):
                if branch_query not in branch_col:
                    continue

        # Fee filter
        if max_annual_fee:
            try:
                if int(float(row.get("fees_annual_inr", 0))) > max_annual_fee:
                    continue
            except (ValueError, TypeError):
                pass

        # Ownership filter
        if ownership_preference and ownership_preference.lower() not in row.get("ownership", "").lower():
            continue

        key = f"{row['college_name']}|{row['course_specialization']}"
        if key in seen:
            continue
        seen.add(key)
        eligible.append(row)

    if not eligible:
        return json.dumps({
            "status": "no_results",
            "message": (
                f"No colleges found in our database matching your criteria "
                f"(12th: {twelfth_percent}%, community: {label}, "
                f"rank: {tnea_rank or jee_rank}). "
                "Try relaxing city/fee/branch filters."
            ),
            "colleges": [],
        })

    # Sort: tier → NIRF rank
    def sort_key(r):
        try:
            return (int(r.get("tier", 3)), int(float(r.get("nirf_rank", 999))))
        except (ValueError, TypeError):
            return (3, 999)

    eligible.sort(key=sort_key)

    # Group branches by college
    groups: dict = {}
    for r in eligible[:80]:
        cn = r["college_name"]
        if cn not in groups:
            try:
                fee_fmt = f"₹{int(float(r.get('fees_annual_inr', 0))):,}"
            except Exception:
                fee_fmt = r.get("fees_annual_inr", "N/A")
            groups[cn] = {
                "college_name": cn,
                "city": r["city"],
                "ownership": r["ownership"],
                "nirf_rank": r.get("nirf_rank", "N/A"),
                "accreditation": r.get("accreditation", "N/A"),
                "fees_annual": fee_fmt,
                "placement_avg_lpa": r.get("placement_avg_lpa", "N/A"),
                "entrance_exam": r.get("entrance_exam", "N/A"),
                "hostel": r.get("hostel_available", "N/A"),
                "tier": r.get("tier", "N/A"),
                f"cutoff_{label}": r.get(cutoff_col, "N/A"),
                "branches": [],
            }
        groups[cn]["branches"].append(r["course_specialization"])

    colleges_out = list(groups.values())[:20]

    return json.dumps({
        "status": "success",
        "total_eligible": len(groups),
        "showing": len(colleges_out),
        "community_used": label,
        "data_source": "Verified TN Engineering College Database (501 records, ~100 colleges)",
        "colleges": colleges_out,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: Get Full College Details via RAG
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_college_details(college_name: str, branch: Optional[str] = None) -> str:
    """
    Get comprehensive details about a specific college including admission process,
    deadlines, documents, fees, cutoffs, placements, and facilities.
    Data is retrieved from the verified database — no hallucination.

    Args:
        college_name: College name (full, partial, or abbreviation e.g. "VIT", "NIT Trichy")
        branch: Optional branch to focus on (e.g., "Computer Science")
    """
    # RAG: retrieve all chunks for this college
    hits = get_college_chunks(college_name)

    if not hits:
        # Fallback: semantic search with college name as query
        hits = semantic_search(college_name, top_k=8, college_name=college_name)

    if not hits:
        return json.dumps({
            "status": "not_found",
            "message": (
                f"'{college_name}' was not found in our verified database of "
                "Tamil Nadu engineering colleges. Please check the spelling or try "
                "a shorter name (e.g., 'VIT' instead of 'VIT Vellore')."
            ),
        })

    # Filter by branch if specified
    if branch:
        branch_lower = branch.lower()
        branch_hits = [
            h for h in hits
            if any(w in h["metadata"].get("branch", "").lower()
                   for w in branch_lower.split() if len(w) > 3)
        ]
        if branch_hits:
            hits = branch_hits

    context = format_context(hits, max_chars=8000)
    unique_branches = list({h["metadata"].get("branch", "") for h in hits})
    college_display = hits[0]["metadata"].get("college_name", college_name)

    return json.dumps({
        "status": "success",
        "college_name": college_display,
        "branches_found": unique_branches,
        "num_chunks": len(hits),
        "context": context,
        "data_source": "Verified TN Engineering College Database",
    })
