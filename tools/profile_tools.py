"""
Profile Agent Tools
-------------------
Dedicated tools for the Profile Agent:
  - extract_student_info   : parse free-text input for rank/marks/community/prefs
  - validate_student_profile: sanity-check ranges & completeness
  - get_missing_fields     : identify what info is still needed

Anti-hallucination: these tools only VALIDATE — they never invent data.
"""
import json
import re
from typing import Optional
from langchain_core.tools import tool


VALID_COMMUNITIES = {"general", "oc", "obc", "bc", "mbc", "sc", "st", "pwd", "ews"}
TN_CITIES = {
    "chennai", "coimbatore", "madurai", "tiruchirappalli", "trichy",
    "vellore", "salem", "thanjavur", "erode", "tirunelveli",
    "kancheepuram", "thiruvallur", "virudhunagar", "sivakasi", "karur",
    "namakkal", "dharmapuri", "krishnagiri", "cuddalore", "villupuram",
}
BRANCH_ALIASES = {
    "cse": "Computer Science and Engineering",
    "cs": "Computer Science and Engineering",
    "ece": "Electronics and Communication Engineering",
    "eee": "Electrical and Electronics Engineering",
    "mech": "Mechanical Engineering",
    "it": "Information Technology",
    "ai": "Artificial Intelligence",
    "aiml": "Artificial Intelligence and Machine Learning",
    "aids": "Artificial Intelligence and Data Science",
    "ds": "Data Science",
    "civil": "Civil Engineering",
    "chem": "Chemical Engineering",
    "bio": "Biotechnology",
}


@tool
def extract_student_info(user_message: str) -> str:
    """
    Parse the student's free-text message to extract academic profile details.
    Looks for TNEA rank, JEE rank, 12th percentage, community, city preference,
    branch preference, and budget from natural language.

    Args:
        user_message: The student's raw message text
    """
    extracted = {}

    # TNEA rank: "rank 15000", "tnea 15000", "my rank is 15000"
    tnea_match = re.search(
        r'(?:tnea\s*rank|rank\s*is|rank[:=\s])\s*(\d{1,6})', user_message, re.IGNORECASE
    )
    if not tnea_match:
        tnea_match = re.search(r'\brank\s+(\d{4,6})\b', user_message, re.IGNORECASE)
    if tnea_match:
        extracted["tnea_rank"] = int(tnea_match.group(1))

    # JEE rank
    jee_match = re.search(
        r'(?:jee\s*(?:main|mains|advanced)?\s*rank[:=\s])\s*(\d{1,6})', user_message, re.IGNORECASE
    )
    if jee_match:
        extracted["jee_rank"] = int(jee_match.group(1))

    # 12th percentage: "85%", "85.5 percent", "scored 85"
    pct_match = re.search(
        r'(\d{2,3}(?:\.\d{1,2})?)\s*(?:%|percent|marks|scored|percentage)', user_message, re.IGNORECASE
    )
    if not pct_match:
        pct_match = re.search(r'(?:marks?|scored?|got)\s+(\d{2,3}(?:\.\d{1,2})?)', user_message, re.IGNORECASE)
    if pct_match:
        pct = float(pct_match.group(1))
        if 30 <= pct <= 100:
            extracted["twelfth_percent"] = pct

    # Community
    text_lower = user_message.lower()
    for comm in VALID_COMMUNITIES:
        pattern = rf'\b{re.escape(comm)}\b'
        if re.search(pattern, text_lower):
            # Normalize
            if comm in ("oc", "general"):
                extracted["community"] = "general"
            elif comm in ("bc", "obc", "mbc"):
                extracted["community"] = comm
            else:
                extracted["community"] = comm
            break

    # City preference
    for city in TN_CITIES:
        if city in text_lower:
            extracted["preferred_city"] = city.capitalize()
            if city == "tiruchirappalli":
                extracted["preferred_city"] = "Tiruchirappalli"
            break
    if "trichy" in text_lower:
        extracted["preferred_city"] = "Tiruchirappalli"

    # Branch preference
    for alias, full in BRANCH_ALIASES.items():
        if re.search(rf'\b{re.escape(alias)}\b', text_lower):
            extracted["preferred_branch"] = full
            break
    # Also check full branch names
    for full_branch in ["computer science", "electronics", "mechanical", "civil",
                        "electrical", "information technology", "artificial intelligence",
                        "data science", "biotechnology", "chemical"]:
        if full_branch in text_lower:
            extracted.setdefault("preferred_branch", full_branch.title())

    # Budget / fee
    budget_match = re.search(
        r'(?:budget|fee|afford|max\s*fee)\s*(?:is|of|:)?\s*(?:₹|rs\.?|inr)?\s*(\d+(?:[,\d]+)?(?:l|lakh|k)?)',
        text_lower
    )
    if budget_match:
        raw = budget_match.group(1).replace(",", "")
        if raw.endswith("l") or "lakh" in text_lower:
            extracted["max_fee"] = int(float(raw.rstrip("l")) * 100000)
        elif raw.endswith("k"):
            extracted["max_fee"] = int(float(raw.rstrip("k")) * 1000)
        else:
            val = int(raw)
            if val < 1000:
                extracted["max_fee"] = val * 1000  # assume thousands
            else:
                extracted["max_fee"] = val

    return json.dumps({
        "status": "success",
        "extracted": extracted,
        "message": (
            f"Extracted {len(extracted)} field(s): {list(extracted.keys())}"
            if extracted
            else "No academic information detected in the message."
        ),
    })


@tool
def validate_student_profile(profile: dict) -> str:
    """
    Validate the accumulated student profile for completeness and correctness.
    Returns which required fields are missing and flags any out-of-range values.

    Args:
        profile: The current student profile dictionary
    """
    issues = []
    warnings = []
    missing = []

    # Required fields
    if not profile.get("tnea_rank") and not profile.get("jee_rank") and not profile.get("twelfth_percent"):
        missing.append("rank or 12th percentage")

    if not profile.get("community"):
        missing.append("community category (General/OBC/BC/MBC/SC/ST/PWD)")

    # Range checks
    tnea = profile.get("tnea_rank")
    if tnea is not None:
        if not (1 <= tnea <= 250000):
            warnings.append(f"TNEA rank {tnea} seems unusual (expected 1–250000)")

    jee = profile.get("jee_rank")
    if jee is not None:
        if not (1 <= jee <= 1000000):
            warnings.append(f"JEE rank {jee} seems unusual")

    pct = profile.get("twelfth_percent")
    if pct is not None:
        if not (30 <= pct <= 100):
            issues.append(f"12th percentage {pct} is invalid (must be 30–100)")

    is_complete = len(missing) == 0 and len(issues) == 0
    ready_to_search = bool(
        (profile.get("tnea_rank") or profile.get("jee_rank") or profile.get("twelfth_percent"))
        and profile.get("community")
    )

    return json.dumps({
        "status": "valid" if not issues else "invalid",
        "is_complete": is_complete,
        "ready_to_search": ready_to_search,
        "missing_fields": missing,
        "warnings": warnings,
        "issues": issues,
        "profile_summary": {
            k: v for k, v in profile.items() if v is not None and v != ""
        },
    })


@tool
def get_missing_fields(profile: dict) -> str:
    """
    Return a friendly prompt asking for the next missing piece of student information.

    Args:
        profile: The current student profile dictionary
    """
    has_rank = profile.get("tnea_rank") or profile.get("jee_rank")
    has_pct = profile.get("twelfth_percent")
    has_community = profile.get("community")

    if not has_rank and not has_pct:
        question = (
            "To find the right colleges for you, I need your academic scores. "
            "Could you share your **TNEA rank** or **JEE Main rank**? "
            "Also, what was your **12th standard percentage**?"
        )
    elif not has_community:
        question = (
            "What is your **community category**? "
            "(General/OC, OBC/BC/MBC, SC, ST, or PWD) — "
            "This helps me find colleges with the right cutoff for you."
        )
    elif not profile.get("preferred_branch"):
        question = (
            "Great! Which **engineering branch** are you interested in? "
            "(e.g., CSE, ECE, Mechanical, Civil, AI/ML, Data Science, IT…)"
        )
    elif not profile.get("preferred_city"):
        question = (
            "Do you have a **preferred city** in Tamil Nadu? "
            "(Chennai, Coimbatore, Madurai, Vellore, Salem, Trichy…) "
            "Or should I search across all of Tamil Nadu?"
        )
    else:
        question = (
            "I have all the information I need! Let me search for eligible colleges for you."
        )

    return json.dumps({"status": "success", "next_question": question})
