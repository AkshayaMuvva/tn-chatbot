"""
Tamil Nadu Engineering College Admission RAG Chatbot
Built with LangGraph + LangChain + Anthropic Claude
"""

import os
import json
import csv
import re
from typing import TypedDict, Annotated, List, Optional, Any
from pathlib import Path

# LangGraph / LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import operator

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────

CSV_PATH = Path(__file__).parent / "tn_engineering_colleges.csv"

def load_college_data() -> List[dict]:
    colleges = []
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            colleges.append(dict(row))
    return colleges

COLLEGE_DATA = load_college_data()

# Build college lookup by name
COLLEGE_BY_NAME: dict[str, List[dict]] = {}
for row in COLLEGE_DATA:
    name = row['college_name'].lower()
    COLLEGE_BY_NAME.setdefault(name, []).append(row)

# All unique college names for fuzzy matching
ALL_COLLEGE_NAMES = list(set(row['college_name'] for row in COLLEGE_DATA))
ALL_SPECIALIZATIONS = list(set(row['course_specialization'] for row in COLLEGE_DATA))
ALL_CITIES = list(set(row['city'] for row in COLLEGE_DATA))

print(f"✅ Loaded {len(COLLEGE_DATA)} records across {len(ALL_COLLEGE_NAMES)} colleges")

# ─────────────────────────────────────────────────────────────
# 2. TOOLS
# ─────────────────────────────────────────────────────────────

@tool
def find_eligible_colleges(
    tnea_rank: Optional[int] = None,
    jee_rank: Optional[int] = None,
    twelfth_percent: float = 60.0,
    community: str = "general",
    preferred_city: Optional[str] = None,
    preferred_specialization: Optional[str] = None,
    max_annual_fee: Optional[int] = None,
    ownership_preference: Optional[str] = None,
) -> str:
    """
    Find colleges where the student is eligible based on their TNEA rank, JEE rank,
    12th percentage, community, preferred city, specialization, and budget.
    
    Args:
        tnea_rank: Student's TNEA rank (lower is better, e.g. 5000 means rank 5000)
        jee_rank: Student's JEE Main rank
        twelfth_percent: 12th standard percentage (e.g. 85.5)
        community: One of 'general', 'obc', 'bc', 'mbc', 'sc', 'st', 'pwd'
        preferred_city: City preference (e.g. 'Chennai', 'Coimbatore')
        preferred_specialization: Branch preference (e.g. 'Computer Science and Engineering')
        max_annual_fee: Maximum annual fee in INR
        ownership_preference: 'Government', 'Private', or 'Deemed'
    """
    # Map community to cutoff column
    community_lower = community.lower()
    if community_lower in ['obc', 'bc', 'mbc']:
        cutoff_col = 'cutoff_value_obc'
        min_pct_col = 'min_12th_percent_obc'
    elif community_lower == 'sc':
        cutoff_col = 'cutoff_value_sc'
        min_pct_col = 'min_12th_percent_sc'
    elif community_lower == 'st':
        cutoff_col = 'cutoff_value_st'
        min_pct_col = 'min_12th_percent_st'
    elif community_lower == 'pwd':
        cutoff_col = 'cutoff_value_pwd'
        min_pct_col = 'min_12th_percent_sc'
    else:
        cutoff_col = 'cutoff_value_general'
        min_pct_col = 'min_12th_percent_general'

    eligible = []
    seen_colleges_branches = set()

    for row in COLLEGE_DATA:
        # Min percentage check
        try:
            min_pct = float(row[min_pct_col])
        except (ValueError, KeyError):
            min_pct = 60.0
        if twelfth_percent < min_pct:
            continue

        # TNEA rank check (lower rank = better)
        if tnea_rank is not None and 'TNEA' in row.get('entrance_exam', ''):
            try:
                cutoff = int(row[cutoff_col])
                if tnea_rank > cutoff:
                    continue
            except (ValueError, KeyError):
                pass

        # JEE rank check
        if jee_rank is not None and 'JEE' in row.get('entrance_exam', ''):
            try:
                cutoff = int(row[cutoff_col])
                if jee_rank > cutoff:
                    continue
            except (ValueError, KeyError):
                pass

        # City filter
        if preferred_city:
            if preferred_city.lower() not in row.get('city', '').lower():
                continue

        # Specialization filter
        if preferred_specialization:
            if preferred_specialization.lower() not in row.get('course_specialization', '').lower():
                continue

        # Fee filter
        if max_annual_fee:
            try:
                fee = int(row.get('fees_annual_inr', 0))
                if fee > max_annual_fee:
                    continue
            except ValueError:
                pass

        # Ownership filter
        if ownership_preference:
            if ownership_preference.lower() not in row.get('ownership', '').lower():
                continue

        # Dedup
        key = f"{row['college_name']}|{row['course_specialization']}"
        if key in seen_colleges_branches:
            continue
        seen_colleges_branches.add(key)

        eligible.append(row)

    if not eligible:
        return json.dumps({
            "status": "no_results",
            "message": "No colleges found matching your criteria. Try relaxing filters like city or fee limit.",
            "colleges": []
        })

    # Sort by tier, then NIRF rank
    eligible.sort(key=lambda r: (int(r.get('tier', 3)), int(r.get('nirf_rank', 999))))

    # Group by college for cleaner output
    college_groups: dict[str, dict] = {}
    for r in eligible[:60]:  # cap at 60 records
        cn = r['college_name']
        if cn not in college_groups:
            college_groups[cn] = {
                'college_name': cn,
                'city': r['city'],
                'ownership': r['ownership'],
                'nirf_rank': r['nirf_rank'],
                'accreditation': r['accreditation'],
                'fees_annual_inr': r['fees_annual_inr'],
                'placement_avg_lpa': r['placement_avg_lpa'],
                'entrance_exam': r['entrance_exam'],
                'hostel_available': r['hostel_available'],
                'specializations': [],
                'tier': r.get('tier', 3),
            }
        college_groups[cn]['specializations'].append(r['course_specialization'])

    colleges_out = list(college_groups.values())[:25]

    return json.dumps({
        "status": "success",
        "total_found": len(college_groups),
        "showing": len(colleges_out),
        "colleges": colleges_out
    })


@tool
def get_college_details(college_name: str, specialization: Optional[str] = None) -> str:
    """
    Get full details about a specific college — admission process, deadlines,
    documents, fees, eligibility, placement, facilities, etc.
    
    Args:
        college_name: Name of the college (can be partial)
        specialization: Optional branch name to get branch-specific cutoffs
    """
    # Fuzzy match college name
    name_lower = college_name.lower()
    matches = []
    for row in COLLEGE_DATA:
        rname = row['college_name'].lower()
        if name_lower in rname or rname in name_lower:
            matches.append(row)
        elif any(word in rname for word in name_lower.split() if len(word) > 3):
            matches.append(row)

    if not matches:
        # Try acronym matching
        words = college_name.upper().split()
        for row in COLLEGE_DATA:
            rwords = row['college_name'].upper().split()
            initials = ''.join(w[0] for w in rwords if w[0].isalpha())
            if college_name.upper() in initials or initials.startswith(college_name.upper()):
                matches.append(row)

    if not matches:
        return json.dumps({
            "status": "not_found",
            "message": f"College '{college_name}' not found. Please try with more specific name.",
        })

    # Filter by specialization if given
    if specialization:
        spec_matches = [r for r in matches if specialization.lower() in r['course_specialization'].lower()]
        if spec_matches:
            matches = spec_matches

    # Pick the first college, group its branches
    first = matches[0]
    college_rows = [r for r in matches if r['college_name'] == first['college_name']]

    # Aggregate all specializations
    specializations = []
    for r in college_rows:
        specializations.append({
            'branch': r['course_specialization'],
            'seats': r['course_seats'],
            'cutoff_general': r['cutoff_value_general'],
            'cutoff_obc': r['cutoff_value_obc'],
            'cutoff_sc': r['cutoff_value_sc'],
            'cutoff_st': r['cutoff_value_st'],
        })

    detail = {
        "status": "success",
        "college_name": first['college_name'],
        "college_full_name": first.get('college_full_name', first['college_name']),
        "city": first['city'],
        "state": "Tamil Nadu",
        "ownership": first['ownership'],
        "nirf_rank": first['nirf_rank'],
        "established_year": first['established_year'],
        "accreditation": first['accreditation'],
        "website": first['website'],
        "contact_email": first['contact_email'],
        "contact_phone": first['contact_phone'],
        "campus_size_acres": first['campus_size_acres'],
        "total_seats": first['total_seats'],
        # Courses offered
        "branches_offered": specializations,
        # Fees
        "fees_annual_inr": first['fees_annual_inr'],
        "fees_total_4yr_inr": first['fees_total_inr'],
        # Eligibility
        "eligibility": first['eligibility'],
        "subjects_required": first['subjects_required_12th'],
        "entrance_exam": first['entrance_exam'],
        # Admission Process
        "admission_steps": first['admission_steps'],
        "application_mode": first['application_mode'],
        "application_start_date": first['application_start_date'],
        "application_end_date": first['application_end_date'],
        "counselling_date": first['counselling_date'],
        "admission_date": first['admission_date'],
        # Documents
        "document_checklist": first['document_checklist'],
        # Reservations
        "reservation_general": f"{first['reservation_general_percent']}%",
        "reservation_obc": f"{first['reservation_obc_percent']}%",
        "reservation_sc": f"{first['reservation_sc_percent']}%",
        "reservation_st": f"{first['reservation_st_percent']}%",
        "reservation_pwd": f"{first['reservation_pwd_percent']}%",
        "reservation_ews": f"{first['reservation_ews_percent']}%",
        # Facilities
        "hostel": first['hostel_available'],
        "girls_hostel": first['girls_hostel'],
        "boys_hostel": first['boys_hostel'],
        "sports_facilities": first['sports_facilities'],
        "library": first['library'],
        # Placement
        "placement_avg_lpa": first['placement_avg_lpa'],
        "placement_highest_lpa": first['placement_highest_lpa'],
        "top_recruiters": first['top_recruiters'],
        # Scholarships
        "scholarship_available": first['scholarship_available'],
        "scholarship_details": first['scholarship_details'],
        # Quotas
        "management_quota": first['management_quota_available'],
        "nri_quota": first['nri_quota_available'],
        "capitation_fee": first['capitation_fee'],
        # News
        "latest_news": first.get('news_updates', ''),
        "lateral_entry": first['lateral_entry_available'],
    }

    return json.dumps(detail)


@tool
def compare_colleges(college_names: List[str], aspect: str = "overall") -> str:
    """
    Compare 2-4 colleges side by side on fees, placements, cutoffs, facilities.
    
    Args:
        college_names: List of college names to compare
        aspect: What to compare - 'fees', 'placements', 'cutoffs', 'facilities', 'overall'
    """
    results = []
    for cname in college_names[:4]:
        name_lower = cname.lower()
        match = None
        for row in COLLEGE_DATA:
            if name_lower in row['college_name'].lower():
                match = row
                break
        if match:
            results.append(match)

    if len(results) < 2:
        return json.dumps({"status": "error", "message": "Need at least 2 valid college names to compare."})

    comparison = []
    for r in results:
        comparison.append({
            "college": r['college_name'],
            "city": r['city'],
            "ownership": r['ownership'],
            "nirf_rank": r['nirf_rank'],
            "fees_annual": r['fees_annual_inr'],
            "placement_avg_lpa": r['placement_avg_lpa'],
            "placement_highest": r['placement_highest_lpa'],
            "cutoff_general": r['cutoff_value_general'],
            "accreditation": r['accreditation'],
            "hostel": r['hostel_available'],
            "campus_acres": r['campus_size_acres'],
            "entrance_exam": r['entrance_exam'],
        })

    return json.dumps({"status": "success", "comparison": comparison})


@tool
def get_admission_deadlines(city: Optional[str] = None, ownership: Optional[str] = None) -> str:
    """
    Get upcoming admission deadlines and important dates for 2026-27 admissions.
    
    Args:
        city: Filter by city name (optional)
        ownership: Filter by 'Government', 'Private', or 'Deemed' (optional)
    """
    seen = set()
    deadlines = []

    for row in COLLEGE_DATA:
        key = f"{row['college_name']}|{row['application_end_date']}"
        if key in seen:
            continue
        if city and city.lower() not in row.get('city', '').lower():
            continue
        if ownership and ownership.lower() not in row.get('ownership', '').lower():
            continue
        seen.add(key)
        deadlines.append({
            "college": row['college_name'],
            "city": row['city'],
            "ownership": row['ownership'],
            "application_start": row['application_start_date'],
            "application_deadline": row['application_end_date'],
            "counselling_date": row['counselling_date'],
            "admission_date": row['admission_date'],
            "application_mode": row['application_mode'],
            "entrance_exam": row['entrance_exam'],
        })

    # Sort by deadline
    try:
        deadlines.sort(key=lambda x: x['application_deadline'])
    except Exception:
        pass

    return json.dumps({
        "status": "success",
        "total": len(deadlines),
        "deadlines": deadlines[:20]
    })


@tool
def search_by_branch_and_city(
    specialization: str,
    city: Optional[str] = None,
    max_fee: Optional[int] = None,
) -> str:
    """
    Search for all colleges offering a specific branch/specialization in a city.
    
    Args:
        specialization: Branch name like 'Computer Science', 'Mechanical', 'ECE', etc.
        city: Optional city name
        max_fee: Optional max annual fee in INR
    """
    results = []
    seen = set()
    spec_lower = specialization.lower()
    
    # Expand abbreviations
    abbrev_map = {
        'cse': 'computer science', 'cs': 'computer science',
        'ece': 'electronics and communication', 'eee': 'electrical and electronics',
        'mech': 'mechanical', 'civil': 'civil',
        'it': 'information technology', 'ai': 'artificial intelligence',
        'ds': 'data science', 'aids': 'artificial intelligence',
    }
    for abbr, full in abbrev_map.items():
        if spec_lower == abbr or spec_lower.startswith(abbr + ' '):
            spec_lower = full
            break

    for row in COLLEGE_DATA:
        if spec_lower not in row['course_specialization'].lower():
            continue
        if city and city.lower() not in row['city'].lower():
            continue
        if max_fee:
            try:
                if int(row['fees_annual_inr']) > max_fee:
                    continue
            except ValueError:
                pass
        key = f"{row['college_name']}|{row['course_specialization']}"
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "college": row['college_name'],
            "city": row['city'],
            "branch": row['course_specialization'],
            "ownership": row['ownership'],
            "nirf_rank": row['nirf_rank'],
            "fees_annual": row['fees_annual_inr'],
            "placement_avg_lpa": row['placement_avg_lpa'],
            "seats": row['course_seats'],
            "entrance_exam": row['entrance_exam'],
            "cutoff_general": row['cutoff_value_general'],
        })

    results.sort(key=lambda r: (int(r.get('nirf_rank', 999))))

    return json.dumps({
        "status": "success",
        "branch": specialization,
        "total": len(results),
        "colleges": results[:20]
    })


# ─────────────────────────────────────────────────────────────
# 3. LLM SETUP
# ─────────────────────────────────────────────────────────────

TOOLS = [
    find_eligible_colleges,
    get_college_details,
    compare_colleges,
    get_admission_deadlines,
    search_by_branch_and_city,
]

LLM = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=4096)
LLM_WITH_TOOLS = LLM.bind_tools(TOOLS)

TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are Admissions Buddy — a friendly, knowledgeable college admissions counsellor for Tamil Nadu engineering colleges. You help students find the best engineering colleges for 2026-27 admissions.

Your personality:
- Warm, encouraging, and practical
- Always address the student's actual need
- Give specific, actionable information
- Use emojis sparingly but effectively 🎓

Your capabilities (via tools):
1. find_eligible_colleges — match student rank/marks to eligible colleges
2. get_college_details — full details: admission process, deadlines, documents, eligibility, fees, placements
3. compare_colleges — side-by-side comparison of colleges
4. get_admission_deadlines — important dates and deadlines
5. search_by_branch_and_city — find all colleges for a specific branch

Conversation flow:
- First, understand the student's situation: TNEA rank / JEE rank / 12th marks, community, city preference, branch preference
- Then call find_eligible_colleges to show matching colleges
- When they ask about a specific college, call get_college_details
- Always present information clearly with key facts highlighted
- Proactively mention important deadlines and next steps
- For government colleges, explain TNEA counselling process
- For private/deemed, explain direct application process

Data notes:
- TNEA rank: lower number = better (rank 1 is top). Cutoff rank means last rank admitted.
- JEE ranks work similarly.
- Community categories: General, OBC/BC/MBC, SC, ST, PWD
- TNEA is Tamil Nadu Engineering Admissions — the state-level counselling for government-affiliated engineering colleges
- 2026-27 admissions cycle is currently active

Always be specific about dates, fees, and steps. If you don't have specific information, say so clearly."""


# ─────────────────────────────────────────────────────────────
# 4. LANGGRAPH STATE + NODES
# ─────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    student_profile: dict  # accumulated student info


def chatbot_node(state: ChatState) -> ChatState:
    """Main LLM reasoning node."""
    messages = state["messages"]
    
    # Inject system + student profile context
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    
    # Add profile context if available
    profile = state.get("student_profile", {})
    if profile:
        profile_ctx = f"\n\n[Student Profile so far: {json.dumps(profile)}]"
        system_msg = SystemMessage(content=SYSTEM_PROMPT + profile_ctx)
    
    response = LLM_WITH_TOOLS.invoke([system_msg] + messages)
    return {"messages": [response]}


def tool_node(state: ChatState) -> ChatState:
    """Execute tool calls and return results."""
    messages = state["messages"]
    last_msg = messages[-1]
    
    tool_results = []
    updated_profile = dict(state.get("student_profile", {}))
    
    for tool_call in last_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Update student profile from find_eligible_colleges args
        if tool_name == "find_eligible_colleges":
            for key in ["tnea_rank", "jee_rank", "twelfth_percent", "community",
                        "preferred_city", "preferred_specialization"]:
                if key in tool_args and tool_args[key] is not None:
                    updated_profile[key] = tool_args[key]
        
        if tool_name in TOOL_MAP:
            try:
                result = TOOL_MAP[tool_name].invoke(tool_args)
            except Exception as e:
                result = json.dumps({"status": "error", "message": str(e)})
        else:
            result = json.dumps({"status": "error", "message": f"Unknown tool: {tool_name}"})
        
        tool_results.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )
    
    return {"messages": tool_results, "student_profile": updated_profile}


def should_continue(state: ChatState) -> str:
    """Route: if last message has tool calls → execute tools, else → end."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# ─────────────────────────────────────────────────────────────
# 5. BUILD GRAPH
# ─────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", tool_node)
    builder.set_entry_point("chatbot")
    builder.add_conditional_edges("chatbot", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "chatbot")
    return builder.compile()


GRAPH = build_graph()


# ─────────────────────────────────────────────────────────────
# 6. CHAT INTERFACE
# ─────────────────────────────────────────────────────────────

def chat(query: str, history: list, profile: dict) -> tuple[str, list, dict]:
    """
    Process a user message and return (response, updated_history, updated_profile).
    """
    history.append(HumanMessage(content=query))
    
    state = {
        "messages": history,
        "student_profile": profile,
    }
    
    result = GRAPH.invoke(state)
    
    # Extract AI response
    new_messages = result["messages"]
    ai_response = ""
    for msg in reversed(new_messages):
        if isinstance(msg, AIMessage) and msg.content:
            ai_response = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    
    updated_history = new_messages
    updated_profile = result.get("student_profile", profile)
    
    return ai_response, updated_history, updated_profile


# ─────────────────────────────────────────────────────────────
# 7. MAIN CLI
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("🎓 Tamil Nadu Engineering College Admissions Chatbot")
    print("   Powered by LangGraph + Claude AI")
    print("="*60)
    print("\nHello! I'm your college admissions buddy.")
    print("Tell me about yourself — your TNEA/JEE rank, 12th percentage,")
    print("community, preferred city and branch — and I'll help you find")
    print("the best engineering colleges in Tamil Nadu! 🏛️")
    print("\nType 'quit' to exit.\n")

    history = []
    profile = {}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Best of luck with your admissions! 🎓")
            break

        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye! Best of luck with your admissions! 🎓")
            break

        print("Bot: ", end="", flush=True)
        response, history, profile = chat(user_input, history, profile)
        print(response)
        print()


if __name__ == "__main__":
    main()
