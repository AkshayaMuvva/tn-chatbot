"""
LangGraph Supervisor (Simplified, low-API-call design)
------------------------------------------------------
Goals:
- Minimize Groq API pressure by avoiding multi-pass agent loops
- Use deterministic routing/tool selection whenever possible
- Keep responses grounded strictly in verified tool outputs

Per user turn (typical):
- 0 LLM calls for profile-only turns
- 1 LLM call for RAG/advisory summarization turns
"""

import json
import os
import re
import hashlib
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from graph.state import ChatState
from tools.profile_tools import extract_student_info, get_missing_fields, validate_student_profile
from tools.rag_tools import find_eligible_colleges, get_college_details, rag_semantic_search
from tools.structured_tools import (
    compare_colleges,
    get_admission_deadlines,
    get_reservation_policy,
    search_by_branch,
)

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
CACHE_TTL_SECONDS = int(os.getenv("CHAT_RESPONSE_CACHE_TTL_SECONDS", "45"))
MAX_CACHE_SIZE = int(os.getenv("CHAT_RESPONSE_CACHE_MAX_SIZE", "256"))

TN_CITIES = {
    "chennai",
    "coimbatore",
    "madurai",
    "tiruchirappalli",
    "trichy",
    "vellore",
    "salem",
    "thanjavur",
    "erode",
    "tirunelveli",
    "kancheepuram",
    "thiruvallur",
    "virudhunagar",
    "sivakasi",
    "karur",
    "namakkal",
    "dharmapuri",
    "krishnagiri",
    "cuddalore",
    "villupuram",
}

COMMUNITIES = ["general", "oc", "obc", "bc", "mbc", "sc", "st", "pwd", "ews"]

BRANCH_ALIASES = {
    "cse": "Computer Science",
    "cs": "Computer Science",
    "ece": "Electronics and Communication",
    "eee": "Electrical and Electronics",
    "mech": "Mechanical",
    "civil": "Civil",
    "it": "Information Technology",
    "ai": "Artificial Intelligence",
    "aiml": "Artificial Intelligence and Machine Learning",
    "aids": "Artificial Intelligence and Data Science",
    "ds": "Data Science",
}

ADVISORY_SYSTEM = """You are a TN engineering admissions advisor.
Use ONLY the given verified tool output JSON.
Do not invent colleges, fees, cutoffs, deadlines, or rankings.
If data is missing, say so clearly.
Keep answer concise, structured, and practical.
"""

RAG_SYSTEM = """You are a TN engineering college search assistant.
Use ONLY the provided verified tool output JSON.
Do not fabricate any facts.
When showing colleges, include city, ownership, fees, and eligibility/cutoff context when available.
Keep output concise and scannable.
"""

_RESPONSE_CACHE: Dict[str, Dict] = {}
_CACHE_LOCK = Lock()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _profile_cache_payload(profile: dict) -> Dict:
    profile = profile or {}
    keys = [
        "tnea_rank",
        "jee_rank",
        "twelfth_percent",
        "community",
        "preferred_city",
        "preferred_branch",
        "max_fee",
        "ownership_preference",
    ]
    return {k: profile.get(k) for k in keys if profile.get(k) is not None}


def _cache_key(user_input: str, profile: dict) -> str:
    payload = {
        "query": _normalize_text(user_input),
        "profile": _profile_cache_payload(profile),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _prune_cache(now: float) -> None:
    expired = [k for k, v in _RESPONSE_CACHE.items() if v.get("expiry", 0) <= now]
    for key in expired:
        _RESPONSE_CACHE.pop(key, None)


def _get_cached_payload(key: str, now: float) -> Optional[Dict]:
    with _CACHE_LOCK:
        _prune_cache(now)
        entry = _RESPONSE_CACHE.get(key)
        if not entry:
            return None
        if entry.get("expiry", 0) <= now:
            _RESPONSE_CACHE.pop(key, None)
            return None
        return {
            "ai_response": entry.get("ai_response", ""),
            "student_profile": dict(entry.get("student_profile") or {}),
            "colleges_shown": list(entry.get("colleges_shown") or []),
            "rag_context": entry.get("rag_context", ""),
        }


def _set_cached_payload(
    key: str,
    now: float,
    ai_response: str,
    student_profile: dict,
    colleges_shown: list,
    rag_context: str,
) -> None:
    with _CACHE_LOCK:
        _prune_cache(now)
        _RESPONSE_CACHE[key] = {
            "expiry": now + max(CACHE_TTL_SECONDS, 1),
            "ai_response": ai_response,
            "student_profile": dict(student_profile or {}),
            "colleges_shown": list(colleges_shown or []),
            "rag_context": rag_context or "",
        }

        if len(_RESPONSE_CACHE) > MAX_CACHE_SIZE:
            oldest_key = next(iter(_RESPONSE_CACHE.keys()), None)
            if oldest_key:
                _RESPONSE_CACHE.pop(oldest_key, None)


def _make_llm() -> ChatGroq:
    return ChatGroq(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=900,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _message_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text") or item.get("output_text") or item.get("content")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts).strip()
    return str(content)


def _last_user_text(state: ChatState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return _message_text(getattr(msg, "content", "")).strip()
    return ""


def _safe_json(text: str) -> Dict:
    try:
        return json.loads(text) if text else {}
    except Exception:
        return {"status": "error", "message": text}


def _recent_conversation(state: ChatState, keep: int = 6) -> List:
    """Keep only recent human/assistant messages to limit token usage."""
    msgs = [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
    return msgs[-keep:]


def _is_smalltalk(text: str) -> bool:
    t = text.lower().strip()
    if not t:
        return True
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "thanks", "thank you"]
    return any(t == g or t.startswith(g + " ") for g in greetings)


def _heuristic_route(state: ChatState) -> str:
    profile = state.get("student_profile", {}) or {}
    text = _last_user_text(state).lower()

    if _is_smalltalk(text):
        return "direct_response"

    advisory_keywords = [
        "compare",
        "comparison",
        "deadline",
        "last date",
        "reservation",
        "quota",
        "policy",
        "which branch",
        "branch colleges",
    ]
    rag_keywords = [
        "eligible",
        "college",
        "colleges",
        "admission",
        "documents",
        "fees",
        "cutoff",
        "placements",
        "find",
        "search",
        "recommend",
        "suggest",
        "show me",
    ]

    if any(k in text for k in advisory_keywords):
        return "advisory_agent"

    asks_for_colleges = any(k in text for k in rag_keywords)
    profile_cues = ["rank", "percent", "%", "community", "obc", "bc", "mbc", "sc", "st", "pwd"]
    shares_profile_details = any(k in text for k in profile_cues)

    has_core = bool(
        (profile.get("tnea_rank") or profile.get("jee_rank") or profile.get("twelfth_percent"))
        and profile.get("community")
    )

    if asks_for_colleges:
        if has_core or shares_profile_details:
            return "rag_agent"
        return "profile_agent"

    if shares_profile_details:
        return "profile_agent"

    return "direct_response"


def _extract_filters(text: str) -> Dict[str, Optional[str]]:
    text_l = text.lower()

    city = None
    for c in TN_CITIES:
        if c in text_l:
            city = "Tiruchirappalli" if c == "trichy" else c.title()
            break

    ownership = None
    for candidate in ["government", "private", "deemed"]:
        if candidate in text_l:
            ownership = candidate.title()
            break

    max_fee = None
    fee_match = re.search(
        r"(?:under|below|max|budget|fee)\s*(?:₹|rs\.?|inr)?\s*(\d+(?:,\d{3})*|\d+)(?:\s*(lakh|l|k))?",
        text_l,
    )
    if fee_match:
        raw = fee_match.group(1).replace(",", "")
        unit = fee_match.group(2)
        try:
            val = int(raw)
            if unit in ("l", "lakh"):
                max_fee = val * 100000
            elif unit == "k":
                max_fee = val * 1000
            else:
                max_fee = val
        except ValueError:
            max_fee = None

    branch = None
    for alias, full in BRANCH_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text_l):
            branch = full
            break

    return {
        "city": city,
        "ownership": ownership,
        "max_fee": max_fee,
        "branch": branch,
    }


def _invoke_tool(tool_fn, args: Dict) -> Dict:
    try:
        return _safe_json(tool_fn.invoke(args))
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _summarize_with_llm(system_prompt: str, user_query: str, tool_output: Dict, state: ChatState) -> str:
    if tool_output.get("status") in {"error", "not_found", "no_results"}:
        return tool_output.get("message") or "I couldn't find matching data in the verified database."

    llm = _make_llm()
    prompt = [
        SystemMessage(content=system_prompt),
        *(_recent_conversation(state, keep=4)),
        HumanMessage(
            content=(
                f"User query: {user_query}\n\n"
                "Verified tool output JSON:\n"
                f"{json.dumps(tool_output, ensure_ascii=False)}\n\n"
                "Create a concise user-facing answer using ONLY this JSON."
            )
        ),
    ]
    response = llm.invoke(prompt)
    text = _message_text(response.content).strip()
    return text or "I found the data, but couldn't format the answer. Please retry once."


def _extract_profile_updates(tool_data: Dict, existing: Dict) -> Dict:
    profile = dict(existing or {})
    extracted = tool_data.get("extracted", {}) if isinstance(tool_data, dict) else {}
    for field in [
        "tnea_rank",
        "jee_rank",
        "twelfth_percent",
        "community",
        "preferred_city",
        "preferred_branch",
        "max_fee",
    ]:
        if extracted.get(field) is not None:
            profile[field] = extracted[field]
    return profile


def _infer_college_names(text: str, colleges_shown: List[str]) -> List[str]:
    text_l = text.lower()
    names = []

    for c in colleges_shown or []:
        if c.lower() in text_l:
            names.append(c)

    quoted = re.findall(r"['\"]([^'\"]{4,80})['\"]", text)
    for q in quoted:
        if any(k in q.lower() for k in ["college", "institute", "university", "nit", "vit", "anna"]):
            names.append(q.strip())

    seen = set()
    deduped = []
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(n)
    return deduped


# ----------------------------- Agent nodes ----------------------------------

def supervisor_node(state: ChatState) -> dict:
    route = _heuristic_route(state)

    if route == "direct_response":
        user_text = _last_user_text(state)
        if _is_smalltalk(user_text):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Happy to help! Share your TNEA/JEE rank, 12th %, community, and preferences "
                            "(branch/city/budget), and I’ll find suitable TN colleges."
                        )
                    )
                ],
                "current_agent": "direct_response",
            }

        return {
            "messages": [
                AIMessage(
                    content=(
                        "I can help with college eligibility, details, comparisons, deadlines, and reservation info. "
                        "Tell me what you want to check, and include rank/marks + community for personalized results."
                    )
                )
            ],
            "current_agent": "direct_response",
        }

    return {"messages": [], "current_agent": route}


def profile_agent_node(state: ChatState) -> dict:
    user_text = _last_user_text(state)
    profile = state.get("student_profile", {}) or {}

    extracted = _invoke_tool(extract_student_info, {"user_message": user_text})
    new_profile = _extract_profile_updates(extracted, profile)

    validation = _invoke_tool(validate_student_profile, {"profile": new_profile})
    missing = _invoke_tool(get_missing_fields, {"profile": new_profile})

    ready = bool(validation.get("ready_to_search"))
    if ready:
        msg = (
            "Great — your profile is complete for personalized search. "
            "I can now find eligible colleges based on your rank/marks and community."
        )
    else:
        msg = missing.get("next_question") or (
            "Please share your TNEA/JEE rank, 12th percentage, and community so I can find eligible colleges."
        )

    return {
        "messages": [AIMessage(content=msg)],
        "student_profile": new_profile,
        "current_agent": "profile_agent",
        "profile_complete": ready,
        "profile_asks": state.get("profile_asks", 0) + 1,
    }


def rag_agent_node(state: ChatState) -> dict:
    user_text = _last_user_text(state)
    profile = state.get("student_profile", {}) or {}
    filters = _extract_filters(user_text)

    user_l = user_text.lower()
    asks_eligibility = any(k in user_l for k in ["eligible", "recommend", "suggest", "find", "show me"]) and (
        "college" in user_l or "colleges" in user_l
    )
    asks_detail = any(k in user_l for k in ["details", "about", "cutoff", "fees", "documents", "admission process"]) and (
        "college" in user_l or "university" in user_l or "institute" in user_l
    )

    colleges_shown = list(state.get("colleges_shown", []))
    rag_context = state.get("rag_context", "")

    if asks_eligibility:
        payload = _invoke_tool(
            find_eligible_colleges,
            {
                "twelfth_percent": float(profile.get("twelfth_percent", 60.0)),
                "community": profile.get("community", "general"),
                "tnea_rank": profile.get("tnea_rank"),
                "jee_rank": profile.get("jee_rank"),
                "preferred_city": profile.get("preferred_city") or filters.get("city"),
                "preferred_specialization": profile.get("preferred_branch") or filters.get("branch"),
                "max_annual_fee": profile.get("max_fee") or filters.get("max_fee"),
                "ownership_preference": filters.get("ownership"),
            },
        )
    elif asks_detail:
        inferred = _infer_college_names(user_text, colleges_shown)
        if inferred:
            payload = _invoke_tool(get_college_details, {"college_name": inferred[0]})
        else:
            payload = {
                "status": "not_found",
                "message": "Please mention the college name to fetch detailed admission information.",
            }
    else:
        payload = _invoke_tool(
            rag_semantic_search,
            {
                "query": user_text,
                "city": filters.get("city"),
                "ownership": filters.get("ownership"),
                "top_k": 3,
            },
        )

    for c in payload.get("colleges", []):
        cname = c.get("college_name")
        if cname and cname not in colleges_shown:
            colleges_shown.append(cname)
    if payload.get("college_name") and payload["college_name"] not in colleges_shown:
        colleges_shown.append(payload["college_name"])
    if payload.get("context"):
        rag_context = payload["context"]

    final_text = _summarize_with_llm(RAG_SYSTEM, user_text, payload, state)

    return {
        "messages": [AIMessage(content=final_text)],
        "current_agent": "rag_agent",
        "colleges_shown": colleges_shown,
        "rag_context": rag_context,
    }


def advisory_agent_node(state: ChatState) -> dict:
    user_text = _last_user_text(state)
    user_l = user_text.lower()
    filters = _extract_filters(user_text)
    colleges_shown = list(state.get("colleges_shown", []))

    if "compare" in user_l or "comparison" in user_l:
        college_names = _infer_college_names(user_text, colleges_shown)
        if len(college_names) < 2:
            payload = {
                "status": "error",
                "message": "Please mention at least two college names to compare.",
            }
        else:
            payload = _invoke_tool(compare_colleges, {"college_names": college_names[:4], "aspect": "overall"})

    elif any(k in user_l for k in ["deadline", "last date", "application date", "counselling date"]):
        payload = _invoke_tool(
            get_admission_deadlines,
            {
                "city": filters.get("city"),
                "ownership": filters.get("ownership"),
            },
        )

    elif any(k in user_l for k in ["reservation", "quota", "community policy"]):
        community = "general"
        for c in COMMUNITIES:
            if re.search(rf"\b{re.escape(c)}\b", user_l):
                community = c
                break
        inferred = _infer_college_names(user_text, colleges_shown)
        payload = _invoke_tool(
            get_reservation_policy,
            {"community": community, "college_name": inferred[0] if inferred else None},
        )

    else:
        branch = filters.get("branch")
        if not branch:
            for alias, full in BRANCH_ALIASES.items():
                if re.search(rf"\b{re.escape(alias)}\b", user_l):
                    branch = full
                    break
        if not branch:
            payload = {
                "status": "error",
                "message": "Please mention the branch (for example: CSE, ECE, Mechanical) to search colleges.",
            }
        else:
            payload = _invoke_tool(
                search_by_branch,
                {
                    "specialization": branch,
                    "city": filters.get("city"),
                    "max_fee": filters.get("max_fee"),
                    "ownership": filters.get("ownership"),
                },
            )

    final_text = _summarize_with_llm(ADVISORY_SYSTEM, user_text, payload, state)

    return {
        "messages": [AIMessage(content=final_text)],
        "current_agent": "advisory_agent",
    }


# ----------------------------- Routing --------------------------------------

def route_supervisor(state: ChatState) -> Literal["profile_agent", "rag_agent", "advisory_agent", "__end__"]:
    agent = str(state.get("current_agent", "")).strip().lower()
    if "profile" in agent:
        return "profile_agent"
    if "rag" in agent or "college" in agent:
        return "rag_agent"
    if "advisory" in agent or "compare" in agent or "deadline" in agent:
        return "advisory_agent"
    return "__end__"


def build_graph():
    builder = StateGraph(ChatState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("profile_agent", profile_agent_node)
    builder.add_node("rag_agent", rag_agent_node)
    builder.add_node("advisory_agent", advisory_agent_node)

    builder.set_entry_point("supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "profile_agent": "profile_agent",
            "rag_agent": "rag_agent",
            "advisory_agent": "advisory_agent",
            "__end__": END,
        },
    )

    builder.add_edge("profile_agent", END)
    builder.add_edge("rag_agent", END)
    builder.add_edge("advisory_agent", END)

    return builder.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ----------------------------- Public API -----------------------------------

def chat(
    user_input: str,
    history: list,
    profile: dict,
    colleges_shown: list,
    rag_context: str,
) -> tuple[str, list, dict, list, str]:
    """
    Process one user turn. Returns:
      (ai_response, updated_history, updated_profile, updated_colleges_shown, updated_rag_context)

    Token guardrails:
    - bound prior history before invoking graph
    - bound returned history for subsequent turns
    """
    now = time.time()
    key = _cache_key(user_input=user_input, profile=profile)
    cached = _get_cached_payload(key=key, now=now)

    if cached:
        bounded_history = list(history)[-12:]
        bounded_history = bounded_history + [
            HumanMessage(content=user_input),
            AIMessage(content=cached["ai_response"]),
        ]
        updated_history = bounded_history[-16:]
        return (
            cached["ai_response"],
            updated_history,
            cached["student_profile"],
            cached["colleges_shown"],
            cached["rag_context"],
        )

    bounded_history = list(history)[-12:]
    bounded_history = bounded_history + [HumanMessage(content=user_input)]

    state = {
        "messages": bounded_history,
        "student_profile": profile,
        "rag_context": rag_context,
        "current_agent": "",
        "colleges_shown": colleges_shown,
        "profile_complete": bool(
            (profile.get("tnea_rank") or profile.get("jee_rank") or profile.get("twelfth_percent"))
            and profile.get("community")
        ),
        "profile_asks": 0,
    }

    result = get_graph().invoke(state)

    ai_response = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            content_text = _message_text(msg.content)
            if content_text:
                ai_response = content_text
                break

    if not ai_response:
        ai_response = (
            "I received your message, but couldn't produce a complete answer this turn. "
            "Please retry once and I’ll continue from your details."
        )

    updated_history = list(result.get("messages", []))[-16:]

    out_profile = result.get("student_profile", profile)
    out_colleges = result.get("colleges_shown", colleges_shown)
    out_context = result.get("rag_context", rag_context)

    _set_cached_payload(
        key=key,
        now=now,
        ai_response=ai_response,
        student_profile=out_profile,
        colleges_shown=out_colleges,
        rag_context=out_context,
    )

    return (
        ai_response,
        updated_history,
        out_profile,
        out_colleges,
        out_context,
    )
