"""
LangGraph Multi-Agent Supervisor
---------------------------------
Graph topology:
  START → supervisor
  supervisor → profile_agent  (if need student info)
  supervisor → rag_agent      (if college search/details)
  supervisor → advisory_agent (if comparison/deadline/branch/policy)
  supervisor → END

Each agent node runs its dedicated LLM+tools, then returns to supervisor.
The supervisor synthesizes the final response.

Anti-hallucination:
  - System prompts strictly ground responses to tool outputs
  - LLM is forbidden from inventing college names/fees/dates
  - All tool outputs reference "Verified TN Engineering College Database"
"""
import json
import os
from typing import Literal
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from graph.state import ChatState
from tools.profile_tools import extract_student_info, validate_student_profile, get_missing_fields
from tools.rag_tools import rag_semantic_search, find_eligible_colleges, get_college_details
from tools.structured_tools import (
    compare_colleges,
    get_admission_deadlines,
    search_by_branch,
    get_reservation_policy,
)

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

def _make_llm(tools=None):
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0.1,  # Low temp = less hallucination
        max_tokens=4096,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    if tools:
        return llm.bind_tools(tools)
    return llm


# ─── Tool registries per agent ────────────────────────────────────────────────
PROFILE_TOOLS = [extract_student_info, validate_student_profile, get_missing_fields]
RAG_TOOLS = [rag_semantic_search, find_eligible_colleges, get_college_details]
ADVISORY_TOOLS = [compare_colleges, get_admission_deadlines, search_by_branch, get_reservation_policy]
ALL_TOOL_MAP = {t.name: t for t in PROFILE_TOOLS + RAG_TOOLS + ADVISORY_TOOLS}


def _message_text(content) -> str:
    """Normalize LLM content that may be str or a list of content parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                txt = item.get("text") or item.get("output_text") or item.get("content")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def _heuristic_route(state: ChatState) -> str:
    """Fast deterministic routing to avoid an extra LLM call on obvious intents."""
    profile = state.get("student_profile", {}) or {}

    last_user_text = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            last_user_text = _message_text(getattr(msg, "content", ""))
            break

    text = last_user_text.lower().strip()
    if not text:
        return ""

    advisory_keywords = [
        "compare", "comparison", "deadline", "date", "last date",
        "reservation", "quota", "policy", "branch", "specialization",
    ]
    rag_keywords = [
        "eligible", "college", "colleges", "admission", "documents",
        "fees", "fee", "cutoff", "placements", "nirf", "hostel",
        "find", "search", "show me", "suggest", "recommend",
    ]

    if any(k in text for k in advisory_keywords):
        return "advisory_agent"

    # If user asks for recommendations/search and also shares profile details,
    # go directly to RAG in the same turn.
    asks_for_colleges = any(k in text for k in rag_keywords)

    # Prioritize profile collection when user shares rank/marks/community.
    profile_cues = ["rank", "%", "percent", "community", "obc", "bc", "mbc", "sc", "st", "pwd"]
    shares_profile_details = any(k in text for k in profile_cues)
    missing_core = not (
        (profile.get("tnea_rank") or profile.get("jee_rank") or profile.get("twelfth_percent"))
        and profile.get("community")
    )
    if asks_for_colleges and shares_profile_details:
        return "rag_agent"

    if missing_core and shares_profile_details:
        return "profile_agent"

    if asks_for_colleges:
        return "rag_agent"

    return ""

# ─── System Prompts ───────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM = """You are the Supervisor of an AI admissions assistant for Tamil Nadu engineering colleges (2026-27).

Your job: decide which specialist agent should handle the user's message.

Agents available:
- profile_agent  : Collect/validate student info (rank, marks, community, preferences)
- rag_agent      : Search colleges, check eligibility, get college details
- advisory_agent : Compare colleges, check deadlines, find by branch, explain reservations

ROUTING RULES:
1. If the student hasn't provided TNEA/JEE rank AND 12th percentage → route to "profile_agent"
2. If user asks to list/find eligible colleges → route to "rag_agent"
3. If user asks about a specific college's details, admission, documents → route to "rag_agent"
4. If user asks to compare colleges → route to "advisory_agent"
5. If user asks about deadlines, dates → route to "advisory_agent"
6. If user asks about a specific branch across colleges → route to "advisory_agent"
7. If user asks about reservations, quotas → route to "advisory_agent"
8. General greetings or off-topic → answer directly without routing.

Respond ONLY with one of: profile_agent | rag_agent | advisory_agent | direct_response

Current student profile: {profile}
"""

PROFILE_SYSTEM = """You are the Profile Collection Agent for a Tamil Nadu engineering college admissions assistant.

Your ONLY job: collect the student's academic details conversationally.

Required information:
1. TNEA rank OR JEE Main rank (lower = better)
2. 12th standard percentage
3. Community category (General/OC, OBC/BC/MBC, SC, ST, PWD)
4. Preferred branch (optional but helpful)
5. Preferred city in Tamil Nadu (optional)

RULES:
- Use extract_student_info to parse the user's message first
- Use validate_student_profile to check completeness
- Use get_missing_fields to know what to ask next
- Ask for ONE missing field at a time — don't overwhelm the student
- Be warm, encouraging, and conversational
- NEVER invent or assume academic scores
- If the student seems confused about TNEA, briefly explain: TNEA is Tamil Nadu Engineering Admissions — the state-level entrance/counselling process for govt-affiliated engineering colleges.
"""

RAG_SYSTEM = """You are the College Search Agent for Tamil Nadu engineering college admissions.

CRITICAL ANTI-HALLUCINATION RULES:
1. ONLY use information returned by your tools — NEVER invent college names, fees, cutoffs, or dates
2. If a college is not found by tools, say "I don't have data for that college" — do NOT make up info
3. Always cite that data comes from the "Verified TN Engineering College Database"
4. When listing colleges, always include: city, ownership, NIRF rank, fees, entrance exam
5. When showing cutoffs, always clarify which community category they apply to

Your tools:
- find_eligible_colleges : filters colleges by rank/marks/community from verified data
- get_college_details    : retrieves full info for a named college
- rag_semantic_search    : semantic search for general college queries

Student profile: {profile}

Response format:
- List colleges with key facts in a clean format
- Include application deadlines proactively
- Mention TNEA counselling for government colleges
- Mention direct application for private/deemed colleges
"""

ADVISORY_SYSTEM = """You are the Advisory Agent for Tamil Nadu engineering college admissions.

CRITICAL ANTI-HALLUCINATION RULES:
1. ONLY use data returned by your tools — never fabricate comparisons or facts
2. If a college is not in the database, explicitly state that
3. Base all recommendations on verified cutoff/placement/fee data

Your tools:
- compare_colleges        : side-by-side comparison of 2–4 colleges
- get_admission_deadlines : important dates and deadlines
- search_by_branch        : colleges offering a specific branch
- get_reservation_policy  : Tamil Nadu reservation percentages

Student profile: {profile}

Be specific, data-driven, and honest. If data is missing, say so.
"""

# ─── Helper: run tool calls ────────────────────────────────────────────────────

def _execute_tools(last_message, tool_map):
    """Execute all tool calls in last_message and return ToolMessage list."""
    results = []
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return results
    for tc in last_message.tool_calls:
        fn = tool_map.get(tc["name"])
        if fn:
            try:
                out = fn.invoke(tc["args"])
            except Exception as e:
                out = json.dumps({"status": "error", "message": str(e)})
        else:
            out = json.dumps({"status": "error", "message": f"Unknown tool: {tc['name']}"})
        results.append(ToolMessage(content=out, tool_call_id=tc["id"]))
    return results


def _extract_profile_updates(state: ChatState, tool_results: list) -> dict:
    """Scan tool results for profile fields to merge into student_profile."""
    profile = dict(state.get("student_profile", {}))
    for msg in tool_results:
        try:
            data = json.loads(msg.content)
            extracted = data.get("extracted", {})
            for field in ["tnea_rank", "jee_rank", "twelfth_percent", "community",
                          "preferred_city", "preferred_branch", "max_fee"]:
                if extracted.get(field) is not None:
                    profile[field] = extracted[field]
        except Exception:
            pass
    return profile


# ─── Agent Nodes ──────────────────────────────────────────────────────────────

def profile_agent_node(state: ChatState) -> dict:
    """Collect and validate student profile information."""
    profile = state.get("student_profile", {})
    profile_str = json.dumps(profile) if profile else "No profile collected yet."

    system = SystemMessage(content=PROFILE_SYSTEM)
    llm = _make_llm(PROFILE_TOOLS)

    response = llm.invoke([system] + state["messages"])
    tool_results = _execute_tools(response, {t.name: t for t in PROFILE_TOOLS})

    new_profile = _extract_profile_updates(state, tool_results)

    # Check completeness
    is_complete = bool(
        (new_profile.get("tnea_rank") or new_profile.get("jee_rank") or new_profile.get("twelfth_percent"))
        and new_profile.get("community")
    )

    new_msgs = [response] + tool_results
    if tool_results:
        # Second LLM pass to generate user-facing response
        follow_up = llm.invoke([system] + state["messages"] + new_msgs)
        # Only keep final AIMessage (no more tool calls expected)
        if not (hasattr(follow_up, "tool_calls") and follow_up.tool_calls):
            new_msgs.append(follow_up)

    return {
        "messages": new_msgs,
        "student_profile": new_profile,
        "current_agent": "profile_agent",
        "profile_complete": is_complete,
        "profile_asks": state.get("profile_asks", 0) + 1,
    }


def rag_agent_node(state: ChatState) -> dict:
    """Search colleges and retrieve details using RAG."""
    profile = state.get("student_profile", {})
    profile_str = json.dumps(profile) if profile else "{}"

    system = SystemMessage(content=RAG_SYSTEM.format(profile=profile_str))
    llm = _make_llm(RAG_TOOLS)

    response = llm.invoke([system] + state["messages"])
    tool_results = _execute_tools(response, {t.name: t for t in RAG_TOOLS})

    # Extract colleges shown for context tracking
    colleges_shown = list(state.get("colleges_shown", []))
    rag_context = state.get("rag_context", "")

    for msg in tool_results:
        try:
            data = json.loads(msg.content)
            if "colleges" in data:
                for c in data["colleges"]:
                    name = c.get("college_name", "")
                    if name and name not in colleges_shown:
                        colleges_shown.append(name)
            if "context" in data:
                rag_context = data["context"]
        except Exception:
            pass

    new_msgs = [response] + tool_results
    if tool_results:
        follow_up = llm.invoke([system] + state["messages"] + new_msgs)
        if not (hasattr(follow_up, "tool_calls") and follow_up.tool_calls):
            new_msgs.append(follow_up)

    return {
        "messages": new_msgs,
        "current_agent": "rag_agent",
        "colleges_shown": colleges_shown,
        "rag_context": rag_context,
    }


def advisory_agent_node(state: ChatState) -> dict:
    """Handle comparisons, deadlines, branch searches, and reservations."""
    profile = state.get("student_profile", {})
    profile_str = json.dumps(profile) if profile else "{}"

    system = SystemMessage(content=ADVISORY_SYSTEM.format(profile=profile_str))
    llm = _make_llm(ADVISORY_TOOLS)

    response = llm.invoke([system] + state["messages"])
    tool_results = _execute_tools(response, {t.name: t for t in ADVISORY_TOOLS})

    new_msgs = [response] + tool_results
    if tool_results:
        follow_up = llm.invoke([system] + state["messages"] + new_msgs)
        if not (hasattr(follow_up, "tool_calls") and follow_up.tool_calls):
            new_msgs.append(follow_up)

    return {
        "messages": new_msgs,
        "current_agent": "advisory_agent",
    }


def supervisor_node(state: ChatState) -> dict:
    """Route to the appropriate agent or respond directly."""
    heuristic = _heuristic_route(state)
    if heuristic:
        return {
            "messages": [],
            "current_agent": heuristic,
        }

    profile = state.get("student_profile", {})
    profile_str = json.dumps(profile) if profile else "Empty"

    llm = _make_llm()
    system = SystemMessage(content=SUPERVISOR_SYSTEM.format(profile=profile_str))

    response = llm.invoke([system] + state["messages"])
    route = _message_text(response.content).strip().lower()

    return {
        "messages": [response] if "direct_response" in route else [],
        "current_agent": route,
    }


# ─── Routing ──────────────────────────────────────────────────────────────────

def route_supervisor(state: ChatState) -> Literal["profile_agent", "rag_agent", "advisory_agent", "__end__"]:
    agent = str(state.get("current_agent", "")).strip().lower()
    if "profile" in agent:
        return "profile_agent"
    if "rag" in agent or "college" in agent:
        return "rag_agent"
    if "advisory" in agent or "compare" in agent or "deadline" in agent:
        return "advisory_agent"
    return "__end__"


# ─── Build Graph ──────────────────────────────────────────────────────────────

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

    # All agents return to END (supervisor is re-entered on next user message)
    builder.add_edge("profile_agent", END)
    builder.add_edge("rag_agent", END)
    builder.add_edge("advisory_agent", END)

    return builder.compile()


# Singleton
_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ─── Public chat function ─────────────────────────────────────────────────────

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
    """
    history = list(history) + [HumanMessage(content=user_input)]

    state = {
        "messages": history,
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

    # Extract last AI text response
    ai_response = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            content_text = _message_text(msg.content)
            route_labels = {"profile_agent", "rag_agent", "advisory_agent", "direct_response"}
            is_route_label = content_text.strip().lower() in route_labels
            if content_text and not is_route_label:
                # Prefer non-tool-call final messages, but allow tool-call text as fallback.
                if (not msg.tool_calls if hasattr(msg, "tool_calls") else True):
                    ai_response = content_text
                    break
                if not ai_response:
                    ai_response = content_text

    if not ai_response:
        ai_response = "I have your message, but the model did not return a complete response this turn. Please retry once, and I will continue from your provided details."

    return (
        ai_response,
        result["messages"],
        result.get("student_profile", profile),
        result.get("colleges_shown", colleges_shown),
        result.get("rag_context", rag_context),
    )
