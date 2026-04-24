"""
Shared LangGraph State definition.
All agents read from and write to this shared state.
"""
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class StudentProfile(TypedDict, total=False):
    tnea_rank: Optional[int]
    jee_rank: Optional[int]
    twelfth_percent: Optional[float]
    community: Optional[str]
    preferred_city: Optional[str]
    preferred_branch: Optional[str]
    max_fee: Optional[int]
    ownership_preference: Optional[str]


class ChatState(TypedDict):
    # Message history (LangGraph managed)
    messages: Annotated[list, add_messages]

    # Accumulated student academic profile
    student_profile: dict

    # Last RAG context string (from retrieval)
    rag_context: str

    # Which agent handled the last turn
    current_agent: str

    # List of college names shown so far (for context)
    colleges_shown: List[str]

    # Whether we have enough info to search
    profile_complete: bool

    # Number of times we've asked for profile info
    profile_asks: int
