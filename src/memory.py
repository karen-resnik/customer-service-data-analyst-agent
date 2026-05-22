import os
import json
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


PROFILE_DIR = Path("memory/profiles")
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
MEMORY_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

class UserProfile(BaseModel):
    """Persistent user profile distilled from conversation."""

    name: str | None = Field(
        default=None,
        description="User's name, if provided.",
    )
    interests: list[str] = Field(
        default_factory=list,
        description="Topics the user seems interested in.",
    )
    preferences: list[str] = Field(
        default_factory=list,
        description="User preferences about answers or workflow.",
    )
    asked_about_topics: list[str] = Field(
        default_factory=list,
        description="Topics the user has asked about, inferred from their questions.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Other stable facts about the user.",
    )

class UserProfileUpdate(BaseModel):
    """Structured profile update extracted from a user message."""

    name: str | None = Field(
        default=None,
        description="User's name if explicitly stated.",
    )
    interests: list[str] = Field(
        default_factory=list,
        description="Stable user interests or recurring topics.",
    )
    preferences: list[str] = Field(
        default_factory=list,
        description="Stable preferences about answers, workflow, style, or tools.",
    )
    asked_about_topics: list[str] = Field(
        default_factory=list,
        description="Topics inferred from what the user is asking about in this message.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Other stable facts about the user.",
    )
    is_memory_question: bool = Field(
        default=False,
        description="Whether the user is asking what is remembered about them.",
    )


PROFILE_EXTRACTION_SYSTEM_PROMPT = """
You extract persistent user-profile information from a single user message.

Return valid JSON only. Do not use markdown.

Extract two different kinds of memory:

1. Explicit profile facts:
- user's name or preferred name
- topics the user explicitly says they are interested in, care about, focus on, or work with
- preferences about answer style, coding workflow, debugging, tools, or project process
- other stable user facts

2. Observed asked-about topics:
- topics the user asks about in the current message, even if they did not say they are interested in them

Rules:
- Use interests only when the user explicitly says they care about, are interested in, prefer, focus on, or mostly work with a topic.
- Use asked_about_topics for the topic of the current user question.
- For asked_about_topics, use short phrases such as "refund examples", "payment issues", "dataset metadata", "conversation memory", "tool debugging", or "CLI testing".
- Do not store temporary task details, one-off shell commands, raw code, secrets, API keys, file paths, or private credentials.
- Only set name if the user explicitly states their name or preferred name.
- Use short, clean phrases for interests, preferences, asked_about_topics, and notes.
- If the user message is a follow-up such as "more", "next", "another", "same category", or "summarize this category", infer asked_about_topics from the assistant answer if it is available.
- Prefer meaningful topic phrases like "refund examples", "payment issue examples", or "refund category summary" over literal follow-up wording like "1 more" or "same category".
- If the message contains no profile-worthy facts and no clear asked-about topic, return empty lists and null name.
- If the user asks "what do you remember about me" or similar, set is_memory_question to true.
- Do not invent facts.

Return this exact JSON shape:
{
  "name": "name_or_null",
  "interests": ["explicit interest"],
  "preferences": ["preference"],
  "asked_about_topics": ["topic inferred from the message"],
  "notes": ["note"],
  "is_memory_question": true_or_false
}
""".strip()


def get_memory_client() -> OpenAI:
    """Create and return an OpenAI-compatible Nebius client for memory extraction."""
    load_dotenv()

    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError(
            "NEBIUS_API_KEY is not set. Add it to your .env file before running memory extraction."
        )

    return OpenAI(
        base_url=NEBIUS_BASE_URL,
        api_key=api_key,
    )


def extract_user_profile_update(user_message: str, agent_answer: str | None = None,) -> UserProfileUpdate:
    """Use an LLM to extract profile updates from a user message."""
    if not user_message.strip():
        return UserProfileUpdate()

    client = get_memory_client()

    response = client.chat.completions.create(
        model=MEMORY_MODEL,
        messages=[
            {
                "role": "system",
                "content": PROFILE_EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"User message:\n{user_message}\n\n"
                            f"Assistant answer, if available:\n{agent_answer or ''}"
                        ),
                    }
                ],
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Memory extraction returned an empty response.")

    data = json.loads(content)
    return UserProfileUpdate.model_validate(data)


def get_profile_path(session_id: str) -> Path:
    """Return the profile path for a session."""
    safe_session_id = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
    return PROFILE_DIR / f"{safe_session_id}.json"


def load_user_profile(session_id: str) -> UserProfile:
    """Load the user profile for a session, or return an empty profile."""
    profile_path = get_profile_path(session_id)

    if not profile_path.exists():
        return UserProfile()

    data = json.loads(profile_path.read_text(encoding="utf-8"))
    return UserProfile.model_validate(data)


def save_user_profile(session_id: str, profile: UserProfile) -> None:
    """Save the user profile for a session."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    profile_path = get_profile_path(session_id)
    profile_path.write_text(
        json.dumps(profile.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def profile_to_text(profile: UserProfile) -> str:
    """Convert a profile into readable text for the agent or CLI."""
    facts: list[str] = []

    if profile.name:
        facts.append(f"Name: {profile.name}")

    if profile.interests:
        facts.append(f"Interests: {', '.join(profile.interests)}")

    if profile.preferences:
        facts.append(f"Preferences: {', '.join(profile.preferences)}")

    if profile.asked_about_topics:
        facts.append(f"Asked about topics: {', '.join(profile.asked_about_topics)}")

    if profile.notes:
        facts.append(f"Notes: {', '.join(profile.notes)}")

    if not facts:
        return "No saved user profile facts yet."

    return "\n".join(facts)


def profile_to_dict(profile: UserProfile) -> dict[str, Any]:
    """Return a serializable profile dictionary."""
    return profile.model_dump()


def add_unique_value(values: list[str], value: str) -> list[str]:
    """Add a value to a list if it is not already present."""
    normalized_value = value.strip()

    if not normalized_value:
        return values

    if normalized_value.lower() not in {existing.lower() for existing in values}:
        values.append(normalized_value)

    return values


def update_user_profile_from_message(
    session_id: str,
    user_message: str,
    agent_answer: str | None = None,
) -> UserProfile:
    """Update the persistent user profile using LLM-extracted profile facts."""
    profile = load_user_profile(session_id)
    profile_update = extract_user_profile_update(user_message=user_message, agent_answer=agent_answer,)

    if profile_update.name:
        profile.name = profile_update.name

    for interest in profile_update.interests:
        profile.interests = add_unique_value(profile.interests, interest)

    for preference in profile_update.preferences:
        profile.preferences = add_unique_value(profile.preferences, preference)
    
    topics_seen_in_this_message: set[str] = set()

    for topic in profile_update.asked_about_topics:
        normalized_topic = topic.strip()

        if not normalized_topic:
            continue

        normalized_topic_key = normalized_topic.lower()

        if normalized_topic_key in topics_seen_in_this_message:
            continue

        topics_seen_in_this_message.add(normalized_topic_key)
        profile.asked_about_topics.append(normalized_topic)
    
    for note in profile_update.notes:
        profile.notes = add_unique_value(profile.notes, note)

    save_user_profile(session_id, profile)
    return profile


def user_asks_about_memory(user_message: str) -> bool:
    """Return True if the user asks what the assistant remembers about them."""
    profile_update = extract_user_profile_update(user_message=user_message)
    return profile_update.is_memory_question