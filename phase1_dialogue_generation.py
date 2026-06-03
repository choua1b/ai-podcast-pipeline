# phase1_dialogue_generation.py
"""
Phase 1: Scenario and Dialogue Creation
========================================
Generates a structured JSON dialogue between two AI personas (ARIA and MARCUS)
discussing the impact of Artificial Intelligence on society, education, work,
creativity, and science.

Output Contract (consumed by Phase 2):
    outputs/dialogue.json — validated PodcastDialogue schema

Dependencies:
    pip install openai anthropic pydantic python-dotenv tenacity
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ─────────────────────────────────────────────
# 1. DATA MODELS (Phase 1 → Phase 2 Contract)
# ─────────────────────────────────────────────

class Speaker(str, Enum):
    ARIA = "ARIA"
    MARCUS = "MARCUS"


class EmotionTag(str, Enum):
    """
    Controls facial expression in Phase 4 (SadTalker / animation model).
    Maps to expression conditioning vectors in the latent space.
    """
    NEUTRAL    = "neutral"
    ENTHUSIASTIC = "enthusiastic"
    THOUGHTFUL = "thoughtful"
    CONCERNED  = "concerned"
    CURIOUS    = "curious"
    CONFIDENT  = "confident"


class DialogueTurn(BaseModel):
    turn_id: int = Field(..., description="Zero-indexed turn number")
    speaker: Speaker
    text: str = Field(..., min_length=10, description="Spoken dialogue text")
    emotion_tag: EmotionTag = Field(
        default=EmotionTag.NEUTRAL,
        description="Dominant emotion — used by Phase 4 for facial animation"
    )
    duration_hint_seconds: float = Field(
        ...,
        gt=0,
        description="Estimated TTS duration based on 140 WPM; validated in Phase 2"
    )
    word_count: int = Field(..., gt=0)

    @field_validator("text")
    @classmethod
    def no_forbidden_topics(cls, v: str) -> str:
        forbidden = ["religion", "god", "allah", "politics", "democrat",
                     "republican", "election", "vote", "prayer", "church"]
        v_lower = v.lower()
        for word in forbidden:
            if word in v_lower:
                raise ValueError(f"Forbidden topic keyword detected: '{word}'")
        return v

    @model_validator(mode="after")
    def validate_word_count_consistency(self) -> "DialogueTurn":
        actual_wc = len(self.text.split())
        if abs(actual_wc - self.word_count) > 5:
            raise ValueError(
                f"Turn {self.turn_id}: word_count={self.word_count} "
                f"inconsistent with actual={actual_wc}"
            )
        self.word_count = actual_wc  # Normalize
        # Recompute duration hint from actual word count
        self.duration_hint_seconds = round((actual_wc / 140) * 60, 2)
        return self


class PersonaProfile(BaseModel):
    name: Speaker
    voice_descriptor: str = Field(
        ..., description="Natural language voice profile for TTS in Phase 2"
    )
    personality_summary: str
    rhetorical_style: str


class PodcastDialogue(BaseModel):
    """
    Top-level schema. This object is the Phase 1 → Phase 2 contract.
    """
    title: str
    topic: str
    total_word_count: int
    estimated_total_duration_seconds: float
    personas: list[PersonaProfile]
    turns: list[DialogueTurn]

    @model_validator(mode="after")
    def validate_global_constraints(self) -> "PodcastDialogue":
        # Must have both speakers
        speakers_present = {t.speaker for t in self.turns}
        if Speaker.ARIA not in speakers_present or Speaker.MARCUS not in speakers_present:
            raise ValueError("Dialogue must include both ARIA and MARCUS")

        # Must alternate speakers (no two consecutive same speaker)
        for i in range(1, len(self.turns)):
            if self.turns[i].speaker == self.turns[i - 1].speaker:
                raise ValueError(
                    f"Turns {i-1} and {i} have the same speaker — must alternate"
                )

        # Minimum word count
        total_wc = sum(t.word_count for t in self.turns)
        if total_wc < 280:
            raise ValueError(
                f"Total word count {total_wc} is below minimum of 280 for 2-minute video"
            )

        # Normalize totals
        self.total_word_count = total_wc
        self.estimated_total_duration_seconds = round(sum(
            t.duration_hint_seconds for t in self.turns
        ), 2)
        return self


# ─────────────────────────────────────────────
# 2. PROMPT ENGINEERING
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a professional podcast scriptwriter for a university-level documentary series
on Artificial Intelligence and Society.

Your task is to write a realistic, intellectually rigorous podcast dialogue between
two virtual characters. The dialogue will be converted to speech and animated,
so it must sound natural when spoken aloud.

CHARACTERS:
- ARIA: An analytical AI researcher. Uses precise, data-driven language. 
  Optimistic but nuanced. Short, impactful sentences. Emotion: enthusiastic, confident.
- MARCUS: A philosopher of technology and humanist. Uses storytelling and analogy. 
  Reflective and thoughtful. Slightly longer sentences. Emotion: thoughtful, curious.

STRICT RULES:
1. Absolutely NO political or religious content of any kind.
2. The dialogue must cover at least 3 of these themes:
   AI in education, AI in the workplace, AI in creativity, AI in scientific discovery,
   AI and human identity, ethical implications of AI.
3. Total word count across all turns MUST be between 300 and 380 words.
4. Exactly 12 turns total (turn_id 0 through 11), strictly alternating ARIA/MARCUS.
   Turn 0 is ARIA (the host introducing the topic).
5. Each turn must be 20–45 words. Natural spoken register — no bullet points.
6. Assign a dominant emotion to each turn from:
   [neutral, enthusiastic, thoughtful, concerned, curious, confident]

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown, no explanation:
{
  "title": "<podcast episode title>",
  "topic": "<one-sentence topic summary>",
  "total_word_count": <integer>,
  "estimated_total_duration_seconds": <float>,
  "personas": [
    {
      "name": "ARIA",
      "voice_descriptor": "<description of ARIA voice for TTS: tone, pace, pitch>",
      "personality_summary": "<2 sentences>",
      "rhetorical_style": "<1 sentence>"
    },
    {
      "name": "MARCUS", 
      "voice_descriptor": "<description of MARCUS voice for TTS>",
      "personality_summary": "<2 sentences>",
      "rhetorical_style": "<1 sentence>"
    }
  ],
  "turns": [
    {
      "turn_id": 0,
      "speaker": "ARIA",
      "text": "<spoken dialogue>",
      "emotion_tag": "<emotion>",
      "duration_hint_seconds": <float>,
      "word_count": <integer>
    }
    // ... turns 1 through 11
  ]
}
"""

USER_PROMPT = """
Generate the full podcast dialogue JSON now. The episode should open with ARIA
welcoming the audience and framing the central question: How will AI reshape what
it means to be human — in how we learn, work, create, and discover?

The dialogue should build toward a nuanced conclusion: neither utopian nor dystopian,
but focused on the irreplaceable role of human judgment, empathy, and collaboration
alongside AI systems.
"""


# ─────────────────────────────────────────────
# 3. LLM CLIENT (OpenAI or Anthropic)
# ─────────────────────────────────────────────

class DialogueGenerator:
    """
    Wraps OpenAI or Anthropic API with retry logic and JSON validation.
    Set LLM_PROVIDER="openai" or "anthropic" in your .env file.
    """

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
        self._init_client()

    def _init_client(self) -> None:
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: '{self.provider}'")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _call_openai(self) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},  # Enforces JSON output
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT},
            ],
            temperature=0.85,    # Slight creativity for natural dialogue
            max_tokens=2500,
        )
        return response.choices[0].message.content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _call_anthropic(self) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            temperature=0.85,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": USER_PROMPT}],
        )
        # Extract raw text — Anthropic returns TextBlock objects
        raw = "".join(
            block.text for block in response.content
            if hasattr(block, "text")
        )
        # Strip any accidental markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw.strip()

    def generate(self) -> PodcastDialogue:
        """
        Calls the LLM, parses the response, and validates against PodcastDialogue schema.
        Returns a validated PodcastDialogue object.
        """
        print(f"[Phase 1] Calling {self.provider.upper()} ({self.model})...")

        if self.provider == "openai":
            raw_json = self._call_openai()
        else:
            raw_json = self._call_anthropic()

        print("[Phase 1] Raw response received. Parsing JSON...")

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\n\nRaw output:\n{raw_json}")

        print("[Phase 1] Validating schema with Pydantic...")
        dialogue = PodcastDialogue(**data)

        print(
            f"[Phase 1] ✓ Validation passed.\n"
            f"          Title    : {dialogue.title}\n"
            f"          Turns    : {len(dialogue.turns)}\n"
            f"          Words    : {dialogue.total_word_count}\n"
            f"          Duration : {dialogue.estimated_total_duration_seconds:.1f}s "
            f"({dialogue.estimated_total_duration_seconds/60:.2f} min)"
        )
        return dialogue


# ─────────────────────────────────────────────
# 4. OUTPUT & PERSISTENCE
# ─────────────────────────────────────────────

def save_dialogue(dialogue: PodcastDialogue, output_dir: Path) -> Path:
    """
    Serializes the validated dialogue to JSON.
    This file is the Phase 2 input artifact.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dialogue.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dialogue.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"[Phase 1] ✓ Dialogue saved → {output_path}")
    return output_path


def print_dialogue_preview(dialogue: PodcastDialogue) -> None:
    """Prints a formatted preview of the generated dialogue to stdout."""
    print("\n" + "═" * 60)
    print(f"  PODCAST: {dialogue.title}")
    print(f"  TOPIC  : {dialogue.topic}")
    print("═" * 60)
    for turn in dialogue.turns:
        emotion_str = f"[{turn.emotion_tag.value}]"
        print(f"\n  [{turn.speaker.value}] {emotion_str}")
        print(f"  {turn.text}")
        print(f"  ↳ {turn.word_count} words | ~{turn.duration_hint_seconds:.1f}s")
    print("\n" + "═" * 60)
    print(f"  TOTAL: {dialogue.total_word_count} words | "
          f"{dialogue.estimated_total_duration_seconds:.1f}s "
          f"({dialogue.estimated_total_duration_seconds/60:.2f} min)")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────
# 5. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_phase1(output_dir: Path = Path("outputs")) -> PodcastDialogue:
    """
    Full Phase 1 execution.

    Args:
        output_dir: Directory where dialogue.json will be written.

    Returns:
        Validated PodcastDialogue object (also written to disk).
    """
    generator = DialogueGenerator()
    dialogue  = generator.generate()

    print_dialogue_preview(dialogue)
    save_dialogue(dialogue, output_dir)

    return dialogue


if __name__ == "__main__":
    run_phase1()