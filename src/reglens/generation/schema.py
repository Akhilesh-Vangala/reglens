"""Structured answer contract enforced on the model through JSON-schema structured output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["answered", "not_in_corpus", "refused"]


class Claim(BaseModel):
    statement: str = Field(description="One plain-English fact for the member.")
    chunk_id: str = Field(description="id of the passage this claim rests on, copied exactly.")
    quote: str = Field(description="Verbatim sentence(s) copied from that passage that support the statement.")


class Answer(BaseModel):
    status: Status = Field(description="answered | not_in_corpus | refused")
    answer: str = Field(description="2 to 5 sentence answer in plain English for a credit union member.")
    claims: list[Claim] = Field(description="Every factual statement in the answer, each with a verbatim quote.")
    needs_human_review: bool = Field(description="True when a member service representative must act or decide.")
    review_reason: str = Field(description="Why human review is needed, or empty string.")


class ClosedBookClaim(BaseModel):
    statement: str
    citation: str = Field(description='Regulation citation, e.g. "12 CFR 1005.11(c)(1)".')


class ClosedBookAnswer(BaseModel):
    status: Status
    answer: str
    claims: list[ClosedBookClaim]
    needs_human_review: bool
    review_reason: str
