# ζ-engine Retrieval Context

ζ-engine turns crawled university pages into evidence-backed answers. These terms distinguish stored source material from the assertions made in an answer.

## Language

**Document**:
A complete web page captured by the crawler as source material.
_Avoid_: Evidence, search result

**Evidence**:
An immutable, query-relevant passage from a Document that can support or contradict a Claim.
_Avoid_: Document, context, result

**Claim**:
One independently verifiable assertion in a proposed or final answer.
_Avoid_: Answer, summary

**Citation**:
The explicit relationship showing which Evidence supports a Claim.
_Avoid_: Source list, search result

**Requirement**:
One answer slot that can be retrieved and verified independently, optionally after other Requirements are answered.
_Avoid_: Sub-question, prompt

**Requirement State**:
The current outcome of a Requirement: pending, answered, missing, conflicting, blocked, or error.
_Avoid_: Overall answer status

**Partial Answer**:
A final answer that preserves every answered Requirement while identifying Requirements that remain unresolved.
_Avoid_: Failed answer, insufficient evidence
