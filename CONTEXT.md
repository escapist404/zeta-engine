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
