# NNNN. <Title>

**Date:** YYYY-MM-DD

**Invariant affected:** <link to the section in CONTRIBUTING.md>

**Decision:** Accepted | Superseded by [NNNN](NNNN-*.md)

## Status

<One sentence. "Proposed" while under review, "Accepted" once merged.>

## Motivation

<The concrete problem this relaxation solves. Not the cost of the
invariant in the abstract — the cost of *keeping* it in this case. What
is blocked, what is degraded, what work cannot proceed?

If the honest answer is only that the invariant is inconvenient, write
that, and expect the reviewer to reject the ADR.>

## Alternatives considered

<At least one option that preserved the invariant entirely, and why it
did not work. A list of options without a reason each was rejected is
not this section — it is a table of contents.>

| Alternative | Why it was rejected |
| --- | --- |
| <Option that kept the invariant> | <Specific, not "too complex"> |
| <Option that kept the invariant> | <...> |
| <Option that relaxed the invariant less> | <...> |

## Migration strategy

<How existing code, data, and identifiers move to the relaxed regime.
Include the sequence if ordering matters: code first, then data, then
removal of the old path. State what happens to records written before
the change and whether both regimes must coexist during a window.>

## Compatibility impact

<What breaks. Specifically:

- Stored records affected, and whether they still verify.
- Content-addressed identifiers that change value.
- Serialized payloads that stop round-tripping.
- Public API or Protocol signatures that shift.

If nothing breaks, say so explicitly and name the check that proves it —
a claim of zero impact is only as good as the test behind it.>

## Why the previous invariant is no longer appropriate

<The section that forces real engagement.

Explain what the invariant assumed that no longer holds: a fact about
the system that changed, a failure mode it was designed for that has
been addressed by other means, or a scope that turned out to be wrong.

Restating the invariant's cost here is not an answer. If the underlying
assumption still holds, the ADR should not be accepted — the better
resolution is to strengthen or clarify the invariant instead.>

## Consequences

<What becomes easier, what becomes harder, and what future work this
unblocks or prevents. Note any CI changes: an ADR that relaxes an
enforced invariant necessarily changes what the checker permits, and the
change belongs in the same PR as this record.>
