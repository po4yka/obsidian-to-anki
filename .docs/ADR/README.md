# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) that document the architectural decisions made for the Obsidian to Anki sync service.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## ADR Structure

Each ADR follows this structure:

-   **Title**: Clear, descriptive title
-   **Status**: Current status (Proposed, Accepted, Rejected, Deprecated, Superseded)
-   **Context**: Situation that led to the decision
-   **Decision**: What was decided and why
-   **Consequences**: Positive and negative outcomes
-   **Alternatives**: Other options considered
-   **Implementation**: How the decision was implemented
-   **Testing**: How the decision affects testing
-   **Migration**: How to migrate to/from this decision

## Status Definitions

-   **Proposed**: Decision is being considered
-   **Accepted**: Decision is implemented and active
-   **Rejected**: Decision was considered but not implemented
-   **Deprecated**: Decision is no longer recommended
-   **Superseded**: Decision replaced by a newer one

## Current ADRs

| ADR                                                  | Title                                    | Status   | Date       |
| ---------------------------------------------------- | ---------------------------------------- | -------- | ---------- |
| [ADR-001](ADR-001-adopt-clean-architecture.md)       | Adopt Clean Architecture                 | Accepted | 2024-12-XX |
| [ADR-002](ADR-002-dependency-injection-container.md) | Implement Dependency Injection Container | Accepted | 2024-12-XX |
| [ADR-003](ADR-003-interface-segregation.md)          | Apply Interface Segregation Principle    | Accepted | 2024-12-XX |

## Creating New ADRs

When creating a new ADR:

1. **Copy the template**: Use the structure from existing ADRs
2. **Choose filename**: `ADR-XXX-descriptive-title.md`
3. **Assign number**: Next sequential number (001, 002, etc.)
4. **Set status**: Start with "Proposed" or "Accepted"
5. **Document thoroughly**: Include all context, alternatives, and consequences

## ADR Template

```markdown
# ADR XXX: Title

## Status

Proposed/Accepted/Rejected/Deprecated/Superseded

## Context

[Describe the situation that led to this decision]

## Decision

[What was decided and why]

## Consequences

### Positive

[Benefits of this decision]

### Negative

[Drawbacks of this decision]

### Neutral

[Neutral impacts]

## Alternatives Considered

[Other options that were considered]

## Implementation

[How this decision was implemented]

## Testing Strategy

[How this affects testing]

## Migration Strategy

[How to adopt this change]

## Success Metrics

[How to measure success]

## References

[Links to relevant documentation, articles, etc.]
```

## ADR Process

1. **Identify Decision**: Recognize when an architectural decision needs to be made
2. **Document Context**: Write down the current situation and constraints
3. **Propose Solution**: Document the proposed decision
4. **Evaluate Alternatives**: Consider other options and their trade-offs
5. **Make Decision**: Choose the best option based on analysis
6. **Implement**: Put the decision into practice
7. **Review**: Periodically review the decision and its outcomes

## Questions?

For questions about ADRs or architectural decisions, see:

-   [Project Architecture](../README.md)
-   [Contributing Guidelines](../../CONTRIBUTING.md)
-   [Development Guidelines](../../DEVELOPMENT.md)
