# Contributing to Predictron Engine

Thank you for your interest in contributing to Predictron Engine.

Predictron Engine is a modular, explainable venture intelligence system. Contributions should preserve that focus and keep changes easy to test, review, and maintain.

## Design Principles

All contributions should align with the project's core principles:

- Explainability
- Deterministic behavior
- Modular architecture
- Evidence-first reasoning
- Continuous evaluation

## Development Workflow

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Add or update tests where appropriate.
5. Update documentation if the change affects behavior or usage.
6. Run the verified checks before opening a pull request:
	- `python -m pytest tests`
	- `python -m ruff check`

## Coding Standards

Please:

- Write clear, maintainable code.
- Keep components modular.
- Favor explicit behavior over implicit assumptions.
- Preserve backward compatibility whenever practical.
- Document significant architectural decisions.

## Testing

Before opening a pull request, verify that:

- Unit tests pass successfully with `python -m pytest tests`.
- Ruff checks pass with `python -m ruff check`.
- Existing functionality remains unaffected.
- Documentation reflects the implemented changes where relevant.

## Pull Requests

Each pull request should include:

- A concise summary of the change.
- Motivation for the implementation.
- Relevant testing information.
- References to related issues where applicable.

## Code of Conduct

Please communicate respectfully and constructively.

The goal of the project is to encourage collaborative engineering and continuous improvement.
