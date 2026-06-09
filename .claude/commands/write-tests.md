# Write Tests

Write production-grade tests for the requested feature.

Requirements:

* meaningful coverage
* maintainable test structure
* deterministic tests
* avoid flaky behavior
* avoid excessive mocking
* focus on business behavior

Backend testing:

* pytest
* async test support
* API integration tests
* service layer tests
* repository tests where appropriate

Frontend testing:

* Vitest
* React Testing Library
* component behavior testing
* user interaction testing
* loading/error state testing

Focus on:

* critical business logic
* validation behavior
* API correctness
* state transitions
* edge cases
* failure scenarios

Avoid:

* testing implementation details
* brittle snapshot tests
* excessive mocking
* redundant tests

Include:

* clear test naming
* reusable fixtures/helpers
* clean setup/teardown
* typed test utilities

After writing tests:

1. summarize coverage
2. identify remaining testing gaps
3. explain tradeoffs
4. provide a professional git commit message
