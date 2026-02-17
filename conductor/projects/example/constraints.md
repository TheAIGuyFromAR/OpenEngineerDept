# Project Constraints

## Code Style
- Python 3.11+, type hints on all public functions
- Use dataclasses over dicts for structured data
- Prefer composition over inheritance

## Architecture
- No direct database access from handlers — use repository pattern
- All external calls must have timeout and retry logic
- Errors are logged, not swallowed

## Testing
- Every new function must have at least one test
- Tests must not depend on network or external services
- Use pytest with fixtures, no unittest.TestCase
