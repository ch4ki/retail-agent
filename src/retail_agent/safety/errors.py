class SafetyError(Exception):
    """Base class for guard and policy failures."""


class GuardViolation(SafetyError):
    def __init__(self, violations: tuple[str, ...]) -> None:
        super().__init__("; ".join(violations))
        self.violations = violations
