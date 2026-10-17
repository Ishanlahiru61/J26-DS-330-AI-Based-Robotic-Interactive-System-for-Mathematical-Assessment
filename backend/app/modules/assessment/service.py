"""Assessment session use cases."""

from backend.app.shared.contracts import AssessmentEvent


class AssessmentService:
    """Coordinates lifecycle operations without owning research logic."""

    def register_event(self, event: AssessmentEvent) -> AssessmentEvent:
        """Return an already validated event for a future persistence adapter."""

        return event
