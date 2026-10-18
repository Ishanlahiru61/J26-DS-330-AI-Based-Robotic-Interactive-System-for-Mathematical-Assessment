"""Robot-control application use cases."""

from typing import Protocol


class RobotController(Protocol):
    """Hardware adapter implemented on the edge computer."""

    async def demonstrate(self, movement_name: str) -> None:
        """Execute a calibrated, allow-listed movement."""


class RobotControlService:
    def __init__(self, controller: RobotController) -> None:
        self._controller = controller

    async def demonstrate(self, movement_name: str) -> None:
        await self._controller.demonstrate(movement_name)
