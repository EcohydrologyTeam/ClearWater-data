from clearwater_data.variables.base import Variable
from clearwater_data import ArrayLike
from datetime import datetime


class VariableRegistry:
    """
    A simple registry for storing variables by key.
    """

    def __init__(self):
        self._registry = {}

    def register(self, key: str, value: Variable, overwrite: bool = False):
        """
        Register a variable with a given key.
        """
        if key in self._registry and not overwrite:
            raise ValueError(f"Variable {key} already registered.")
        self._registry[key] = value

    def unregister(self, key: str):
        """
        Remove a variable from the registry.
        """
        if key in self._registry:
            del self._registry[key]

    def get(self, key: str) -> object:
        """
        Retrieve a variable by key.
        """
        variable: Variable | None = None
        try:
            variable = self._registry[key]
        except KeyError:
            raise ValueError(f"Variable {key} not found.")

        return variable.get()

    def get_at_time(self, key: str, time: datetime) -> object:
        """
        Retrieve a variable by key and time.
        """
        variable: Variable | None = None
        try:
            variable = self._registry[key]
        except KeyError:
            raise ValueError(
                f"Variable {key} not found in registry. Did you forget to register a variable?"
            )

        return variable.get_at_time(time)

    def set(self, key: str, value: ArrayLike) -> None:
        """
        Helper function to find a variable by key and call the set method.
        """
        variable: Variable | None = None
        try:
            variable = self._registry[key]
        except KeyError:
            raise ValueError(
                f"Variable {key} not found in registry. Did you forget to register a variable?"
            )
        return variable.set(value)       

    def set_at_time(self, key: str, time: datetime, value: ArrayLike) -> None:
        """
        Helper function to find variable by key and call the set_at_time method.
        """
        variable: Variable | None = None
        try:
            variable = self._registry[key]
        except KeyError:
            raise ValueError(
                f"Variable {key} not found in registry. Did you forget to register a variable?"
            )
        return variable.set_at_time(time, value)

    def __contains__(self, key: str) -> bool:
        return key in self._registry
