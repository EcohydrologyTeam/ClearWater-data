from __future__ import annotations

from clearwater_data.variables.base import Variable
from clearwater_data import ArrayLike
from datetime import datetime, timedelta

import warnings


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
    
    def get_space_dimension(self, key: str) -> str | list[str] | None:
        try:
            variable = self._registry[key]
        except KeyError:
            raise ValueError(f"Variable {key} not found.")
        return variable.space_dimension

    def get_variable(self, key: str) -> Variable:
        """
        Retrieve a variable by key.
        """
        try:
            variable = self._registry[key]
        except KeyError:
            raise ValueError(f"Variable {key} not found.")
        return variable

    # TODO: We should chat about this method.
    # It's odd that this returns the underlying data instead of the variable itself, but it is a common use case to want to get the data out of the variable,
    # I'm thinking this should be deprecated in favor of get_variable & get_data method.
    def get(self, key: str) -> ArrayLike:
        """
        Retrieve a variable by key.
        """
        # Phase H-14 (2026-05-21): the deprecation message previously
        # referenced a ``get_data`` method that does not exist on
        # ``Variable`` (the actual method is ``get``). The corrected
        # guidance is ``registry.get_variable(key).get()``. Also adds
        # ``stacklevel=2`` so the warning is annotated at the caller's
        # line, not registry.py.
        warnings.warn(
            "VariableRegistry.get is deprecated; "
            "use get_variable(key).get() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            variable = self._registry[key]
        except KeyError:
            raise ValueError(f"Variable {key} not found.")

        return variable.get()

    def get_at_time(
        self,
        key: str,
        time: datetime,
        tolerance: timedelta | None = None,
    ) -> ArrayLike:
        """Retrieve a variable's value at a specific time.

        Phase H-13 (2026-05-21): the optional ``tolerance`` kwarg
        threads through to ``Variable.get_at_time``; when non-None,
        the underlying selection uses ``method='nearest'`` with the
        given tolerance. Default ``None`` preserves exact-match
        behaviour.
        """
        variable: Variable | None = None
        try:
            variable = self._registry[key]
        except KeyError:
            raise ValueError(
                f"Variable {key} not found in registry. Did you forget to register a variable?"
            )

        return variable.get_at_time(time, tolerance=tolerance)

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
