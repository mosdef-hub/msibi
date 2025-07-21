class PotentialError(Exception):
    """Base class for exceptions related to potential operations."""

    pass


class PotentialNotOptimizedError(PotentialError):
    """Exception raised when attempting an operation that requires optimization to be enabled."""

    def __init__(self, operation: str):
        self.message = f"This Force isn't set to be optimized; you can't {operation}."
        super().__init__(self.message)


class PotentialImmutableError(PotentialError):
    """Exception raised when attempting an operation that requires a muttable (table) potential."""

    def __init__(self, operation: str):
        self.message = f"This Force isn't muttable (table); you can't {operation}."
        super().__init__(self.message)
