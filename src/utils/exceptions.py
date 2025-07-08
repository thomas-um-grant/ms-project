class RelationalDBError(Exception):
    """Exception raised for errors in the relational database operations."""

    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.error_code = error_code or "000"  # Default error code if not provided

    def __str__(self):
        return f"{self.args[0]} (Error Code: {self.error_code})"
