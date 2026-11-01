"""Cookidoo API exceptions."""

# Longest excerpt of a response body kept on an exception. Enough to read the
# server's validation message, short enough to keep logs and tracebacks sane.
BODY_EXCERPT_LENGTH = 500


class CookidooException(Exception):
    """General exception occurred."""


class CookidooConfigException(CookidooException):
    """When the config is invalid."""


class CookidooAuthException(CookidooException):
    """When an authentication error is encountered."""


class CookidooParseException(CookidooException):
    """When data could not be parsed."""


class CookidooRequestException(CookidooException):
    """When a request returns an error.

    Attributes
    ----------
    status
        The HTTP status of the response, or ``None`` when no response was
        received (timeouts, transport errors).
    body
        An excerpt of the response body, so a server-side validation message
        (e.g. a rejected recipe payload) is available to the caller.

    """

    status: int | None
    body: str | None

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str | None = None,
    ) -> None:
        """Init the exception with an optional status and body excerpt."""
        super().__init__(message)
        self.status = status
        self.body = body[:BODY_EXCERPT_LENGTH] if body else None

    def __str__(self) -> str:
        """Render the message with the status and body excerpt when known."""
        text = super().__str__()
        if self.status is not None:
            text = f"{text} [status {self.status}]"
        if self.body:
            text = f"{text}: {self.body}"
        return text


class CookidooResponseException(CookidooParseException):
    """When a response could not be parsed (not JSON, e.g. an HTML error page)."""


class CookidooUnavailableException(CookidooRequestException):
    """When the network or server is not available.

    Raised for 5xx and 429 responses, timeouts and transport errors: the
    request may succeed later, and the credentials are not at fault.
    """
