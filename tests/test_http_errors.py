"""HTTP error classification.

Errors are classified by status, carry the server's answer, and a 401 on an API
call is retried once after a forced refresh.
"""

from datetime import date
from http import HTTPStatus
import json
import logging
import re
from typing import Any

from aiohttp import ClientConnectionError, ClientSession
from aioresponses import CallbackResult, aioresponses
import pytest

from cookidoo_api import (
    Cookidoo,
    CookidooAuthData,
    CookidooAuthException,
    CookidooParseException,
    CookidooRequestException,
    CookidooResponseException,
    CookidooUnavailableException,
)
from cookidoo_api.const import CIAM_LOGIN_SRV_URL, OIDC_DISCOVERY_URL
from cookidoo_api.exceptions import BODY_EXCERPT_LENGTH
from tests.responses import (
    COOKIDOO_TEST_LOGIN_PAGE_HTML,
    COOKIDOO_TEST_OIDC_DISCOVERY,
    COOKIDOO_TEST_REFRESHED_TOKEN_RESPONSE,
    COOKIDOO_TEST_RESPONSE_CALENDAR_WEEK_WITH_CUSTOM_RECIPES,
    COOKIDOO_TEST_RESPONSE_USER_INFO,
)

TOKEN_ENDPOINT = COOKIDOO_TEST_OIDC_DISCOVERY["token_endpoint"]
AUTHZ_RE = re.compile(
    re.escape(COOKIDOO_TEST_OIDC_DISCOVERY["authorization_endpoint"]) + r".*"
)
PROFILE_URL = "https://cookidoo.ch/community/profile/de-CH"
RECIPE_URL = "https://cookidoo.ch/recipes/recipe/de-CH/r123"
VALID = CookidooAuthData("valid", "ref", 9999999999.0)


def _logged_in(cookidoo: Cookidoo) -> None:
    cookidoo.apply_auth_data(VALID)


def _count(mocked: aioresponses, method: str, url: str) -> int:
    """Count the recorded requests for a method and URL (query ignored)."""
    return sum(
        len(calls)
        for (seen_method, seen_url), calls in mocked.requests.items()
        if seen_method.upper() == method and str(seen_url).startswith(url)
    )


# -- exception shape --------------------------------------------------------


def test_request_exception_carries_status_and_body_excerpt() -> None:
    """Status and a truncated body are attached and rendered in str()."""
    error = CookidooRequestException("Op failed.", status=422, body="x" * 1000)
    assert error.status == 422
    assert error.body is not None
    assert len(error.body) == BODY_EXCERPT_LENGTH
    assert str(error).startswith("Op failed. [status 422]: xxx")


def test_request_exception_without_details_renders_plainly() -> None:
    """No status and no body: the message is rendered as is."""
    error = CookidooRequestException("Op failed.", body="")
    assert error.status is None
    assert error.body is None
    assert str(error) == "Op failed."


def test_exception_hierarchy_is_backward_compatible() -> None:
    """Callers catching the broad classes keep catching the new ones."""
    assert issubclass(CookidooUnavailableException, CookidooRequestException)
    assert issubclass(CookidooResponseException, CookidooParseException)


# -- API requests -----------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        HTTPStatus.TOO_MANY_REQUESTS,
        HTTPStatus.INTERNAL_SERVER_ERROR,
        HTTPStatus.BAD_GATEWAY,
        HTTPStatus.SERVICE_UNAVAILABLE,
    ],
)
async def test_transient_statuses_are_unavailable(
    mocked: aioresponses, cookidoo: Cookidoo, status: HTTPStatus
) -> None:
    """5xx and 429 raise the unavailable subclass with the status attached."""
    _logged_in(cookidoo)
    mocked.get(PROFILE_URL, status=status, body="upstream down")

    with pytest.raises(CookidooUnavailableException) as caught:
        await cookidoo.get_user_info()
    assert caught.value.status == status
    assert caught.value.body == "upstream down"
    assert isinstance(caught.value, CookidooRequestException)


@pytest.mark.parametrize(
    "status",
    [HTTPStatus.NOT_FOUND, HTTPStatus.CONFLICT, HTTPStatus.UNPROCESSABLE_ENTITY],
)
async def test_client_errors_keep_status_and_server_message(
    mocked: aioresponses, cookidoo: Cookidoo, status: HTTPStatus
) -> None:
    """A 404/409/422 is a plain request error carrying the server's message."""
    _logged_in(cookidoo)
    mocked.get(
        PROFILE_URL, status=status, payload={"message": "instructions[2] invalid"}
    )

    with pytest.raises(CookidooRequestException) as caught:
        await cookidoo.get_user_info()
    assert not isinstance(caught.value, CookidooUnavailableException)
    assert caught.value.status == status
    assert caught.value.body is not None
    assert "instructions[2] invalid" in caught.value.body
    assert f"[status {int(status)}]" in str(caught.value)


@pytest.mark.parametrize("error", [TimeoutError(), ClientConnectionError()])
async def test_transport_failures_are_unavailable(
    mocked: aioresponses, cookidoo: Cookidoo, error: Exception
) -> None:
    """Timeouts and dropped connections are transient and carry no status."""
    _logged_in(cookidoo)
    mocked.get(PROFILE_URL, exception=error)

    with pytest.raises(CookidooUnavailableException) as caught:
        await cookidoo.get_user_info()
    assert caught.value.status is None


async def test_json_is_parsed_whatever_the_content_type(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """A JSON body served as text/plain is still parsed."""
    _logged_in(cookidoo)
    mocked.get(
        PROFILE_URL,
        status=HTTPStatus.OK,
        body=json.dumps(COOKIDOO_TEST_RESPONSE_USER_INFO),
        content_type="text/plain",
    )

    user = await cookidoo.get_user_info()
    assert user.id


async def test_html_page_is_a_response_parse_error(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """A 200 HTML page (Cloudflare, maintenance) is a parse error, not transport."""
    _logged_in(cookidoo)
    mocked.get(
        PROFILE_URL,
        status=HTTPStatus.OK,
        body="<html><body>Just a moment...</body></html>",
        content_type="text/html",
    )

    with pytest.raises(CookidooResponseException) as caught:
        await cookidoo.get_user_info()
    assert isinstance(caught.value, CookidooParseException)


async def test_response_body_is_only_logged_at_debug_and_truncated(
    mocked: aioresponses,
    cookidoo: Cookidoo,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bodies reach the log only at DEBUG, and never in full."""
    _logged_in(cookidoo)
    mocked.get(
        PROFILE_URL, status=HTTPStatus.OK, payload=COOKIDOO_TEST_RESPONSE_USER_INFO
    )
    with caplog.at_level(logging.INFO, logger="cookidoo_api.cookidoo"):
        await cookidoo.get_user_info()
    assert "Response from" not in caplog.text

    long_body = "y" * 5000
    mocked.get(PROFILE_URL, status=HTTPStatus.BAD_GATEWAY, body=long_body)
    with (
        caplog.at_level(logging.DEBUG, logger="cookidoo_api.cookidoo"),
        pytest.raises(CookidooUnavailableException),
    ):
        await cookidoo.get_user_info()
    logged = [r for r in caplog.records if r.getMessage().startswith("Response from")]
    assert len(logged) == 1
    assert "yyyy" in logged[0].getMessage()
    assert long_body not in logged[0].getMessage()


# -- 401 handling -----------------------------------------------------------


async def test_401_is_retried_once_after_a_forced_refresh(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """A revoked access token is refreshed and the request replayed once."""
    _logged_in(cookidoo)
    seen_tokens: list[str] = []

    def _profile(url: Any, **kwargs: Any) -> CallbackResult:
        token = kwargs["headers"]["Authorization"]
        seen_tokens.append(token)
        if token == "Bearer valid":
            return CallbackResult(status=HTTPStatus.UNAUTHORIZED, payload={})
        return CallbackResult(
            status=HTTPStatus.OK, payload=COOKIDOO_TEST_RESPONSE_USER_INFO
        )

    mocked.get(PROFILE_URL, callback=_profile, repeat=True)
    mocked.get(OIDC_DISCOVERY_URL, payload=COOKIDOO_TEST_OIDC_DISCOVERY)
    mocked.post(TOKEN_ENDPOINT, payload=COOKIDOO_TEST_REFRESHED_TOKEN_RESPONSE)

    user = await cookidoo.get_user_info()

    assert user.id
    assert seen_tokens == ["Bearer valid", "Bearer refreshed-access-token"]
    assert cookidoo.auth_data is not None
    assert cookidoo.auth_data.access_token == "refreshed-access-token"


async def test_401_after_a_successful_refresh_is_an_auth_error(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """Still 401 with a fresh token: give up, exactly one refresh was spent."""
    _logged_in(cookidoo)
    mocked.get(PROFILE_URL, status=HTTPStatus.UNAUTHORIZED, repeat=True)
    mocked.get(OIDC_DISCOVERY_URL, payload=COOKIDOO_TEST_OIDC_DISCOVERY)
    mocked.post(TOKEN_ENDPOINT, payload=COOKIDOO_TEST_REFRESHED_TOKEN_RESPONSE)

    with pytest.raises(CookidooAuthException, match="authorization failure"):
        await cookidoo.get_user_info()
    assert _count(mocked, "POST", TOKEN_ENDPOINT) == 1
    assert _count(mocked, "GET", PROFILE_URL) == 2


async def test_401_with_a_failing_refresh_is_an_auth_error_with_cause(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """The refresh failing keeps the 401 as the verdict, with the cause kept."""
    _logged_in(cookidoo)
    mocked.get(PROFILE_URL, status=HTTPStatus.UNAUTHORIZED)
    mocked.get(OIDC_DISCOVERY_URL, payload=COOKIDOO_TEST_OIDC_DISCOVERY)
    mocked.post(TOKEN_ENDPOINT, status=HTTPStatus.SERVICE_UNAVAILABLE)

    with pytest.raises(CookidooAuthException, match="could not be refreshed") as c:
        await cookidoo.get_user_info()
    assert isinstance(c.value.__cause__, CookidooUnavailableException)


async def test_401_without_a_refresh_token_is_not_retried(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """With nothing to refresh with, a 401 is reported straight away."""
    mocked.get(PROFILE_URL, status=HTTPStatus.UNAUTHORIZED, payload={})

    with pytest.raises(CookidooAuthException):
        await cookidoo.get_user_info()
    assert _count(mocked, "GET", PROFILE_URL) == 1


# -- token endpoint ---------------------------------------------------------


@pytest.mark.parametrize(
    "status", [HTTPStatus.TOO_MANY_REQUESTS, HTTPStatus.SERVICE_UNAVAILABLE]
)
async def test_refresh_server_errors_are_not_auth_failures(
    mocked: aioresponses, cookidoo: Cookidoo, status: HTTPStatus
) -> None:
    """A 503 or 429 from the token endpoint does not force a re-login."""
    _logged_in(cookidoo)
    mocked.get(OIDC_DISCOVERY_URL, payload=COOKIDOO_TEST_OIDC_DISCOVERY)
    mocked.post(TOKEN_ENDPOINT, status=status, body="try later")

    with pytest.raises(CookidooUnavailableException) as caught:
        await cookidoo.refresh()
    assert not isinstance(caught.value, CookidooAuthException)
    assert caught.value.status == status
    assert caught.value.body == "try later"
    # The refresh token is kept for a later attempt.
    assert cookidoo.auth_data == VALID


@pytest.mark.parametrize("status", [HTTPStatus.BAD_REQUEST, HTTPStatus.UNAUTHORIZED])
async def test_refresh_rejected_grant_is_an_auth_failure(
    mocked: aioresponses, cookidoo: Cookidoo, status: HTTPStatus
) -> None:
    """400 (invalid_grant) and 401 mean the refresh token is no longer valid."""
    _logged_in(cookidoo)
    mocked.get(OIDC_DISCOVERY_URL, payload=COOKIDOO_TEST_OIDC_DISCOVERY)
    mocked.post(TOKEN_ENDPOINT, status=status)

    with pytest.raises(CookidooAuthException, match="Token refresh failed"):
        await cookidoo.refresh()


async def test_refresh_non_json_token_response_is_a_parse_error(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """A 200 that is not JSON (HTML page) is a parse error, not a KeyError."""
    _logged_in(cookidoo)
    mocked.get(OIDC_DISCOVERY_URL, payload=COOKIDOO_TEST_OIDC_DISCOVERY)
    mocked.post(TOKEN_ENDPOINT, status=HTTPStatus.OK, body="<html>nope</html>")

    with pytest.raises(CookidooParseException):
        await cookidoo.refresh()


async def test_refresh_non_object_token_response_is_a_parse_error(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """A JSON list where an object is expected is a parse error."""
    _logged_in(cookidoo)
    mocked.get(OIDC_DISCOVERY_URL, payload=COOKIDOO_TEST_OIDC_DISCOVERY)
    mocked.post(TOKEN_ENDPOINT, status=HTTPStatus.OK, payload=["not", "an", "object"])

    with pytest.raises(CookidooParseException):
        await cookidoo.refresh()


# -- discovery --------------------------------------------------------------


@pytest.mark.parametrize("missing", ["authorization_endpoint", "token_endpoint"])
async def test_incomplete_discovery_is_a_parse_error_on_refresh(
    mocked: aioresponses, cookidoo: Cookidoo, missing: str
) -> None:
    """A discovery document without an endpoint is reported, not a KeyError."""
    _logged_in(cookidoo)
    document = {k: v for k, v in COOKIDOO_TEST_OIDC_DISCOVERY.items() if k != missing}
    mocked.get(OIDC_DISCOVERY_URL, payload=document)

    with pytest.raises(CookidooParseException, match="incomplete"):
        await cookidoo.refresh()


async def test_incomplete_discovery_is_a_parse_error_on_login(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """Same on the password login path."""
    mocked.get(OIDC_DISCOVERY_URL, payload={"issuer": "x"})

    with pytest.raises(CookidooParseException, match="incomplete"):
        await cookidoo.login()


async def test_discovery_server_error_is_unavailable(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """A 503 on the discovery document is transient."""
    mocked.get(OIDC_DISCOVERY_URL, status=HTTPStatus.SERVICE_UNAVAILABLE)

    with pytest.raises(CookidooUnavailableException):
        await cookidoo.login()


async def test_discovery_html_is_a_parse_error(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """A 200 HTML page instead of the discovery JSON is a parse error."""
    mocked.get(OIDC_DISCOVERY_URL, status=HTTPStatus.OK, body="<html></html>")

    with pytest.raises(CookidooParseException):
        await cookidoo.login()


# -- credential submission --------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (HTTPStatus.FORBIDDEN, CookidooRequestException),
        (HTTPStatus.SERVICE_UNAVAILABLE, CookidooUnavailableException),
    ],
)
async def test_credential_post_errors_are_not_invalid_credentials(
    mocked: aioresponses,
    cookidoo: Cookidoo,
    status: HTTPStatus,
    expected: type[CookidooRequestException],
) -> None:
    """A blocked (403) or failing (503) login service is reported as such."""
    mocked.get(OIDC_DISCOVERY_URL, payload=COOKIDOO_TEST_OIDC_DISCOVERY)
    mocked.get(AUTHZ_RE, status=HTTPStatus.OK, body=COOKIDOO_TEST_LOGIN_PAGE_HTML)
    mocked.post(CIAM_LOGIN_SRV_URL, status=status, body="blocked")

    with pytest.raises(expected) as caught:
        await cookidoo.login()
    assert not isinstance(caught.value, CookidooAuthException)
    assert caught.value.status == status
    assert "invalid credentials" not in str(caught.value)


async def test_login_page_403_is_a_request_error(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """A 403 on the login page (bot protection) is not an auth failure."""
    mocked.get(OIDC_DISCOVERY_URL, payload=COOKIDOO_TEST_OIDC_DISCOVERY)
    mocked.get(AUTHZ_RE, status=HTTPStatus.FORBIDDEN, body="Just a moment")

    with pytest.raises(CookidooRequestException) as caught:
        await cookidoo.login()
    assert not isinstance(caught.value, CookidooAuthException)
    assert caught.value.status == HTTPStatus.FORBIDDEN
    assert caught.value.body == "Just a moment"


# -- calendar resolution ----------------------------------------------------


async def test_calendar_resolution_raises_on_transient_failure(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """An outage while resolving a planned custom recipe is not a missing recipe."""
    mocked.get(
        re.compile(r"https://cookidoo\.ch/planning/de-CH/api/my-week/.*"),
        payload=COOKIDOO_TEST_RESPONSE_CALENDAR_WEEK_WITH_CUSTOM_RECIPES,
    )
    mocked.get(
        re.compile(r"https://cookidoo\.ch/created-recipes/.*"),
        status=HTTPStatus.SERVICE_UNAVAILABLE,
        repeat=True,
    )

    with pytest.raises(CookidooUnavailableException):
        await cookidoo.get_recipes_in_calendar_week(
            date(2025, 3, 3), resolve_custom_recipes=True
        )


async def test_calendar_resolution_raises_on_malformed_recipe(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """A recipe the API returns but cannot be parsed is surfaced too."""
    mocked.get(
        re.compile(r"https://cookidoo\.ch/planning/de-CH/api/my-week/.*"),
        payload=COOKIDOO_TEST_RESPONSE_CALENDAR_WEEK_WITH_CUSTOM_RECIPES,
    )
    mocked.get(
        re.compile(r"https://cookidoo\.ch/created-recipes/.*"),
        status=HTTPStatus.OK,
        payload={"unexpected": True},
        repeat=True,
    )

    with pytest.raises(CookidooParseException):
        await cookidoo.get_recipes_in_calendar_week(
            date(2025, 3, 3), resolve_custom_recipes=True
        )


async def test_sessions_are_not_closed_by_the_client(session: ClientSession) -> None:
    """Sanity: the fixtures hand over an open session, the client never closes it."""
    client = Cookidoo(session)
    assert client._session is session
    assert not session.closed
