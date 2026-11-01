"""Credentials must stay out of repr/logs, token files must be private."""

from dataclasses import asdict
import logging
from pathlib import Path
import stat
import sys

from aiohttp import ClientSession
import pytest

from cookidoo_api import Cookidoo, CookidooAuthData, CookidooConfig
from cookidoo_api.types import CookidooLocalizationConfig


def test_config_repr_hides_password() -> None:
    """repr() and str() of the config never contain the password."""
    cfg = CookidooConfig(email="me@example.com", password="SENSITIVE_PASSWORD")
    for rendered in (repr(cfg), str(cfg), f"{cfg}"):
        assert "SENSITIVE_PASSWORD" not in rendered
        assert "me@example.com" in rendered
    # The value itself is untouched.
    assert cfg.password == "SENSITIVE_PASSWORD"


def test_config_logging_hides_password(caplog: pytest.LogCaptureFixture) -> None:
    """A config logged with %s or %r does not leak the password."""
    cfg = CookidooConfig(password="SENSITIVE_PASSWORD")
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("test").debug("cfg: %s / %r", cfg, cfg)
    assert "SENSITIVE_PASSWORD" not in caplog.text


def test_auth_data_repr_hides_tokens() -> None:
    """repr() of auth data hides both tokens but keeps the expiry."""
    auth = CookidooAuthData(
        access_token="SENSITIVE_ACCESS",
        refresh_token="SENSITIVE_REFRESH",
        expires_at=1234.5,
    )
    rendered = repr(auth)
    assert "SENSITIVE_" not in rendered
    assert "1234.5" in rendered
    # Persistence paths still see the full data.
    assert asdict(auth)["refresh_token"] == "SENSITIVE_REFRESH"
    assert vars(auth)["access_token"] == "SENSITIVE_ACCESS"


def test_auth_data_positional_construction_still_works() -> None:
    """Field order is unchanged for callers that use positional arguments."""
    auth = CookidooAuthData("a", "r", 1.0)
    assert (auth.access_token, auth.refresh_token, auth.expires_at) == ("a", "r", 1.0)


async def test_default_config_is_not_shared_between_clients() -> None:
    """Each client built without a config gets its own config object."""
    async with ClientSession() as session:
        first = Cookidoo(session)
        second = Cookidoo(session)
        assert first._cfg is not second._cfg
        first._cfg.localization = CookidooLocalizationConfig(
            country_code="fr",
            language="fr-FR",
            url="https://cookidoo.fr/foundation/fr-FR",
        )
        assert second._cfg.localization.country_code == "ch"


async def test_explicit_config_is_used_as_is() -> None:
    """A config passed explicitly is the one the client holds."""
    cfg = CookidooConfig(email="me@example.com")
    async with ClientSession() as session:
        client = Cookidoo(session, cfg)
        assert client._cfg is cfg


@pytest.mark.parametrize("pre_existing_mode", [None, 0o644, 0o666])
async def test_save_token_writes_private_file(
    tmp_path: Path, pre_existing_mode: int | None
) -> None:
    """The token file is owner-only, for new files and for existing wider ones."""
    token_file = tmp_path / "token.json"
    if pre_existing_mode is not None:
        token_file.write_text("old", encoding="utf-8")
        token_file.chmod(pre_existing_mode)

    async with ClientSession() as session:
        client = Cookidoo(session)
        client.apply_auth_data(
            CookidooAuthData(
                access_token="SENSITIVE_ACCESS",
                refresh_token="SENSITIVE_REFRESH",
                expires_at=9999999999.0,
            )
        )
        client.save_token(token_file)

        # Windows has no POSIX permission bits (chmod only toggles read-only),
        # so the owner-only mode is asserted on POSIX platforms only.
        if sys.platform != "win32":
            assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
        assert '"refresh_token": "SENSITIVE_REFRESH"' in token_file.read_text(
            encoding="utf-8"
        )

        fresh = Cookidoo(session)
        fresh.load_token(token_file)
        assert fresh.auth_data == client.auth_data
