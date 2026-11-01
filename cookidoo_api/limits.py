"""Validation of guided-cooking settings against what Cookidoo accepts.

Wrong types raise ``TypeError``, out-of-range values ``ValueError``; the client
reports both as ``ValueError`` with the step number.

Shared by step-level settings and annotations, so a speed of 11 or a
temperature of 130 is rejected locally with a clear message instead of as an
opaque server-side rejection of the whole recipe.
"""

from __future__ import annotations

from enum import StrEnum

from cookidoo_api.types import (
    ThermomixBrowningPower,
    ThermomixDirection,
    ThermomixMode,
    ThermomixSpeed,
    ThermomixSteamingAccessory,
    ThermomixTemperature,
)

# 99 h 59 min, the longest time the appliance takes.
MAX_TIME_S = 99 * 3600 + 59 * 60
# Browning mode: 140 to 160 degrees by 5, at most 30 minutes.
BROWNING_TEMPERATURES = frozenset({"140", "145", "150", "155", "160"})
BROWNING_MAX_TIME_S = 30 * 60
# Steaming mode: speed up to 5 (no "soft" restriction beyond that).
STEAMING_MAX_SPEED = 5.0
# Warm-up mode takes no time (Cookidoo drops it).
VAROMA = ThermomixTemperature.VAROMA.value

_SPEEDS = frozenset(member.value for member in ThermomixSpeed)
_TEMPERATURES = frozenset(member.value for member in ThermomixTemperature)
_MODES = frozenset(member.value for member in ThermomixMode)
_DIRECTIONS = frozenset(member.value for member in ThermomixDirection)
_POWERS = frozenset(member.value for member in ThermomixBrowningPower)
_ACCESSORIES = frozenset(member.value for member in ThermomixSteamingAccessory)


def _plain(value: object) -> object:
    """Unwrap a StrEnum member to its value."""
    return value.value if isinstance(value, StrEnum) else value


def normalize_speed(value: object) -> str:
    """Return a speed as the string Cookidoo accepts, or raise ValueError.

    Accepts ``"soft"``, strings from ``"0.5"`` to ``"10"`` by halves, and
    numbers (``3`` and ``3.0`` become ``"3"``, ``3.5`` stays ``"3.5"``).
    """
    plain = _plain(value)
    if isinstance(plain, bool):
        raise TypeError(f"Invalid speed: {plain!r}.")
    if isinstance(plain, int | float):
        text = str(int(plain)) if float(plain).is_integer() else str(plain)
    elif isinstance(plain, str):
        text = plain.strip().lower()
        if text.endswith(".0"):
            text = text[:-2]
    else:
        raise TypeError(f"Invalid speed: {plain!r}.")
    if text not in _SPEEDS:
        raise ValueError(
            f"Invalid speed {plain!r}: expected 'soft' or 0.5 to 10 by halves."
        )
    return text


def normalize_temperature(value: object, *, browning: bool = False) -> str:
    """Return a temperature as the string Cookidoo accepts, or raise ValueError.

    Accepts ``"varoma"`` (any case) and the values of ``ThermomixTemperature``
    as strings or numbers (37, 40 to 95 by 5, 98, 100 to 120 by 5). With
    ``browning`` the range 140 to 160 by 5 is accepted too.
    """
    plain = _plain(value)
    if isinstance(plain, bool):
        raise TypeError(f"Invalid temperature: {plain!r}.")
    if isinstance(plain, int | float):
        if not float(plain).is_integer():
            raise ValueError(f"Invalid temperature {plain!r}: must be a whole number.")
        text = str(int(plain))
    elif isinstance(plain, str):
        text = plain.strip().lower()
    else:
        raise TypeError(f"Invalid temperature: {plain!r}.")
    if text == VAROMA:
        return VAROMA
    if text in _TEMPERATURES or (browning and text in BROWNING_TEMPERATURES):
        return text
    raise ValueError(
        f"Invalid temperature {plain!r}: expected 37, 40 to 95 by 5, 98, "
        "100 to 120 by 5 or 'varoma'"
        + (", or 140 to 160 by 5 in browning mode." if browning else ".")
    )


def validate_time(value: int, *, max_seconds: int = MAX_TIME_S) -> int:
    """Validate a duration in seconds, or raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Invalid time {value!r}: must be a whole number of seconds.")
    if value < 0:
        raise ValueError("Time must not be negative.")
    if value > max_seconds:
        raise ValueError(f"Time {value} s exceeds the maximum of {max_seconds} s.")
    return value


def validate_direction(value: object) -> str:
    """Validate a rotation direction, or raise ValueError."""
    plain = _plain(value)
    if not isinstance(plain, str) or plain.upper() not in _DIRECTIONS:
        raise ValueError(f"Invalid direction {plain!r}: expected CW or CCW.")
    return plain.upper()


def validate_mode(
    mode: object,
    *,
    time: int | None,
    temperature: str | None,
    speed: str | None,
    power: object,
    accessory: object,
) -> str:
    """Validate the fields of a MODE annotation against the mode's limits.

    ``temperature`` and ``speed`` are the already normalized strings. Returns
    the plain mode name. Raises ValueError on an unknown mode or a field the
    mode does not take or takes only within a range.
    """
    plain_mode = _plain(mode)
    if not isinstance(plain_mode, str) or plain_mode not in _MODES:
        raise ValueError(
            f"Invalid mode {plain_mode!r}: expected one of {sorted(_MODES)}."
        )
    plain_power = _plain(power)
    plain_accessory = _plain(accessory)

    if plain_power is not None and plain_power not in _POWERS:
        raise ValueError(f"Invalid browning power {plain_power!r}.")
    if plain_accessory is not None and plain_accessory not in _ACCESSORIES:
        raise ValueError(f"Invalid steaming accessory {plain_accessory!r}.")

    if (
        plain_mode == ThermomixMode.BROWNING
        and time is not None
        and time > BROWNING_MAX_TIME_S
    ):
        raise ValueError("Browning time must not exceed 30 minutes.")

    if plain_mode == ThermomixMode.STEAMING and speed not in (None, "soft"):
        if float(speed) > STEAMING_MAX_SPEED:
            raise ValueError("Steaming speed must not exceed 5.")

    return plain_mode
