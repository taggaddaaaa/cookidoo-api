"""Lossless custom recipes.

Custom recipes keep what the API sent, annotations land where they belong,
and settings are validated before anything is sent.
"""

from http import HTTPStatus
from typing import Any, cast

from aioresponses import aioresponses
import pytest
from yarl import URL

from cookidoo_api import (
    Cookidoo,
    CookidooCreateCustomRecipe,
    CookidooCustomAnnotation,
    CookidooIngredientAnnotation,
    CookidooInstruction,
    CookidooModeAnnotation,
    CookidooParseException,
    CookidooRequestException,
    CookidooStepSettings,
    CookidooTemperatureSetting,
    CookidooTTSAnnotation,
    CookidooUpdateCustomRecipe,
    ThermomixMode,
    ThermomixSpeed,
)
from cookidoo_api.helpers import (
    _duration_to_seconds,
    _parse_custom_recipe_annotation,
    _raw_instructions,
    cookidoo_custom_recipe_from_json,
)
from cookidoo_api.limits import (
    normalize_speed,
    normalize_temperature,
    validate_mode,
    validate_time,
)
from cookidoo_api.raw_types import CustomRecipeJSON

RECIPE_URL = "https://cookidoo.ch/created-recipes/de-CH/{id}"
CREATE_URL = "https://cookidoo.ch/created-recipes/de-CH"


def _recipe(recipe_id: str, instructions: list[Any]) -> CustomRecipeJSON:
    """Build a minimal custom recipe response around the given instructions."""
    return cast(
        CustomRecipeJSON,
        {
            "recipeId": recipe_id,
            "recipeContent": {
                "name": "Original",
                "prepTime": 60,
                "totalTime": 180,
                "tools": ["TM7"],
                "yield": {"value": 2, "unitText": "portion"},
                "ingredients": [{"type": "INGREDIENT", "text": "1 onion"}],
                "instructions": instructions,
            },
        },
    )


def _patch_payload(mocked: aioresponses, url: str) -> dict[str, Any]:
    return cast(dict[str, Any], mocked.requests[("patch", URL(url))][0].kwargs["json"])


# -- lossless updates -------------------------------------------------------

RICH_STEP = {
    "type": "STEP",
    "text": "Cook 2 min/120°C/speed 3.5, then knead",
    "time": 120,
    "temperature": 120,
    "speed": 3.5,
    "annotations": [
        {
            "type": "MODE",
            "name": "dough",
            "data": {"time": 120, "foo": "bar"},
            "position": {"offset": 32, "length": 5},
        },
        {
            "type": "TTS",
            "data": {"time": 120, "speed": 3.5, "futureField": {"x": 1}},
            "position": {"offset": 5, "length": 20},
        },
        {
            "type": "FUTURE_TYPE",
            "data": {"anything": [1, 2]},
            "position": {"offset": 0, "length": 4},
        },
    ],
}


async def test_rename_sends_instructions_back_verbatim(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """Changing the name re-sends the instructions exactly as received."""
    url = RECIPE_URL.format(id="rich")
    mocked.get(url, payload=_recipe("rich", [RICH_STEP]), repeat=True)
    mocked.patch(url, status=HTTPStatus.NO_CONTENT)

    await cookidoo.update_custom_recipe("rich", CookidooUpdateCustomRecipe(name="New"))

    payload = _patch_payload(mocked, url)
    assert payload["name"] == "New"
    assert payload["instructions"] == [RICH_STEP]
    # The MODE annotation keeps its unknown field, the TTS its numeric speed.
    assert payload["instructions"][0]["annotations"][0]["data"] == {
        "time": 120,
        "foo": "bar",
    }
    assert payload["instructions"][0]["annotations"][1]["data"]["speed"] == 3.5


async def test_rename_with_legacy_string_instructions(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """Plain string instructions (legacy shape) are echoed as STEP items."""
    url = RECIPE_URL.format(id="legacy")
    mocked.get(url, payload=_recipe("legacy", ["Chop", "Cook"]), repeat=True)
    mocked.patch(url, status=HTTPStatus.NO_CONTENT)

    await cookidoo.update_custom_recipe(
        "legacy", CookidooUpdateCustomRecipe(name="New")
    )

    assert _patch_payload(mocked, url)["instructions"] == [
        {"type": "STEP", "text": "Chop"},
        {"type": "STEP", "text": "Cook"},
    ]


async def test_edited_instructions_replace_the_raw_ones(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """Passing instructions explicitly rebuilds them from the model."""
    url = RECIPE_URL.format(id="rich")
    mocked.get(url, payload=_recipe("rich", [RICH_STEP]), repeat=True)
    mocked.patch(url, status=HTTPStatus.NO_CONTENT)

    await cookidoo.update_custom_recipe(
        "rich",
        CookidooUpdateCustomRecipe(
            instructions=[CookidooInstruction("Serve", CookidooStepSettings(time=5))]
        ),
    )

    assert _patch_payload(mocked, url)["instructions"] == [
        {"type": "STEP", "text": "Serve", "time": 5}
    ]


async def test_malformed_annotation_does_not_block_a_rename(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """An out-of-range server position no longer breaks every later update."""
    broken = {
        "type": "STEP",
        "text": "Mix",
        "annotations": [
            {
                "type": "TTS",
                "data": {"time": 10},
                "position": {"offset": 40, "length": 3},
            }
        ],
    }
    url = RECIPE_URL.format(id="broken")
    mocked.get(url, payload=_recipe("broken", [broken]), repeat=True)
    mocked.patch(url, status=HTTPStatus.NO_CONTENT)

    await cookidoo.update_custom_recipe(
        "broken", CookidooUpdateCustomRecipe(name="Still fine")
    )

    assert _patch_payload(mocked, url)["instructions"] == [broken]


def test_parsed_annotations_round_trip_extra_fields_and_offsets() -> None:
    """The parsed model keeps unknown fields, numeric speeds and positions."""
    recipe = cookidoo_custom_recipe_from_json(_recipe("rich", [RICH_STEP]))
    step = recipe.instructions[0]
    assert isinstance(step, CookidooInstruction)
    mode, tts, custom = step.annotations
    assert isinstance(mode, CookidooModeAnnotation)
    assert mode.extra == {"foo": "bar"}
    assert mode.offset == 32
    assert isinstance(tts, CookidooTTSAnnotation)
    assert tts.speed == "3.5"
    assert tts.extra == {"futureField": {"x": 1}}
    assert tts.offset == 5
    assert isinstance(custom, CookidooCustomAnnotation)
    assert custom.offset == 0
    assert recipe.raw_instructions == [RICH_STEP]

    # Re-serializing the parsed model (an explicit edit) keeps the extras and
    # the positions, so nothing is lost on that path either.
    serialized = Cookidoo._process_recipe_steps([step], ingredients=[])
    annotations = serialized[0]["annotations"]
    assert annotations[0]["data"] == {"foo": "bar", "time": 120}
    assert annotations[0]["position"] == {"offset": 32, "length": 5}
    assert annotations[1]["data"]["futureField"] == {"x": 1}
    assert annotations[1]["data"]["speed"] == "3.5"


# -- annotation positions ---------------------------------------------------


def test_repeated_slots_get_distinct_offsets() -> None:
    """Two annotations on the same phrase land on two occurrences."""
    text = "Mix 10 sec/speed 5, then mix 10 sec/speed 5 again"
    steps = [
        CookidooInstruction(
            text,
            annotations=[
                CookidooTTSAnnotation("10 sec/speed 5", time=10, speed="5"),
                CookidooTTSAnnotation("10 sec/speed 5", time=10, speed="5"),
            ],
        )
    ]

    result = Cookidoo._process_recipe_steps(steps, ingredients=[])

    offsets = [a["position"]["offset"] for a in result[0]["annotations"]]
    assert offsets == [4, 29]
    assert offsets[0] != offsets[1]


def test_slot_is_matched_on_word_boundaries() -> None:
    """``salt`` is not found inside ``salted``."""
    steps = [
        CookidooInstruction(
            "Add salted butter, then salt",
            annotations=[CookidooIngredientAnnotation("salt", "salt")],
        )
    ]

    result = Cookidoo._process_recipe_steps(steps, ingredients=["salt"])

    assert result[0]["annotations"][0]["position"] == {"offset": 24, "length": 4}


def test_slot_falls_back_to_substring_when_no_whole_word() -> None:
    """A slot that only exists inside a word is still located."""
    steps = [
        CookidooInstruction(
            "Preheat",
            annotations=[CookidooTTSAnnotation("heat", time=60)],
        )
    ]

    result = Cookidoo._process_recipe_steps(steps, ingredients=[])

    assert result[0]["annotations"][0]["position"] == {"offset": 3, "length": 4}


def test_explicit_offset_is_honoured_and_checked() -> None:
    """An explicit offset wins; a wrong one is an error, not a silent move."""
    text = "mix, then mix again"
    steps = [
        CookidooInstruction(
            text, annotations=[CookidooTTSAnnotation("mix", time=10, offset=10)]
        )
    ]
    result = Cookidoo._process_recipe_steps(steps, ingredients=[])
    assert result[0]["annotations"][0]["position"] == {"offset": 10, "length": 3}

    wrong = [
        CookidooInstruction(
            text, annotations=[CookidooTTSAnnotation("mix", time=10, offset=5)]
        )
    ]
    with pytest.raises(ValueError, match="not at offset 5"):
        Cookidoo._process_recipe_steps(wrong, ingredients=[])


def test_offsets_count_utf16_units() -> None:
    """Positions are UTF-16 code units, so an emoji counts for two."""
    text = "🍳 Fry 2 min, then fry 2 min"
    steps = [
        CookidooInstruction(
            text,
            annotations=[
                CookidooTTSAnnotation("2 min", time=120),
                CookidooTTSAnnotation("2 min", time=120),
            ],
        )
    ]

    result = Cookidoo._process_recipe_steps(steps, ingredients=[])

    first, second = (a["position"]["offset"] for a in result[0]["annotations"])
    assert first == 7  # emoji (2) + " Fry " (5)
    assert second == 23
    # And an explicit UTF-16 offset maps back to the same place.
    explicit = Cookidoo._process_recipe_steps(
        [
            CookidooInstruction(
                text, annotations=[CookidooTTSAnnotation("2 min", time=1, offset=23)]
            )
        ],
        ingredients=[],
    )
    assert explicit[0]["annotations"][0]["position"]["offset"] == 23


# -- validation -------------------------------------------------------------


@pytest.mark.parametrize("speed", [11, "11", 0, "0.3", "fast", 10.5])
def test_invalid_step_speed_is_rejected_locally(speed: Any) -> None:
    """A speed outside soft / 0.5 to 10 never reaches the server."""
    steps = [CookidooInstruction("Mix", CookidooStepSettings(speed=speed))]
    with pytest.raises(ValueError, match="Step 1: Invalid speed"):
        Cookidoo._process_recipe_steps(steps, ingredients=[])


@pytest.mark.parametrize("temperature", [130, "130", 36, "99", 140, "hot"])
def test_invalid_step_temperature_is_rejected_locally(temperature: Any) -> None:
    """A temperature Cookidoo does not offer never reaches the server."""
    steps = [CookidooInstruction("Heat", CookidooStepSettings(temperature=temperature))]
    with pytest.raises(ValueError, match="Step 1: Invalid temperature"):
        Cookidoo._process_recipe_steps(steps, ingredients=[])


def test_invalid_annotation_settings_are_rejected_locally() -> None:
    """Annotations go through the same validation as step settings."""
    bad_speed = [
        CookidooInstruction(
            "Mix", annotations=[CookidooTTSAnnotation("Mix", speed="11")]
        )
    ]
    with pytest.raises(ValueError, match="Step 1: Invalid speed"):
        Cookidoo._process_recipe_steps(bad_speed, ingredients=[])

    bad_temperature = [
        CookidooInstruction(
            "Heat",
            annotations=[
                CookidooTTSAnnotation(
                    "Heat", temperature=CookidooTemperatureSetting(130)
                )
            ],
        )
    ]
    with pytest.raises(ValueError, match="Step 1: Invalid temperature"):
        Cookidoo._process_recipe_steps(bad_temperature, ingredients=[])

    too_long = [
        CookidooInstruction(
            "Cook", annotations=[CookidooTTSAnnotation("Cook", time=100 * 3600)]
        )
    ]
    with pytest.raises(ValueError, match="exceeds the maximum"):
        Cookidoo._process_recipe_steps(too_long, ingredients=[])

    bad_direction = [
        CookidooInstruction(
            "Stir", annotations=[CookidooTTSAnnotation("Stir", direction="left")]
        )
    ]
    with pytest.raises(ValueError, match="Invalid direction"):
        Cookidoo._process_recipe_steps(bad_direction, ingredients=[])


def test_browning_temperatures_only_in_browning_mode() -> None:
    """150 degrees is fine when browning, rejected elsewhere."""
    browning = [
        CookidooInstruction(
            "Brown",
            annotations=[
                CookidooModeAnnotation(
                    "Brown",
                    mode=ThermomixMode.BROWNING,
                    time=300,
                    temperature=CookidooTemperatureSetting(150),
                )
            ],
        )
    ]
    result = Cookidoo._process_recipe_steps(browning, ingredients=[])
    assert result[0]["annotations"][0]["data"]["temperature"] == {
        "value": "150",
        "unit": "C",
    }

    too_long = [
        CookidooInstruction(
            "Brown",
            annotations=[
                CookidooModeAnnotation("Brown", mode="browning", time=31 * 60)
            ],
        )
    ]
    with pytest.raises(ValueError, match="30 minutes"):
        Cookidoo._process_recipe_steps(too_long, ingredients=[])

    not_browning = [
        CookidooInstruction(
            "Heat",
            annotations=[
                CookidooTTSAnnotation(
                    "Heat", temperature=CookidooTemperatureSetting(150)
                )
            ],
        )
    ]
    with pytest.raises(ValueError, match="Invalid temperature"):
        Cookidoo._process_recipe_steps(not_browning, ingredients=[])

    unknown_mode = [
        CookidooInstruction(
            "X", annotations=[CookidooModeAnnotation("X", mode="sous_vide")]
        )
    ]
    with pytest.raises(ValueError, match="Invalid mode"):
        Cookidoo._process_recipe_steps(unknown_mode, ingredients=[])


def test_step_settings_normalize_varoma_and_keep_numbers() -> None:
    """Step-level temperatures get the same varoma handling as annotations."""
    steps = [
        CookidooInstruction(
            "Steam",
            CookidooStepSettings(time=600, temperature="Varoma", speed=2),
        ),
        CookidooInstruction("Mix", CookidooStepSettings(speed=ThermomixSpeed.SOFT)),
        CookidooInstruction("Heat", CookidooStepSettings(temperature="100")),
    ]

    result = Cookidoo._process_recipe_steps(steps, ingredients=[])

    assert result[0] == {
        "type": "STEP",
        "text": "Steam",
        "time": 600,
        "temperature": "varoma",
        "speed": 2,
    }
    assert result[1]["speed"] == "soft"
    assert result[2]["temperature"] == "100"


def test_limits_helpers_directly() -> None:
    """The shared validators accept enums, numbers and strings alike."""
    assert normalize_speed(ThermomixSpeed.SPEED_3) == "3"
    assert normalize_speed(3.0) == "3"
    assert normalize_speed("3.5") == "3.5"
    assert normalize_speed(" Soft ") == "soft"
    assert normalize_temperature(100) == "100"
    assert normalize_temperature("VAROMA") == "varoma"
    assert normalize_temperature(160, browning=True) == "160"
    assert validate_time(0) == 0
    with pytest.raises(ValueError, match="negative"):
        validate_time(-1)
    with pytest.raises(TypeError):
        validate_time(cast(int, "10"))
    with pytest.raises(TypeError):
        normalize_speed(None)
    with pytest.raises(TypeError):
        normalize_temperature(True)
    with pytest.raises(ValueError, match="whole number"):
        normalize_temperature(100.5)
    with pytest.raises(ValueError, match="Steaming speed"):
        validate_mode(
            "steaming",
            time=None,
            temperature=None,
            speed="6",
            power=None,
            accessory=None,
        )
    with pytest.raises(ValueError, match="browning power"):
        validate_mode(
            "browning",
            time=None,
            temperature=None,
            speed=None,
            power="max",
            accessory=None,
        )
    with pytest.raises(ValueError, match="steaming accessory"):
        validate_mode(
            "steaming",
            time=None,
            temperature=None,
            speed=None,
            power=None,
            accessory="pot",
        )


def test_malformed_duration_is_a_parse_error(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """A malformed ISO duration is a ValueError (so a parse error for callers)."""
    with pytest.raises(ValueError):
        _duration_to_seconds("not-a-duration")


async def test_malformed_duration_surfaces_as_parse_exception(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """Through the client, the same recipe is a CookidooParseException."""
    url = RECIPE_URL.format(id="bad-duration")
    response = _recipe("bad-duration", [])
    cast(dict[str, Any], response["recipeContent"])["totalTime"] = "PTgarbage"
    mocked.get(url, payload=response)

    with pytest.raises(CookidooParseException):
        await cookidoo.get_custom_recipe("bad-duration")


# -- creation ---------------------------------------------------------------


def _mock_create_with_failing_patch(mocked: aioresponses) -> str:
    recipe_id = "stub-1"
    mocked.post(CREATE_URL, payload={"recipeId": recipe_id})
    mocked.patch(
        RECIPE_URL.format(id=recipe_id),
        status=HTTPStatus.UNPROCESSABLE_ENTITY,
        payload={"message": "instructions[0].annotations[0] invalid"},
    )
    return recipe_id


async def test_create_cleans_up_the_stub_on_failure(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """With cleanup_on_failure the rejected stub is deleted and noted as such."""
    recipe_id = _mock_create_with_failing_patch(mocked)
    mocked.delete(RECIPE_URL.format(id=recipe_id), status=HTTPStatus.NO_CONTENT)

    with pytest.raises(CookidooRequestException) as caught:
        await cookidoo.create_custom_recipe(
            CookidooCreateCustomRecipe(
                name="Rejected",
                ingredients=["1 onion"],
                instructions=["Chop"],
                serving_size=2,
                total_time=60,
                active_time=30,
            ),
            cleanup_on_failure=True,
        )

    assert caught.value.status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert caught.value.body is not None
    assert "annotations[0] invalid" in caught.value.body
    assert any("was removed" in note for note in caught.value.__notes__)
    assert ("delete", URL(RECIPE_URL.format(id=recipe_id))) in mocked.requests


async def test_create_keeps_the_stub_by_default(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """Without the option the behaviour is unchanged: the id is only noted."""
    recipe_id = _mock_create_with_failing_patch(mocked)

    with pytest.raises(CookidooRequestException) as caught:
        await cookidoo.create_custom_recipe(
            CookidooCreateCustomRecipe(
                name="Rejected",
                ingredients=["1 onion"],
                instructions=["Chop"],
                serving_size=2,
                total_time=60,
                active_time=30,
            )
        )

    assert any(
        f"Orphaned custom recipe stub id: {recipe_id}" in n
        for n in caught.value.__notes__
    )
    assert ("delete", URL(RECIPE_URL.format(id=recipe_id))) not in mocked.requests


async def test_create_cleanup_failure_is_reported_not_raised(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """A failing cleanup keeps the original error and notes the orphan."""
    recipe_id = _mock_create_with_failing_patch(mocked)
    mocked.delete(
        RECIPE_URL.format(id=recipe_id), status=HTTPStatus.INTERNAL_SERVER_ERROR
    )

    with pytest.raises(CookidooRequestException) as caught:
        await cookidoo.create_custom_recipe(
            CookidooCreateCustomRecipe(
                name="Rejected",
                ingredients=["1 onion"],
                instructions=["Chop"],
                serving_size=2,
                total_time=60,
                active_time=30,
            ),
            cleanup_on_failure=True,
        )

    assert caught.value.status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert any("Orphaned custom recipe stub id" in n for n in caught.value.__notes__)


def test_create_rejects_invalid_settings_before_any_request(
    mocked: aioresponses, cookidoo: Cookidoo
) -> None:
    """Local validation runs before the stub POST, so nothing is created."""
    recipe = CookidooCreateCustomRecipe(
        name="Bad",
        ingredients=[],
        instructions=[CookidooInstruction("Mix", CookidooStepSettings(speed=11))],
        serving_size=2,
        total_time=60,
        active_time=30,
    )
    with pytest.raises(ValueError, match="Invalid speed"):
        Cookidoo._build_custom_recipe_payload(
            name=recipe.name,
            ingredients=recipe.ingredients,
            steps=recipe.instructions,
            servings=recipe.serving_size,
            active_time=recipe.active_time,
            total_time=recipe.total_time,
            hints=[],
            machine_types=["TM7"],
            unit_text="portion",
            image=None,
            image_owned_by_user=False,
            work_status="PRIVATE",
            requires_annotations_check=False,
        )
    assert not mocked.requests


# -- edge cases for the parsers and validators -------------------------------


def test_explicit_offset_past_the_end_is_an_error() -> None:
    """An offset beyond the text cannot match and is reported as such."""
    steps = [
        CookidooInstruction(
            "mix", annotations=[CookidooTTSAnnotation("mix", time=10, offset=50)]
        )
    ]
    with pytest.raises(ValueError, match="not at offset 50"):
        Cookidoo._process_recipe_steps(steps, ingredients=[])

    # An offset equal to the text length is the end: nothing starts there.
    at_end = [
        CookidooInstruction(
            "mix", annotations=[CookidooTTSAnnotation("mix", time=10, offset=3)]
        )
    ]
    with pytest.raises(ValueError, match="not at offset 3"):
        Cookidoo._process_recipe_steps(at_end, ingredients=[])


def test_parser_ignores_boolean_speed_and_odd_instruction_items() -> None:
    """A boolean speed is dropped; non-text, non-object items are skipped."""
    annotation = _parse_custom_recipe_annotation(
        {
            "type": "TTS",
            "data": {"speed": True},
            "position": {"offset": 0, "length": 1},
        },
        "x",
    )
    assert isinstance(annotation, CookidooTTSAnnotation)
    assert annotation.speed is None

    assert _raw_instructions(["Chop", 42, {"type": "STEP", "text": "Cook"}]) == [
        {"type": "STEP", "text": "Chop"},
        {"type": "STEP", "text": "Cook"},
    ]
    assert _raw_instructions("not a list") is None


def test_validators_edge_values() -> None:
    """Remaining branches of the shared validators."""
    with pytest.raises(TypeError):
        normalize_speed(True)
    assert normalize_speed("3.0") == "3"
    with pytest.raises(TypeError):
        normalize_temperature(None)
    for speed in ("soft", "3", None):
        assert (
            validate_mode(
                "steaming",
                time=None,
                temperature=None,
                speed=speed,
                power=None,
                accessory=None,
            )
            == "steaming"
        )
