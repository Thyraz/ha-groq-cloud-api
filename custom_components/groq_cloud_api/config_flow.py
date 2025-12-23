"""Config flow for Groq Cloud API integration."""

from __future__ import annotations

import asyncio
from types import MappingProxyType
from typing import Any

import groq
import requests
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
)

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_NAME,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_OPTIONS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
    }
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect."""
    response = await asyncio.to_thread(
        requests.get,
        url="https://api.groq.com/openai/v1/models",
        headers={
            "Authorization": f"Bearer {data.get(CONF_API_KEY)}",
            "Content-Type": "application/json",
        },
        timeout=10,
    )

    LOGGER.debug(
        "Models request took %f s and returned %d - %s",
        response.elapsed.total_seconds(),
        response.status_code,
        response.reason,
    )

    if response.status_code == 401:
        raise InvalidAPIKey

    if response.status_code == 403:
        raise UnauthorizedError

    if response.status_code != 200:
        raise UnknownError


class GroqConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle UI config flow for Groq Cloud API."""

    VERSION = 1
    MINOR_VERSION = 1

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=STEP_USER_DATA_SCHEMA,
            )

        errors: dict[str, str] = {}

        self._async_abort_entries_match(user_input)
        try:
            await validate_input(self.hass, user_input)
        except groq.APIConnectionError:
            errors["base"] = "cannot_connect"
        except groq.AuthenticationError:
            errors["base"] = "invalid_auth"
        except InvalidAPIKey:
            errors["base"] = "invalid_auth"
        except UnauthorizedError:
            errors["base"] = "unauthorized"
        except Exception:
            LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(
                title=DEFAULT_NAME,
                data=user_input,
                options=RECOMMENDED_OPTIONS,
            )

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return GroqOptionsFlow()



class GroqOptionsFlow(OptionsFlow):
    """Groq Cloud API options flow handler."""

    last_rendered_recommended: bool = False

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = (
            self.config_entry.options
        )

        if user_input is None:
            self.last_rendered_recommended = options.get(CONF_RECOMMENDED, False)

        if user_input is not None:
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                if not user_input.get(CONF_LLM_HASS_API):
                    user_input.pop(CONF_LLM_HASS_API, None)
                return self.async_create_entry(title="", data=user_input)

            # Re-render the options again, now with the recommended options shown/hidden
            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]

            options = {
                CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                CONF_PROMPT: user_input.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                ),
                CONF_LLM_HASS_API: user_input.get(CONF_LLM_HASS_API),
            }

        schema = groq_config_option_schema(self.hass, options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


def groq_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
) -> dict:
    """Return a schema for Groq Cloud completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    ]

    schema: dict = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Required(
            CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, False)
        ): bool,
    }

    if options.get(CONF_RECOMMENDED):
        return schema

    schema.update(
        {
            vol.Optional(
                CONF_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
                default=RECOMMENDED_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=RECOMMENDED_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
        }
    )
    return schema


class UnknownError(HomeAssistantError):
    """Unknown error."""


class UnauthorizedError(HomeAssistantError):
    """API key valid but doesn't have the rights."""


class InvalidAPIKey(HomeAssistantError):
    """Invalid api_key error."""


class ModelNotFound(HomeAssistantError):
    """Model can't be found in the Groq Cloud model's list."""
