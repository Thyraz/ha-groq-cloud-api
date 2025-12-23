"""Constants for the Groq Cloud API integration."""

import logging

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

DOMAIN = "groq_cloud_api"

LOGGER = logging.getLogger(__name__)

DEFAULT_NAME = "Groq Cloud API"
DEFAULT_CONVERSATION_NAME = "Groq Conversation"
CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "llama-3.3-70b-versatile"
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 3000
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}