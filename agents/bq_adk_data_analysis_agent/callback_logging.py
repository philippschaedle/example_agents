"""Module docstring for callback_logging.py."""

import logging

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse


def log_query_to_model(callback_context: CallbackContext, llm_request: LlmRequest):
    """Log the query to the model.

    Args:
        callback_context: Context of the agent.
        llm_request: The LLM Request.
    """
    """Log a query to the model.

    Args:
        callback_context: The callback context.
        llm_request: The LLM request.
    """
    if llm_request.contents and llm_request.contents[-1].role == "user":
        for part in llm_request.contents[-1].parts:
            if part.text:
                logging.info(
                    "[query to %s]: %s", callback_context.agent_name, part.text
                )


def log_model_response(callback_context: CallbackContext, llm_response: LlmResponse):
    """Log the model response.

    Args:
        callback_context: Context of the agent.
        llm_response: The LLM Response.
    """
    """Log a model response.

    Args:
        callback_context: The callback context.
        llm_response: The LLM response.
    """
    if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            if part.text:
                logging.info(
                    "[response from %s]: %s", callback_context.agent_name, part.text
                )
            elif part.function_call:
                logging.info(
                    "[function call from %s]: %s",
                    callback_context.agent_name,
                    part.function_call.name,
                )
