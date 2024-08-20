import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from helper.logging import get_logger

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get a logger for this module
logger = get_logger(__name__)


# Data models
class AutoEvaluationResult(BaseModel):
    rationale: str
    result: str


# GPT-4 query functions
@st.cache_data
def query_gpt4(prompt, system_prompt="", model="gpt-4o", json_response=False):
    try:
        messages = [ChatCompletionUserMessageParam(role="user", content=prompt)]
        if system_prompt:
            messages.insert(
                0,
                ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            )

        response_format = {"type": "json_object"} if json_response else None
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format=response_format,
        )
        logger.info("GPT-4 Response: %s", response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Error querying GPT-4: \n%s", str(e))
        return f"Error: {str(e)}"


@st.cache_data
def query_structured_gpt4(prompt, system_prompt="", model="gpt-4o-2024-08-06"):
    try:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=0,
            response_format=AutoEvaluationResult,
        )
        logger.info(
            "Structured GPT-4 Response: \n%s", response.choices[0].message.parsed
        )
        return response.choices[0].message.parsed
    except Exception as e:
        logger.error("Error querying structured GPT-4: %s", str(e))
        return f"Error: {str(e)}"


# Helper functions
def parse_auto_evaluation_response(result):
    rational_parts = result.lower().split("rationale:")
    result_parts = result.lower().split("result:")

    parsed_rational = rational_parts[1].strip() if len(rational_parts) > 1 else ""
    parsed_result = "UNKNOWN"

    if len(result_parts) > 1:
        parsed_result = result_parts[1].strip().upper()
        if parsed_result not in ["ACCEPT", "REJECT"]:
            parsed_result = "UNKNOWN"

    return parsed_rational, parsed_result
