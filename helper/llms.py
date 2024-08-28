import os
from typing import List, Dict, Any
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
import openai
from helper.logging import get_logger
from prompts.auto_evaluation_prompts import (
    LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT,
    LLM_AS_A_JUDGE_SME_FEEDBACK_PROMPT,
)
import pandas as pd
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get a logger for this module
logger = get_logger(__name__)


# Data Class for Auto Evaluation Result from LLM
class AutoEvaluationResult(BaseModel):
    rationale: str
    result: str


# GPT-4 query functions
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
    ),
)
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


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
    ),
)
@st.cache_resource
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


# Function to auto-evaluate responses
def auto_evaluate_responses(df):
    rational_list = []
    auto_evaluation_results = []
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for index, row in df.iterrows():
        progress = (index + 1) / len(df)
        progress_bar.progress(progress)
        progress_text.text(f"Auto Evaluation Progress: {int(progress * 100)}%")

        logger.info(f"Auto Evaluation Question {index + 1}")
        logger.debug(f"Question: {row['question']}")

        if row["rating"] == "ACCEPT":
            formated_prompt = LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT.format(
                old_response=row["response"], new_response=row["new_response"]
            )
        elif row["edited_gt"] != "":
            formated_prompt = LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT.format(
                old_response=row["edited_gt"], new_response=row["new_response"]
            )
        elif row["sme_feedback"] != "":
            formated_prompt = LLM_AS_A_JUDGE_SME_FEEDBACK_PROMPT.format(
                old_response=row["response"],
                sme_feedback=row["sme_feedback"],
                new_response=row["new_response"],
            )
        else:
            auto_evaluation_results.append("UNKNOWN")
            rational_list.append("")
            logger.warning("No edited ground truth or SME feedback available.")
            logger.info("Auto Evaluation: N/A")
            continue

        auto_evaluate_response = query_structured_gpt4(formated_prompt)
        logger.debug(f"LLM Response:\n{auto_evaluate_response}")

        rationale = (
            f"SME Feedback: {row['sme_feedback']}\n\n" if row["sme_feedback"] else ""
        )
        rationale += auto_evaluate_response.rationale

        auto_evaluation = auto_evaluate_response.result

        rational_list.append(rationale)
        auto_evaluation_results.append(auto_evaluation)
        logger.debug(f"Old Response:\n{row['response']}")
        logger.debug(f"New Response:\n{row['new_response']}")
        logger.info(f"Auto Evaluation: {auto_evaluation}")

        logger.info("-----------------------------------")

    progress_bar.empty()
    progress_text.empty()

    df["auto_evaluation"] = auto_evaluation_results
    df["rationale"] = rational_list

    return df
