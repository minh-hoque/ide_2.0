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
    LLM_AS_A_JUDGE_SME_FEEDBACK_PROMPT_V2,
    LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT_V2,
    LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT_V3,
    LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT_V4,
)
import pandas as pd
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
import concurrent.futures
from functools import partial
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get a logger for this module
logger = get_logger(__name__)

# Set the logging level based on the session state
if "logging_level" in st.session_state:
    logger.setLevel(st.session_state["logging_level"])

# Set auto_eval prompt
AUTO_EVAL_EQUIVALENCE_PROMPT = LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT_V4
AUTO_EVAL_SME_FEEDBACK_PROMPT = LLM_AS_A_JUDGE_SME_FEEDBACK_PROMPT_V2


# Data Class for Auto Evaluation Result from LLM
class AutoEvaluationResult(BaseModel):
    rationale: str
    result: str


# Data Class for Extraction Result from LLM
class ExtractionResult(BaseModel):
    extraction: Dict[str, List[str]]


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
def query_structured_gpt4(
    prompt, system_prompt="", model="gpt-4o-2024-08-06", response_format=None
):
    try:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=0,
            response_format=response_format,
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
def auto_evaluate_batch(batch):
    results = []
    for row in batch:
        logger.debug(f"Processing row with index {row['index']}")
        if row["rating"] == "ACCEPT":
            formated_prompt = AUTO_EVAL_EQUIVALENCE_PROMPT.format(
                old_response=row["response"],
                new_response=row["new_response"],
                question=row["question"],
            )
            logger.debug("Using ACCEPT prompt")
        elif row["edited_gt"] != "":
            formated_prompt = AUTO_EVAL_EQUIVALENCE_PROMPT.format(
                old_response=row["edited_gt"],
                new_response=row["new_response"],
                question=row["question"],
            )
            logger.debug("Using edited_gt prompt")
        elif row["sme_feedback"] != "":
            formated_prompt = AUTO_EVAL_SME_FEEDBACK_PROMPT.format(
                old_response=row["response"],
                sme_feedback=row["sme_feedback"],
                new_response=row["new_response"],
                question=row["question"],
            )
            logger.debug("Using SME feedback prompt")
        else:
            logger.warning(
                f"No valid prompt found for row {row['index']}, marking as UNKNOWN"
            )
            results.append((row["index"], "", "UNKNOWN"))
            continue

        logger.debug(f"Querying GPT-4 for row {row['index']}")
        auto_evaluate_response = query_structured_gpt4(
            formated_prompt, response_format=AutoEvaluationResult
        )
        rationale = (
            f"SME Feedback: {row['sme_feedback']}\n\n" if row["sme_feedback"] else ""
        )
        rationale += auto_evaluate_response.rationale
        results.append((row["index"], rationale, auto_evaluate_response.result))
        logger.debug(
            f"Processed row {row['index']} with result: {auto_evaluate_response.result}"
        )
    return results


def auto_evaluate_responses(df):
    logger.info("Starting auto-evaluation process")
    batch_size = 2  # Adjust this based on your needs and rate limits
    num_workers = 5  # Adjust based on your CPU cores and rate limits

    df = df.reset_index()  # Ensure we have an index column
    batches = [df[i : i + batch_size] for i in range(0, len(df), batch_size)]
    logger.info(f"Created {len(batches)} batches of size {batch_size}")

    progress_bar = st.progress(0)
    progress_text = st.empty()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        logger.info(f"Starting ThreadPoolExecutor with {num_workers} workers")

        # Add ScriptRunContext to all threads in the pool
        for t in executor._threads:
            add_script_run_ctx(t)

        # Use executor.map to process batches
        batch_results = executor.map(
            auto_evaluate_batch, [batch.to_dict("records") for batch in batches]
        )

        for i, batch_result in enumerate(batch_results):
            results.extend(batch_result)

            logger.info(f"Completed batch {i+1}/{len(batches)}")

            # Update progress bar and text
            progress = (i + 1) / len(batches)
            progress_bar.progress(progress)
            progress_text.text(f"Auto Evaluation Progress: {int(progress * 100)}%")

    progress_bar.empty()
    progress_text.empty()

    logger.info("All batches processed, sorting results")
    results.sort(key=lambda x: x[0])

    df["rationale"] = [result[1] for result in results]
    df["auto_evaluation"] = [result[2] for result in results]

    df = df.set_index("index")  # Reset the index to its original state
    logger.info("Auto-evaluation process completed")
    return df


# Update the progress outside of the threaded execution
def update_progress(current, total):
    progress = current / total
    st.progress(progress)
    st.text(f"Auto Evaluation Progress: {int(progress * 100)}%")
