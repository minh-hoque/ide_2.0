import os
from typing import List, Dict, Any
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam
from helper.logging import get_logger
from prompts.auto_evaluation_prompts import (
    LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT,
    LLM_AS_A_JUDGE_SME_FEEDBACK_PROMPT,
)
import pandas as pd

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = get_logger(__name__)


# Data Class for Auto Evaluation Result from LLM
class AutoEvaluationResult(BaseModel):
    """
    Pydantic model for storing auto-evaluation results from the LLM.
    """

    rationale: str
    result: str


# GPT-4 query functions
@st.cache_data
def query_gpt4(
    prompt: str,
    system_prompt: str = "",
    model: str = "gpt-4o",
    json_response: bool = False,
) -> str:
    """
    Query the GPT-4 model with the given prompt and optional parameters.

    Args:
        prompt (str): The main prompt to send to the model.
        system_prompt (str, optional): A system-level prompt to set context. Defaults to "".
        model (str, optional): The specific GPT-4 model to use. Defaults to "gpt-4o".
        json_response (bool, optional): Whether to request a JSON-formatted response. Defaults to False.

    Returns:
        str: The model's response as a string.
    """
    try:
        # Prepare messages for the chat completion
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt} if system_prompt else None,
            {"role": "user", "content": prompt},
        ]
        messages = [msg for msg in messages if msg]

        # Create chat completion
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"} if json_response else None,
        )
        logger.info("GPT-4 Response: %s", response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Error querying GPT-4: %s", str(e))
        return f"Error: {str(e)}"


@st.cache_resource
def query_structured_gpt4(
    prompt: str, system_prompt: str = "", model: str = "gpt-4o-2024-08-06"
) -> AutoEvaluationResult:
    """
    Query the GPT-4 model and parse the response into a structured format.

    Args:
        prompt (str): The main prompt to send to the model.
        system_prompt (str, optional): A system-level prompt to set context. Defaults to "".
        model (str, optional): The specific GPT-4 model to use. Defaults to "gpt-4o-2024-08-06".

    Returns:
        AutoEvaluationResult: A structured object containing the parsed response from GPT-4.
                              This includes fields for 'rationale' (str) and 'result' (str).
    """
    try:
        # Prepare messages for the chat completion
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt} if system_prompt else None,
            {"role": "user", "content": prompt},
        ]
        messages = [msg for msg in messages if msg]

        # Create chat completion with structured parsing
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
        raise e


# Helper functions
def parse_auto_evaluation_response(result: str) -> tuple[str, str]:
    """
    Parse the auto-evaluation response from the LLM.

    Args:
        result (str): The raw response string from the LLM.

    Returns:
        tuple[str, str]: A tuple containing the parsed rationale and result.
    """
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
def auto_evaluate_responses(df: pd.DataFrame) -> pd.DataFrame:
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def evaluate_row(row: pd.Series) -> Dict[str, Any]:
        if row["rating"] == "ACCEPT":
            prompt = LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT.format(
                old_response=row["response"], new_response=row["new_response"]
            )
        elif row["edited_gt"]:
            prompt = LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT.format(
                old_response=row["edited_gt"], new_response=row["new_response"]
            )
        elif row["sme_feedback"]:
            prompt = LLM_AS_A_JUDGE_SME_FEEDBACK_PROMPT.format(
                old_response=row["response"],
                sme_feedback=row["sme_feedback"],
                new_response=row["new_response"],
            )
        else:
            logger.warning("No edited ground truth or SME feedback available.")
            return {"auto_evaluation": "UNKNOWN", "rationale": ""}

        response = query_structured_gpt4(prompt)
        logger.debug(f"LLM Response:\n{response}")
        logger.info(f"Auto Evaluation: {response.result}")

        # Add SME feedback to the beginning of the rationale if available
        rationale = (
            f"SME Feedback: {row['sme_feedback']}\n\n" if row["sme_feedback"] else ""
        )
        rationale += response.rationale

        return {"auto_evaluation": response.result, "rationale": rationale}

    results = []
    for index, row in df.iterrows():
        progress = (index + 1) / len(df)
        progress_bar.progress(progress)
        progress_text.text(f"Auto Evaluation Progress: {int(progress * 100)}%")

        logger.info(f"Auto Evaluation Question {index + 1}")
        logger.debug(f"Question: {row['question']}")

        results.append(evaluate_row(row))

    progress_bar.empty()
    progress_text.empty()

    df = df.assign(**pd.DataFrame(results))
    return df
