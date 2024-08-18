# Import necessary libraries
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import json
import os
from helper.logging import get_logger
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

# Import custom modules
from css.style import apply_snorkel_style
from prompts.auto_evaluation_prompts import (
    LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT,
    LLM_AS_A_JUDGE_SME_FEEDBACK_PROMPT,
)
from helper.llms import query_gpt4, query_structured_gpt4

# Load environment variables
load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get a logger for this module
logger = get_logger(__name__)

# Page configuration
st.set_page_config(page_title="Prompt Iteration", page_icon=":pencil2:", layout="wide")
st.markdown(apply_snorkel_style(), unsafe_allow_html=True)
st.markdown(
    '<h1 class="header">Prompt Development Workflow</h1>', unsafe_allow_html=True
)

# Add logging level selector
logging_level = st.selectbox(
    "Select Logging Level",
    ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    index=1,  # Default to INFO
)
logger.setLevel(logging_level)


# # Define AutoEvaluationResult model
# class AutoEvaluationResult(BaseModel):
#     rationale: str
#     result: str


# # Function to query GPT-4 with structured output
# @st.cache_data
# def query_structured_gpt4(prompt, system_prompt="", model="gpt-4o-2024-08-06"):
#     try:
#         messages = [{"role": "user", "content": prompt}]
#         if system_prompt:
#             messages.insert(0, {"role": "system", "content": system_prompt})

#         response = client.beta.chat.completions.parse(
#             model=model,
#             messages=messages,
#             temperature=0,
#             response_format=AutoEvaluationResult,
#         )
#         logger.info("Structured GPT-4 Response: %s", response.choices[0].message.parsed)
#         return response.choices[0].message.parsed
#     except Exception as e:
#         logger.error(f"Error querying GPT-4: {str(e)}")
#         return f"Error: {str(e)}"


# # Function to query GPT-4 with optional JSON response
# @st.cache_data
# def query_gpt4(prompt, system_prompt="", model="gpt-4o", json_response=False):
#     try:
#         messages = [ChatCompletionUserMessageParam(role="user", content=prompt)]
#         if system_prompt:
#             messages.insert(
#                 0,
#                 ChatCompletionSystemMessageParam(role="system", content=system_prompt),
#             )

#         response_format = {"type": "json_object"} if json_response else None

#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             temperature=0,
#             response_format=response_format,
#         )
#         logger.info("GPT-4 Response: %s", response.choices[0].message.content)
#         return response.choices[0].message.content
#     except Exception as e:
#         logger.error(f"Error querying GPT-4: {str(e)}")
#         return f"Error: {str(e)}"


# # Function to parse auto-evaluation response
# def parse_auto_evaluation_response(result):
#     rational_parts = result.lower().split("rationale:")
#     result_parts = result.lower().split("result:")

#     parsed_rational = ""
#     parsed_result = "UNKNOWN"

#     if len(rational_parts) > 1:
#         parsed_rational = rational_parts[1].strip()

#     if len(result_parts) > 1:
#         parsed_result = result_parts[1].strip().upper()
#         if parsed_result not in ["ACCEPT", "REJECT"]:
#             parsed_result = "UNKNOWN"

#     return parsed_rational, parsed_result


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
        elif row["edited_gt"] != "nan":
            formated_prompt = LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT.format(
                old_response=row["edited_gt"], new_response=row["new_response"]
            )
        elif row["sme_feedback"] != "nan":
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

        rational = auto_evaluate_response.rationale
        auto_evaluation = auto_evaluate_response.result

        rational_list.append(rational)
        auto_evaluation_results.append(auto_evaluation)
        logger.debug(f"Old Response:\n{row['response']}")
        logger.debug(f"New Response:\n{row['new_response']}")
        logger.info(f"Auto Evaluation: {auto_evaluation}")

        logger.info("-----------------------------------")

    progress_bar.empty()
    progress_text.empty()

    df["auto_evaluation"] = auto_evaluation_results
    df["rational"] = rational_list

    return df


# Load and prepare the dataframe
if "df" not in st.session_state:
    try:
        st.session_state.df = pd.read_csv(
            "./storage/manual_annotations/evaluated_responses.csv"
        )
    except FileNotFoundError:
        logger.error("No evaluated responses found.")
        st.error(
            "No evaluated responses found. Please complete manual annotations first."
        )
        st.stop()

    st.session_state.df.drop(columns=["label", "feedback"], inplace=True)
    st.session_state.df = st.session_state.df.sample(n=4, random_state=0).reset_index(
        drop=True
    )

df = st.session_state.df
df["edited_gt"] = df["edited_gt"].astype(str)
df["sme_feedback"] = df["sme_feedback"].astype(str)

# Display the evaluated responses
st.subheader("Evaluated Responses")

column_config = {
    "question": st.column_config.TextColumn("Question", width="medium"),
    "response": st.column_config.TextColumn("Response", width="medium"),
    "rating": st.column_config.TextColumn("Rating", width="small"),
    "edited_gt": st.column_config.TextColumn("Edited Ground Truth", width="large"),
    "sme_feedback": st.column_config.TextColumn("SME Feedback", width="large"),
}

st.data_editor(
    df,
    column_config=column_config,
    hide_index=True,
    num_rows="fixed",
    use_container_width=True,
)

# Prompt playground
st.subheader("Prompt Dev Box")
st.write("Modify the baseline prompt based on the feedback and evaluated responses.")

try:
    with open("./storage/baseline_prompt.txt", "r") as f:
        baseline_prompt = f.read()
except FileNotFoundError:
    logger.warning("No baseline prompt found.")
    baseline_prompt = "No baseline prompt found. Please create one."

modified_prompt = st.text_area("Modified Prompt", value=baseline_prompt, height=600)

if st.button("Preview Prompt"):
    new_responses = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_rows = len(df)

    for index, row in df.iterrows():
        logger.info(f"Inference index {index}")
        question = row["question"]
        formated_prompt = modified_prompt.format(user_question=question)
        response = query_gpt4(formated_prompt)
        new_responses.append(response)

        logger.debug(f"Processed {index + 1} out of {total_rows}")
        progress = float(index + 1) / float(total_rows)
        progress_bar.progress(progress)
        progress_text.text(f"Generating new responses: {index + 1}/{total_rows}")

    progress_bar.empty()
    progress_text.empty()

    auto_eval_df = df.copy()
    auto_eval_df["new_response"] = new_responses
    auto_evaled_df = auto_evaluate_responses(auto_eval_df)

    st.write("Responses generated with the modified prompt:")

    display_df = auto_evaled_df[
        [
            "question",
            "response",
            "new_response",
            "rating",
            "auto_evaluation",
            "rational",
        ]
    ]

    column_config = {
        "question": st.column_config.TextColumn("Question"),
        "response": st.column_config.TextColumn("Original Response", width="large"),
        "new_response": st.column_config.TextColumn("New Response", width="large"),
        "rating": st.column_config.TextColumn("Previous Rating", width="small"),
        "auto_evaluation": st.column_config.TextColumn(
            "Auto Evaluation",
            help="Green checkmark for ACCEPT, red X for REJECT",
            width="small",
        ),
        "rational": st.column_config.TextColumn("Rational", width="large"),
    }

    display_df["auto_evaluation"] = display_df["auto_evaluation"].apply(
        lambda x: "✅" if x == "ACCEPT" else "❌"
    )

    st.data_editor(
        display_df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        disabled=[
            "question",
            "response",
            "new_response",
            "rating",
            "auto_evaluation",
            "rational",
        ],
    )

# # Save the updated dataframe
# df.to_csv("./storage/manual_annotations/new_responses.csv", index=False)
# st.success("New responses generated and saved successfully!")

# # Optional: Add a section to test the modified prompt
# st.subheader("Test Modified Prompt")
# test_question = st.text_input("Enter a test question:")
# if st.button("Generate Response"):
#     # Here you would typically call your LLM API with the modified prompt and test question
#     # For demonstration, we'll just display a placeholder response
#     st.write("Generated Response: [Your LLM-generated response would appear here]")
