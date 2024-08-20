# Import necessary libraries
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import json
import os
from helper.logging import get_logger
from openai import OpenAI

from pydantic import BaseModel

# Import custom modules
from css.style import apply_snorkel_style
from prompts.auto_evaluation_prompts import (
    LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT,
    LLM_AS_A_JUDGE_SME_FEEDBACK_PROMPT,
)
from prompts.base_prompts import PROMPT_1, PROMPT_2, PROMPT_3
from helper.llms import query_gpt4, query_structured_gpt4, auto_evaluate_responses

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


# Create a state variable for button click
if "sme_eval_button_clicked" not in st.session_state:
    st.session_state.sme_eval_button_clicked = False

# Check if the button was clicked and perform the action
if st.session_state.sme_eval_button_clicked:
    print(st.session_state.auto_evaled_df)
    print("HI2")
    # Save the auto_evaled_df to a CSV file
    try:
        st.session_state.auto_evaled_df.to_csv(
            "./storage/iteration_responses/new_responses2.csv", index=False
        )
        st.success("Responses saved for SME evaluation!")
        # Reset the button state
        st.session_state.sme_eval_button_clicked = False
    except Exception as e:
        st.error(f"Error saving responses: {str(e)}")
        logger.error(f"Error saving responses: {str(e)}")

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

# Select a model
model = st.selectbox(
    "Select a model", ["gpt-4o", "gpt-4o-2024-08-06", "gpt-4-turbo-preview"]
)

try:
    with open("./storage/baseline_prompt.txt", "r") as f:
        baseline_prompt = f.read()
except FileNotFoundError:
    logger.warning("No baseline prompt found.")
    baseline_prompt = PROMPT_1

col1, col2 = st.columns([2, 1])

# Prompt Development View
with col1:
    modified_prompt = st.text_area("Modified Prompt", value=baseline_prompt, height=600)

# SME Feedback View
with col2:
    st.markdown("<b>SME Feedback</b>", unsafe_allow_html=True)
    sme_feedback_container = st.container(height=600)
    with sme_feedback_container:
        for feedback in df["sme_feedback"]:
            if feedback != "nan":
                st.markdown(f"- {feedback}")


if st.button("Preview Prompt"):
    new_responses = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_rows = len(df)

    for index, row in df.iterrows():
        logger.info(f"Inference index {index}")
        question = row["question"]
        formated_prompt = modified_prompt.format(user_question=question)
        response = query_gpt4(formated_prompt, model=model)
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
            "rationale",
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
        "rationale": st.column_config.TextColumn("Rationale", width="large"),
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
            "rationale",
        ],
    )
    st.session_state.auto_evaled_df = auto_evaled_df
    print("HI")

    # # Add a button to send for SME evaluation
    # if st.button("Send for SME Evaluation"):
    #     st.session_state.auto_evaled_df = auto_evaled_df
    #     st.session_state.sme_eval_button_clicked = True
    #     st.rerun()

# Check if the button was clicked and perform the action
if st.button("Send for SME Evaluation"):
    if st.session_state.auto_evaled_df is not None:
        # Rename columns
        df_to_save = st.session_state.auto_evaled_df.rename(
            columns={
                "response": "old_response",
                "new_response": "response",
                "rating": "old_rating",
            }
        )
        # Save the modified dataframe to a CSV file
        try:
            df_to_save.to_csv(
                "./storage/iteration_responses/new_responses.csv", index=False
            )
            st.success("Responses saved for SME evaluation!")
            # Reset the button state
            st.session_state.sme_eval_button_clicked = False
        except Exception as e:
            st.error(f"Error saving responses: {str(e)}")
            logger.error(f"Error saving responses: {str(e)}")
    else:
        st.error(
            "No responses to send for SME evaluation. Need to preview prompt first."
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
