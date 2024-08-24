import os
from typing import Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from css.style import apply_snorkel_style
from helper.llms import query_gpt4, auto_evaluate_responses
from helper.logging import get_logger
from prompts.base_prompts import PROMPT_1, PROMPT_4
import pandas as pd

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get a logger for this module
logger = get_logger(__name__)


def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Prompt Iteration", page_icon=":pencil2:", layout="wide"
    )
    st.markdown(apply_snorkel_style(), unsafe_allow_html=True)
    st.markdown(
        '<h1 class="header">Prompt Development Workflow</h1>', unsafe_allow_html=True
    )


def setup_logging():
    """Set up logging level selector."""
    logging_level = st.selectbox(
        "Select Logging Level",
        ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        index=1,
    )
    logger.setLevel(logging_level)


def load_data() -> pd.DataFrame:
    """Load and prepare the dataframe."""
    eval_files = [
        f
        for f in os.listdir("./storage/manual_annotations")
        if f.startswith("evaluated_responses") and f.endswith(".csv")
    ]

    if not eval_files:
        st.error(
            "No evaluated responses found. Please complete manual annotations first."
        )
        st.stop()

    selected_file = st.selectbox("Select evaluated responses file:", ["-"] + eval_files)

    if selected_file == "-":
        st.error("No file selected. Please try again.")
        st.stop()

    if "df" not in st.session_state or selected_file != st.session_state.get(
        "selected_file"
    ):
        try:
            df = pd.read_csv(
                os.path.join("./storage/manual_annotations", selected_file),
                na_values=[
                    "",
                    "nan",
                    "NaN",
                    "None",
                ],  # Specify values to be treated as NaN
                keep_default_na=True,
            )
            df = preprocess_dataframe(df)
            st.session_state.df = df
            st.session_state.selected_file = selected_file
        except FileNotFoundError:
            logger.error(f"Selected file {selected_file} not found.")
            st.error(f"Selected file {selected_file} not found. Please try again.")
            st.stop()

    return st.session_state.df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the loaded dataframe."""
    columns_to_drop = [
        "label",
        "feedback",
        "rationale",
        "auto_evaluation",
        "old_response",
        "old_rating",
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Convert 'edited_gt' and 'sme_feedback' to string, replacing NaN with empty string
    df["edited_gt"] = df["edited_gt"].fillna("").astype(str)
    df["sme_feedback"] = df["sme_feedback"].fillna("").astype(str)

    return df


def display_evaluated_responses(df: pd.DataFrame):
    """Display the evaluated responses in a data editor."""
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


def load_baseline_prompt() -> str:
    """Load the baseline prompt from file or use default."""
    try:
        with open("./storage/baseline_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("No baseline prompt found. Using default prompt.")
        return PROMPT_4


def display_prompt_dev_box(baseline_prompt: str, df: pd.DataFrame) -> Tuple[str, str]:
    """Display the prompt development box for development."""
    st.subheader("Prompt Dev Box")
    st.write(
        "Modify the baseline prompt based on the feedback and evaluated responses."
    )

    model = st.selectbox(
        "Select a model", ["gpt-4o", "gpt-4o-2024-08-06", "gpt-4-turbo-preview"]
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        modified_prompt = st.text_area(
            "Modified Prompt", value=baseline_prompt, height=600
        )
    with col2:
        st.markdown("<b>SME Feedback</b>", unsafe_allow_html=True)
        sme_feedback_container = st.container(height=600)
        with sme_feedback_container:
            for feedback in df["sme_feedback"]:
                if feedback != "":
                    st.markdown(f"- {feedback}")

    return modified_prompt, model


def calculate_metrics(auto_evaled_df: pd.DataFrame) -> dict:
    """Calculate metrics for the auto-evaluated responses."""
    total_responses = len(auto_evaled_df)
    accepted_responses = (auto_evaled_df["auto_evaluation"] == "ACCEPT").sum()
    accept_percentage = (accepted_responses / total_responses) * 100

    total_previous_accept = (auto_evaled_df["rating"] == "ACCEPT").sum()
    regressions = (
        (auto_evaled_df["rating"] == "ACCEPT")
        & (auto_evaled_df["auto_evaluation"] == "REJECT")
    ).sum()
    regression_percentage = (
        (regressions / total_previous_accept * 100) if total_previous_accept > 0 else 0
    )

    total_previous_reject = (auto_evaled_df["rating"] == "REJECT").sum()
    improvements = (
        (auto_evaled_df["rating"] == "REJECT")
        & (auto_evaled_df["auto_evaluation"] == "ACCEPT")
        & (auto_evaled_df["sme_feedback"] != "")
    ).sum()
    improvement_percentage = (
        (improvements / total_previous_reject * 100) if total_previous_reject > 0 else 0
    )

    return {
        "accept_percentage": accept_percentage,
        "regression_percentage": regression_percentage,
        "improvement_percentage": improvement_percentage,
        "accepted_responses": accepted_responses,
        "total_responses": total_responses,
        "regressions": regressions,
        "improvements": improvements,
    }


def display_metrics(current_metrics: dict):
    """Display the calculated metrics with tooltips for descriptions and deltas."""
    st.subheader("Evaluation Metrics")
    col1, col2, col3 = st.columns(3)

    previous_metrics = st.session_state.get("previous_metrics", {})

    def get_delta(current, previous, key):
        if previous and key in previous:
            delta = current[key] - previous[key]
            delta_str = f"{delta:+.2f}%"
            delta_color = "green" if delta > 0 else "red"
            return delta_str, delta_color
        return None, None

    with col1:
        delta, color = get_delta(current_metrics, previous_metrics, "accept_percentage")
        st.metric(
            label="Accepted Responses",
            value=f"{current_metrics['accept_percentage']:.2f}%",
            delta=delta,
            # delta_color=color,
            help=f"Percentage of total responses that were accepted in this iteration. ({current_metrics['accepted_responses']} / {current_metrics['total_responses']})",
        )

    with col2:
        delta, color = get_delta(
            current_metrics, previous_metrics, "regression_percentage"
        )
        if delta:
            color = (
                "red" if color == "green" else "green"
            )  # Invert color for regressions
        st.metric(
            label="Regressions",
            value=f"{current_metrics['regression_percentage']:.2f}%",
            delta=delta,
            # delta_color=color,
            help=f"Percentage of responses that were previously ACCEPT but now REJECT, out of all previously accepted responses. ({current_metrics['regressions']} regressions)",
        )

    with col3:
        delta, color = get_delta(
            current_metrics, previous_metrics, "improvement_percentage"
        )
        st.metric(
            label="Improvements",
            value=f"{current_metrics['improvement_percentage']:.2f}%",
            delta=delta,
            # delta_color=color,
            help=f"Percentage of responses that were previously REJECT with SME feedback and are now ACCEPT, out of all previously rejected responses. ({current_metrics['improvements']} improvements)",
        )

    # Store current metrics for next comparison
    st.session_state.previous_metrics = current_metrics


def preview_prompt(df: pd.DataFrame, modified_prompt: str, model: str):
    """Preview the modified prompt and generate new responses."""
    if st.button("Preview Prompt"):
        new_responses = generate_new_responses(df, modified_prompt, model)
        auto_eval_df = df.copy()
        auto_eval_df["new_response"] = new_responses
        auto_evaled_df = auto_evaluate_responses(auto_eval_df)

        metrics = calculate_metrics(auto_evaled_df)
        display_metrics(metrics)

        display_preview_results(auto_evaled_df)
        st.session_state.auto_evaled_df = auto_evaled_df


def generate_new_responses(df: pd.DataFrame, modified_prompt: str, model: str) -> list:
    """Generate new responses using the modified prompt."""
    new_responses = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_rows = len(df)

    for index, row in df.iterrows():
        logger.info(f"Inference index {index}")
        question = row["question"]
        formatted_prompt = modified_prompt.format(user_question=question)
        response = query_gpt4(formatted_prompt, model=model)
        new_responses.append(response)

        progress = float(index + 1) / float(total_rows)
        progress_bar.progress(progress)
        progress_text.text(f"Generating new responses: {index + 1}/{total_rows}")

    progress_bar.empty()
    progress_text.empty()
    return new_responses


def display_preview_results(auto_evaled_df: pd.DataFrame):
    """Display the results of the preview."""
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


def send_for_sme_evaluation():
    """Send the generated responses for SME evaluation."""
    if st.button("Send for SME Evaluation"):
        if "auto_evaled_df" in st.session_state:
            df_to_save = st.session_state.auto_evaled_df.rename(
                columns={
                    "response": "old_response",
                    "new_response": "response",
                    "rating": "old_rating",
                }
            )
            try:
                df_to_save.to_csv(
                    "./storage/iteration_responses/new_experiment_responses.csv",
                    index=False,
                )
                st.success("Responses saved for SME evaluation!")
            except Exception as e:
                st.error(f"Error saving responses: {str(e)}")
                logger.error(f"Error saving responses: {str(e)}")
        else:
            st.error(
                "No responses to send for SME evaluation. Need to preview prompt first."
            )


def main():
    """Main function to run the Streamlit app."""
    setup_page()
    setup_logging()
    df = load_data()
    display_evaluated_responses(df)
    baseline_prompt = load_baseline_prompt()
    modified_prompt, model = display_prompt_dev_box(baseline_prompt, df)
    preview_prompt(df, modified_prompt, model)
    send_for_sme_evaluation()


if __name__ == "__main__":
    print("Starting prompt iteration workflow...")
    main()
