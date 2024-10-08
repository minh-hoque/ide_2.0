import os
import json
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit.delta_generator import DeltaGenerator

from css.style import apply_snorkel_style
from helper.llms import query_llm, auto_evaluate_responses
from helper.logging import get_logger, setup_logging
from prompts.base_prompts import BASELINE_PROMPT

# Constants for directories and folders
STORAGE_DIR = "./storage"
MANUAL_ANNOTATIONS_DIR = os.path.join(STORAGE_DIR, "qna_data")
PROMPTS_DIR = os.path.join(STORAGE_DIR, "qna_prompts")
ITERATION_RESPONSES_DIR = os.path.join(STORAGE_DIR, "qna_results")
MAPPING_FILE = os.path.join(STORAGE_DIR, "prompt_response_mapping.json")

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get a logger for this module
logger = get_logger(__name__)


def setup_page() -> None:
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Prompt Iteration", page_icon=":pencil2:", layout="wide"
    )
    st.markdown(apply_snorkel_style(), unsafe_allow_html=True)
    st.markdown(
        '<h1 class="header">Prompt Development Workflow</h1>', unsafe_allow_html=True
    )


def load_data() -> pd.DataFrame:
    """
    Load and prepare the dataframe from evaluated responses.

    Returns:
        pd.DataFrame: Preprocessed dataframe containing evaluated responses.
    """
    st.subheader("Load Data")

    data_source = st.radio(
        "Choose data source:", ["Select from existing files", "Upload a CSV file"]
    )

    if data_source == "Select from existing files":
        eval_files = [
            f for f in os.listdir(MANUAL_ANNOTATIONS_DIR) if f.endswith("evaluated.csv")
        ]

        if not eval_files:
            st.error(
                "No evaluated responses found. Please complete manual annotations first."
            )
            st.stop()

        selected_file = st.selectbox(
            "Select evaluated responses file:", ["-"] + eval_files
        )

        if selected_file == "-":
            st.error("No file selected. Please try again.")
            st.stop()

        try:
            df = pd.read_csv(
                os.path.join(MANUAL_ANNOTATIONS_DIR, selected_file),
                na_values=["", "nan", "NaN", "None"],
                keep_default_na=True,
            )
        except FileNotFoundError:
            logger.error(f"Selected file {selected_file} not found.")
            st.error(f"Selected file {selected_file} not found. Please try again.")
            st.stop()

    else:  # Upload a CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(
                    uploaded_file,
                    na_values=["", "nan", "NaN", "None"],
                    keep_default_na=True,
                )
            except Exception as e:
                logger.error(f"Error reading uploaded file: {str(e)}")
                st.error(f"Error reading uploaded file: {str(e)}")
                st.stop()
        else:
            st.info("Please upload a CSV file.")
            st.stop()

    return preprocess_dataframe(df)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the loaded dataframe.

    Args:
        df (pd.DataFrame): Raw dataframe loaded from CSV.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    columns_to_drop = [
        "label",
        "feedback",
        "rationale",
        "auto_evaluation",
        "old_response",
        "old_rating",
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    df["edited_gt"] = df["edited_gt"].fillna("").astype(str)
    df["sme_feedback"] = df["sme_feedback"].fillna("").astype(str)
    return df


def display_evaluated_responses(df: pd.DataFrame) -> None:
    """
    Display the evaluated responses in a dataframe and handle row selection.

    Args:
        df (pd.DataFrame): Dataframe containing evaluated responses.
    """
    st.subheader("Evaluated Responses")
    column_config = {
        "question": st.column_config.TextColumn("Question", width="medium"),
        "response": st.column_config.TextColumn("Response", width="medium"),
        "rating": st.column_config.TextColumn("Rating", width="small"),
        "edited_gt": st.column_config.TextColumn("Edited Ground Truth", width="large"),
        "sme_feedback": st.column_config.TextColumn("SME Feedback", width="large"),
    }

    if st.session_state.get("selected_row_index", -1) >= 0:
        displayed_columns = [
            "question",
            "response",
            "rating",
            "edited_gt",
            "sme_feedback",
        ]
        filtered_df = st.session_state.filtered_df[displayed_columns]
        st.dataframe(
            filtered_df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
        )
    else:
        displayed_columns = [
            "question",
            "response",
            "rating",
            "edited_gt",
            "sme_feedback",
        ]
        filtered_df = df[displayed_columns]
        selection = st.dataframe(
            filtered_df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        selected_rows = selection.selection.rows
        if selected_rows:
            st.session_state.selected_row_index = selected_rows[0]
            st.session_state.filtered_df = df.iloc[
                [st.session_state.selected_row_index]
            ]
            st.rerun()


def load_saved_prompts():
    """Load the list of saved prompts from the storage directory."""
    if not os.path.exists(PROMPTS_DIR):
        return []

    prompt_files = [f for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt")]
    prompt_files.sort(reverse=True)  # Sort files in reverse order (newest first)
    return prompt_files


def load_prompt() -> str:
    """
    Load the baseline prompt from file, saved prompt, or use default if not found.

    Returns:
        str: Baseline prompt text.
    """
    st.sidebar.markdown("## Load Saved Prompt")
    saved_prompts = load_saved_prompts()
    prompt_list = ["Current Baseline"] + saved_prompts

    # Initialize last_selected_prompt if it doesn't exist
    if "last_selected_prompt" not in st.session_state:
        st.session_state.last_selected_prompt = prompt_list[0]

    # Define a callback function for the selectbox
    def on_prompt_change():
        selected_prompt = st.session_state.prompt_selector
        if selected_prompt != st.session_state.last_selected_prompt:
            st.session_state.last_selected_prompt = selected_prompt
            load_selected_prompt(selected_prompt)

    # Use the callback for the selectbox
    selected_prompt = st.sidebar.selectbox(
        "Select a saved prompt:",
        prompt_list,
        index=prompt_list.index(st.session_state.last_selected_prompt),
        key="prompt_selector",
        on_change=on_prompt_change,
    )

    # If it's the first time or if we haven't loaded the prompt yet, load it now
    if "modified_prompt" not in st.session_state:
        load_selected_prompt(selected_prompt)

    return st.session_state.modified_prompt


def load_selected_prompt(selected_prompt: str) -> None:
    """
    Load the selected prompt and update the session state.

    Args:
        selected_prompt (str): The name of the selected prompt file.
    """
    if selected_prompt == "Current Baseline":
        loaded_prompt = BASELINE_PROMPT
    else:
        try:
            with open(os.path.join(PROMPTS_DIR, selected_prompt), "r") as f:
                loaded_prompt = f.read()
        except FileNotFoundError:
            logger.error(f"Selected prompt file {selected_prompt} not found.")
            st.sidebar.error(
                f"Selected prompt file {selected_prompt} not found. Using default prompt."
            )
            loaded_prompt = BASELINE_PROMPT

    # Update the session state with the newly loaded prompt
    st.session_state.modified_prompt = loaded_prompt


def display_prompt_dev_box(baseline_prompt: str, df: pd.DataFrame) -> Tuple[str, str]:
    """
    Display the prompt development box for modifying the baseline prompt.

    Args:
        baseline_prompt (str): Initial baseline prompt.
        df (pd.DataFrame): Dataframe containing SME feedback.

    Returns:
        Tuple[str, str]: Modified prompt and selected model.
    """
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
                if feedback:
                    st.markdown(f"- {feedback}")

    return modified_prompt, model


def calculate_metrics(auto_evaled_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate metrics for the auto-evaluated responses.

    Args:
        auto_evaled_df (pd.DataFrame): Dataframe with auto-evaluated responses.

    Returns:
        Dict[str, float]: Dictionary containing calculated metrics.
    """
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


def display_metrics(current_metrics: Dict[str, float]) -> None:
    """
    Display the calculated metrics with tooltips for descriptions and deltas.

    Args:
        current_metrics (Dict[str, float]): Dictionary of current metrics.
    """
    st.subheader("Evaluation Metrics")
    col1, col2, col3 = st.columns(3)

    previous_metrics = st.session_state.get("previous_metrics", {})

    def get_delta(
        current: Dict[str, float], previous: Dict[str, float], key: str
    ) -> Tuple[Optional[str], Optional[str]]:
        delta_color = "normal"
        if previous and key in previous:
            delta = current[key] - previous[key]
            delta_str = f"{delta:+.2f}%"
            if key == "regression_percentage":
                delta_color = "inverse"  # Use 'inverse' for regressions
            return delta_str, delta_color
        return None, delta_color

    metrics_config = [
        (
            "Accepted Responses",
            "accept_percentage",
            f"Percentage of total responses that were accepted in this iteration. ({current_metrics['accepted_responses']} / {current_metrics['total_responses']})",
        ),
        (
            "Regressions",
            "regression_percentage",
            f"Percentage of responses that were previously ACCEPT but now REJECT, out of all previously accepted responses. ({current_metrics['regressions']} regressions)",
        ),
        (
            "Improvements",
            "improvement_percentage",
            f"Percentage of responses that were previously REJECT with SME feedback and are now ACCEPT, out of all previously rejected responses. ({current_metrics['improvements']} improvements)",
        ),
    ]

    for col, (label, key, help_text) in zip([col1, col2, col3], metrics_config):
        with col:
            delta, color = get_delta(current_metrics, previous_metrics, key)
            st.metric(
                label=label,
                value=f"{current_metrics[key]:.2f}%",
                delta=delta,
                delta_color=color,
                help=help_text,
            )

    st.session_state.previous_metrics = current_metrics


def preview_prompt(df: pd.DataFrame, modified_prompt: str, model: str) -> None:
    """
    Preview the modified prompt by generating new responses and displaying results.

    Args:
        df (pd.DataFrame): Original dataframe with questions.
        modified_prompt (str): Modified prompt to be tested.
        model (str): Selected language model for generation.
    """
    if st.button("Preview Prompt"):
        new_responses = generate_new_responses(df, modified_prompt, model)
        auto_eval_df = df.copy()
        auto_eval_df["new_response"] = new_responses

        auto_eval_df["old_auto_evaluation"] = (
            st.session_state.auto_evaled_df.get("auto_evaluation", "UNKNOWN")
            if "auto_evaled_df" in st.session_state
            else "UNKNOWN"
        )

        auto_evaled_df = auto_evaluate_responses(auto_eval_df)

        metrics = calculate_metrics(auto_evaled_df)
        display_metrics(metrics)

        display_preview_results(auto_evaled_df)
        st.session_state.auto_evaled_df = auto_evaled_df


def generate_new_responses(
    df: pd.DataFrame, modified_prompt: str, model: str
) -> List[str]:
    """
    Generate new responses using the modified prompt.

    Args:
        df (pd.DataFrame): Dataframe containing questions.
        modified_prompt (str): Modified prompt to use for generation.
        model (str): Selected language model.

    Returns:
        List[str]: List of generated responses.
    """
    new_responses = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_rows = len(df)

    for index, row in df.iterrows():
        logger.info("=" * 10)
        logger.info(f"Inference index {index}")
        question = row["question"]
        formatted_prompt = modified_prompt.format(user_question=question)
        response = query_llm(formatted_prompt, model=model)
        new_responses.append(response)

        progress = float(index + 1) / float(total_rows)
        progress_bar.progress(progress)
        progress_text.text(f"Generating new responses: {index + 1}/{total_rows}")

    progress_bar.empty()
    progress_text.empty()
    return new_responses


def display_preview_results(auto_evaled_df: pd.DataFrame) -> None:
    """
    Display the results of the preview with highlighted rows for improvements and regressions.

    Args:
        auto_evaled_df (pd.DataFrame): Dataframe with auto-evaluated new responses.
    """
    st.write("Responses generated with the modified prompt:")

    # Add legend

    auto_evaled_df["row_color"] = "white"
    auto_evaled_df.loc[
        (auto_evaled_df["old_auto_evaluation"] == "REJECT")
        & (auto_evaled_df["auto_evaluation"] == "ACCEPT"),
        "row_color",
    ] = "lightgreen"
    auto_evaled_df.loc[
        (auto_evaled_df["old_auto_evaluation"] == "ACCEPT")
        & (auto_evaled_df["auto_evaluation"] == "REJECT"),
        "row_color",
    ] = "lightcoral"

    display_df = auto_evaled_df[
        [
            "question",
            "response",
            "new_response",
            "auto_evaluation",
            "rationale",
            "row_color",
        ]
    ].copy()

    column_config = {
        "question": st.column_config.TextColumn("Question"),
        "response": st.column_config.TextColumn("Original Response", width="large"),
        "new_response": st.column_config.TextColumn("New Response", width="large"),
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

    def highlight_row_of_df(row):
        return ["background-color: " + row["row_color"]] * len(row)

    selection = st.dataframe(
        display_df.style.apply(highlight_row_of_df, axis=1),
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        column_order=[
            "question",
            "response",
            "new_response",
            "auto_evaluation",
            "rationale",
        ],
    )
    # Add a legend to explain the row highlighting colors
    st.markdown(
        """
    <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
        <span style="margin-right: 10px;">Row Highlighting Legend:</span>
        <div style="margin-right: 20px;">
            <span style="background-color: lightgreen; padding: 2px 5px;">■</span> Improvement (REJECT → ACCEPT)
        </div>
        <div>
            <span style="background-color: lightcoral; padding: 2px 5px;">■</span> Regression (ACCEPT → REJECT)
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    # The legend is displayed as HTML for better formatting
    # It shows two colored boxes with explanations:
    # - Light green for improvements (responses that changed from REJECT to ACCEPT)
    # - Light coral for regressions (responses that changed from ACCEPT to REJECT)
    # This helps users quickly understand the meaning of the row colors in the dataframe


def save_prompt_and_responses() -> None:
    """Save the latest prompt and generated responses for SME evaluation."""
    if st.button("Save Prompt"):
        if (
            "auto_evaled_df" in st.session_state
            and "modified_prompt" in st.session_state
        ):
            # Generate a unique identifier for this save operation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

            # Save the responses
            df_to_save = st.session_state.auto_evaled_df.rename(
                columns={
                    "response": "old_response",
                    "new_response": "response",
                    "rating": "old_rating",
                }
            )
            responses_filename = f"experiment_responses_{timestamp}.csv"
            responses_path = os.path.join(ITERATION_RESPONSES_DIR, responses_filename)

            # Save the prompt
            prompt_filename = f"prompt_{timestamp}.txt"
            prompt_path = os.path.join(PROMPTS_DIR, prompt_filename)

            # Create a mapping entry
            mapping_entry = {
                "timestamp": timestamp,
                "prompt_file": prompt_filename,
                "responses_file": responses_filename,
            }

            try:
                # Save responses
                os.makedirs(os.path.dirname(responses_path), exist_ok=True)
                df_to_save.to_csv(responses_path, index=False)

                # Save prompt
                os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
                with open(prompt_path, "w") as f:
                    f.write(st.session_state.modified_prompt)

                # Update the mapping file
                if os.path.exists(MAPPING_FILE):
                    with open(MAPPING_FILE, "r") as f:
                        mapping = json.load(f)
                else:
                    mapping = []

                mapping.append(mapping_entry)

                with open(MAPPING_FILE, "w") as f:
                    json.dump(mapping, f, indent=2)

                st.success("Prompt and responses saved successfully!")
                logger.info(f"Saved prompt and responses with timestamp {timestamp}")
            except Exception as e:
                st.error(f"Error saving prompt and responses: {str(e)}")
                logger.error(f"Error saving prompt and responses: {str(e)}")
        else:
            st.error("No responses to save. Please preview the prompt first.")


def iterate_on_specific_question(
    filtered_df: pd.DataFrame, row_index: int, baseline_prompt: str
) -> None:
    """
    Allow user to iterate on a specific question with an improved UI.

    Args:
        filtered_df (pd.DataFrame): Dataframe containing the selected question.
        row_index (int): Index of the selected row.
        baseline_prompt (str): Initial baseline prompt.
    """
    st.subheader("🔍 Iterate on Specific Question")

    row = filtered_df.iloc[0]
    question, original_response, rating = (
        row["question"],
        row["response"],
        row["rating"],
    )
    edited_gt, sme_feedback = row["edited_gt"], row["sme_feedback"]

    # Display question, original response, and rating in three columns
    col1, col2, col3 = st.columns(3)
    for col, title, content in zip(
        [col1, col2, col3],
        ["Question", "Original Response", "Rating"],
        [question, original_response, rating],
    ):
        with col:
            st.markdown(
                f"<h4 style='text-align: center;'>{title}</h4>", unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='text-align: center; background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>{content}</div>",
                unsafe_allow_html=True,
            )

    if edited_gt:
        st.markdown(f"#### **Edited Ground Truth:**\n{edited_gt}")

    modified_prompt, model = display_prompt_dev_box(baseline_prompt, filtered_df)

    if st.button("Show All Questions"):
        st.session_state.selected_row_index = -1
        st.session_state.filtered_df = pd.DataFrame()
        st.rerun()

    if st.button("Generate New Response", key="generate_specific"):
        with st.spinner("Generating response..."):
            formatted_prompt = modified_prompt.format(user_question=question)
            new_response = query_llm(formatted_prompt, model=model)

        st.markdown("### New Response")
        st.markdown(new_response)

        # Auto-evaluate the new response
        auto_eval_df = pd.DataFrame(
            {
                "question": [question],
                "response": [original_response],
                "new_response": [new_response],
                "rating": [rating],
                "edited_gt": [edited_gt],
                "sme_feedback": [sme_feedback],
            }
        )
        auto_evaled_df = auto_evaluate_responses(auto_eval_df)

        # Display auto-evaluation result with icon
        auto_eval_result = auto_evaled_df.loc[0, "auto_evaluation"]
        auto_eval_icon = "✅" if auto_eval_result == "ACCEPT" else "❌"
        st.markdown(f"### Auto Evaluation\n{auto_eval_icon} {auto_eval_result}")

        # Display rationale in an expander
        with st.expander("See Evaluation Rationale"):
            st.markdown(auto_evaled_df.loc[0, "rationale"])


def main() -> None:
    """Main function to run the Streamlit app for prompt iteration."""
    setup_page()
    logger, logging_level = setup_logging(__name__)
    st.session_state["logging_level"] = logging_level
    df = load_data()
    display_evaluated_responses(df)
    baseline_prompt = load_prompt()
    if st.session_state.get("selected_row_index", -1) >= 0:
        iterate_on_specific_question(
            st.session_state.filtered_df,
            st.session_state.selected_row_index,
            baseline_prompt,
        )
    else:
        modified_prompt, model = display_prompt_dev_box(baseline_prompt, df)
        st.session_state.modified_prompt = (
            modified_prompt  # Store the modified prompt in session state
        )
        # print(modified_prompt)
        preview_prompt(df, modified_prompt, model)
        save_prompt_and_responses()  # Replace send_for_sme_evaluation with this new function


if __name__ == "__main__":
    main()
