import os
from typing import Tuple, Dict, List
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit.delta_generator import DeltaGenerator

from css.style import apply_snorkel_style
from helper.llms import query_gpt4, auto_evaluate_responses
from helper.logging import get_logger
from prompts.base_prompts import PROMPT_4

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get a logger for this module
logger = get_logger(__name__)


def setup_page():
    """
    Set up the Streamlit page configuration.
    This function configures the page title, icon, layout, and applies custom CSS styling.
    """
    st.set_page_config(
        page_title="Prompt Iteration", page_icon=":pencil2:", layout="wide"
    )
    st.markdown(apply_snorkel_style(), unsafe_allow_html=True)
    st.markdown(
        '<h1 class="header">Prompt Development Workflow</h1>', unsafe_allow_html=True
    )


def setup_logging():
    """
    Set up logging level selector.
    Allows users to choose the logging level for the application.
    """
    logging_level = st.selectbox(
        "Select Logging Level",
        ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        index=1,
    )
    logger.setLevel(logging_level)


def load_data() -> pd.DataFrame:
    """
    Load and prepare the dataframe from evaluated responses.

    Returns:
        pd.DataFrame: Preprocessed dataframe containing evaluated responses.
    """
    # List all CSV files in the manual_annotations directory
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

    try:
        df = pd.read_csv(
            os.path.join("./storage/manual_annotations", selected_file),
            na_values=["", "nan", "NaN", "None"],
            keep_default_na=True,
        )
        return preprocess_dataframe(df)
    except FileNotFoundError:
        logger.error(f"Selected file {selected_file} not found.")
        st.error(f"Selected file {selected_file} not found. Please try again.")
        st.stop()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the loaded dataframe by dropping unnecessary columns and handling missing values.

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


def display_evaluated_responses(df: pd.DataFrame):
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

    selection = st.dataframe(
        df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",  # Changed to single selection for simplicity
    )
    selected_rows = selection.selection.rows
    # st.write(selected_rows[0])
    if selected_rows:
        st.session_state.selected_row_index = selected_rows[0]
        st.session_state.filtered_df = df.iloc[[st.session_state.selected_row_index]]
    else:
        st.session_state.selected_row_index = -1
        st.session_state.filtered_df = pd.DataFrame()


@st.cache_data
def load_baseline_prompt() -> str:
    """
    Load the baseline prompt from file or use default if not found.

    Returns:
        str: Baseline prompt text.
    """
    try:
        with open("./storage/baseline_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("No baseline prompt found. Using default prompt.")
        return PROMPT_4


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


def display_metrics(current_metrics: Dict[str, float]):
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
    ) -> Tuple[str, str]:
        if previous and key in previous:
            delta = current[key] - previous[key]
            delta_str = f"{delta:+.2f}%"
            delta_color = "green" if delta > 0 else "red"
            return delta_str, delta_color
        return None, None

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
            if key == "regression_percentage" and delta:
                color = (
                    "red" if color == "green" else "green"
                )  # Invert color for regressions
            st.metric(
                label=label,
                value=f"{current_metrics[key]:.2f}%",
                delta=delta,
                # delta_color=color,  # Uncomment if you want to use color
                help=help_text,
            )

    st.session_state.previous_metrics = current_metrics


def preview_prompt(df: pd.DataFrame, modified_prompt: str, model: str):
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
    """
    Display the results of the preview with highlighted rows for improvements and regressions.

    Args:
        auto_evaled_df (pd.DataFrame): Dataframe with auto-evaluated new responses.
    """
    st.write("Responses generated with the modified prompt:")

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

    st.dataframe(
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


def send_for_sme_evaluation():
    """
    Save the generated responses for SME evaluation.
    """
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


def iterate_on_specific_question(
    filtered_df: pd.DataFrame, row_index: int, baseline_prompt: str
):
    """
    Allow user to iterate on a specific question with an improved UI.

    Args:
        filtered_df (pd.DataFrame): Dataframe containing the selected question.
        row_index (int): Index of the selected row.
        baseline_prompt (str): Initial baseline prompt.
    """
    st.subheader("🔍 Iterate on Specific Question")

    row = filtered_df.iloc[0]
    question = row["question"]
    original_response = row["response"]
    rating = row["rating"]
    edited_gt = row["edited_gt"]
    sme_feedback = row["sme_feedback"]

    # Display question, original response, and rating in three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Question")
        st.markdown(question)
    with col2:
        st.markdown("#### Original Response")
        st.markdown(original_response)
    with col3:
        st.markdown("#### Rating")
        st.markdown(rating)

    # Display edited ground truth and SME feedback if available
    if edited_gt:
        st.markdown(f"#### **Edited Ground Truth:**\n{edited_gt}")
    if sme_feedback:
        st.markdown(f"#### **SME Feedback:**\n{sme_feedback}")

    modified_prompt, model = display_prompt_dev_box(baseline_prompt, row)

    # model = st.selectbox(
    #     "Select Model",
    #     ["gpt-4o", "gpt-4o-2024-08-06", "gpt-4-turbo-preview"],
    #     key="model_select_specific",
    # )
    # modified_prompt = st.text_area(
    #     "Modified Prompt", value=baseline_prompt, height=600, key="prompt_specific"
    # )

    # Generate new response
    if st.button("Generate New Response", key="generate_specific"):
        with st.spinner("Generating response..."):
            formatted_prompt = modified_prompt.format(user_question=question)
            new_response = query_gpt4(formatted_prompt, model=model)

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


def main():
    """
    Main function to run the Streamlit app for prompt iteration.
    """
    setup_page()
    setup_logging()
    df = load_data()
    display_evaluated_responses(df)
    baseline_prompt = load_baseline_prompt()

    if st.session_state.selected_row_index >= 0:
        iterate_on_specific_question(
            st.session_state.filtered_df,
            st.session_state.selected_row_index,
            baseline_prompt,
        )
    else:
        modified_prompt, model = display_prompt_dev_box(baseline_prompt, df)
        preview_prompt(df, modified_prompt, model)

        send_for_sme_evaluation()


if __name__ == "__main__":
    main()
