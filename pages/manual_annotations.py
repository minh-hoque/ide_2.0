import streamlit as st
import pandas as pd
from css.style import apply_snorkel_style
from helper.logging import get_logger, setup_logging
from typing import Tuple
import os

logger = get_logger(__name__)


def setup_page():
    """
    Set up the Streamlit page configuration for manual annotations.
    """
    st.set_page_config(
        page_title="Manual Annotations", page_icon=":clipboard:", layout="wide"
    )
    st.markdown(apply_snorkel_style(), unsafe_allow_html=True)
    st.markdown('<h1 class="header">Manual Annotations</h1>', unsafe_allow_html=True)


def display_question_and_response(
    index: int, row: pd.Series, is_iteration: bool = False
):
    """
    Display a single question and its response.

    Args:
        index (int): The index of the current question.
        row (pd.Series): A row from the DataFrame containing question and response data.
        is_iteration (bool, optional): Whether this is an iteration of responses. Defaults to False.
    """
    question_number = index + 1
    question_text = row["question"]
    response_text = row["response"]

    if is_iteration:
        auto_eval_icon = "✅" if row.get("auto_evaluation") == "ACCEPT" else "❌"
        rationale = row.get("rationale", "")

        st.markdown(
            f"##### **Question {question_number}**: {question_text} ➪ Auto Evaluation: {auto_eval_icon}",
            help=rationale,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'
                f'<strong>Old Response:</strong><br>{row["old_response"]}'
                f"</div>",
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'
                f"<strong>New Response:</strong><br>{response_text}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(f"##### **Question {question_number}**: {question_text}")
        st.write(f"**Response:** {response_text}")


def get_user_feedback(
    index: int, rating: str, old_feedback: str = ""
) -> Tuple[str, str]:
    """
    Get user feedback for a response.

    Args:
        index (int): The index of the current question.
        rating (str): The current rating of the response.
        old_feedback (str, optional): Previous feedback, if any. Defaults to "".

    Returns:
        Tuple[str, str]: A tuple containing the edited ground truth and SME feedback.
    """
    if rating == "ACCEPT":
        return "", ""

    edited_gt = st.text_area(
        "Provide the correct ground truth response (optional):",
        key=f"ground_truth_{index}",
    )
    sme_feedback = st.text_area(
        "Provide feedback for the data scientist:",
        value=old_feedback,
        key=f"feedback_{index}",
        placeholder="Enter your feedback here...",
    )
    return edited_gt, sme_feedback


def update_annotation(
    df: pd.DataFrame, index: int, rating: str, edited_gt: str, sme_feedback: str
) -> pd.DataFrame:
    """
    Update the annotation for a specific row in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing all annotations.
        index (int): The index of the row to update.
        rating (str): The new rating for the response.
        edited_gt (str): The edited ground truth, if any.
        sme_feedback (str): The SME feedback, if any.

    Returns:
        pd.DataFrame: The updated DataFrame.
    """
    df.loc[index, "rating"] = rating
    df.loc[index, "sme_feedback"] = sme_feedback
    df.loc[index, "edited_gt"] = edited_gt
    return df


def save_annotations(df: pd.DataFrame, original_filename: str):
    """
    Save the annotated DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame containing all annotations.
        original_filename (str): The name of the original uploaded file.
    """
    if st.button("Submit Annotations"):
        filename, ext = os.path.splitext(original_filename)
        output_filename = f"{filename}_evaluated{ext}"
        output_path = os.path.join("./storage/manual_annotations", output_filename)
        df.to_csv(output_path, index=False)
        st.success(f"Annotations submitted successfully! Saved as {output_filename}")


def display_and_rate_response(
    index: int, row: pd.Series, is_iteration: bool, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Display a response and allow the user to rate it.
    """
    display_question_and_response(index, row, is_iteration)

    default_index = 0  # Default to ACCEPT
    if is_iteration:
        default_index = 0 if row.get("auto_evaluation") == "ACCEPT" else 1
    elif "rating" in row:
        default_index = 0 if row["rating"] == "ACCEPT" else 1

    rating = st.radio(
        f"Rate the response for Question {index + 1}:",
        ("ACCEPT", "REJECT"),
        key=f"rating_{index}",
        index=default_index,
    )

    old_feedback = row.get("sme_feedback", "")
    edited_gt, sme_feedback = get_user_feedback(index, rating, old_feedback)
    return update_annotation(df, index, rating, edited_gt, sme_feedback)


def process_annotations(uploaded_file):
    """
    Process the uploaded CSV file containing responses for annotation.

    Args:
        uploaded_file: The uploaded CSV file containing responses.
    """
    df = pd.read_csv(uploaded_file, na_values=["", "nan", "NaN", "None"])
    required_columns = ["question", "response"]

    if not set(required_columns).issubset(df.columns):
        st.error(
            "The uploaded CSV file must contain 'question' and 'response' columns."
        )
        return

    is_iteration = "rationale" in df.columns and "auto_evaluation" in df.columns

    if is_iteration:
        st.markdown("#### Displaying responses after prompt iteration")

    highlight_rejects = st.checkbox("Highlight REJECT responses")

    for index, row in df.iterrows():
        use_expander = highlight_rejects and row.get("auto_evaluation") == "ACCEPT"

        if use_expander:
            with st.expander(f"Question {index + 1} (ACCEPT)"):
                df = display_and_rate_response(index, row, is_iteration, df)
        else:
            df = display_and_rate_response(index, row, is_iteration, df)

        st.markdown("<hr>", unsafe_allow_html=True)

    save_annotations(df, uploaded_file.name)


def main():
    """
    Main function to run the Streamlit app for manual annotations.
    """
    setup_page()
    logger, logging_level = setup_logging(__name__)
    st.session_state["logging_level"] = logging_level
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        process_annotations(uploaded_file)
    else:
        st.info("Please upload a CSV file to begin annotating.")


if __name__ == "__main__":
    main()
