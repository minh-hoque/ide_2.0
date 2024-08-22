import streamlit as st
import pandas as pd
from css.style import apply_snorkel_style
from helper.logging import get_logger

logger = get_logger(__name__)


def setup_page():
    """Configure the page and apply styles."""
    st.set_page_config(
        page_title="Manual Annotations", page_icon=":clipboard:", layout="wide"
    )
    st.markdown(apply_snorkel_style(), unsafe_allow_html=True)
    st.markdown('<h1 class="header">Manual Annotations</h1>', unsafe_allow_html=True)


def display_question_and_response(
    index: int, row: pd.Series, is_iteration: bool = False
):
    """Render the question and response(s) for a given row."""
    question_number = index + 1
    question_text = row["question"]
    response_text = row["response"]

    if is_iteration:
        auto_eval_icon = "✅" if row.get("auto_evaluation") == "ACCEPT" else "❌"
        st.markdown(
            f"##### **Question {question_number}**: {question_text} ➪ Auto Evaluation: {auto_eval_icon}",
            help=row.get("rationale", ""),
        )
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Old Response:**")
            st.write(row["old_response"])
        with col2:
            st.write("**New Response:**")
            st.write(response_text)
    else:
        st.markdown(f"##### **Question {question_number}**: {question_text}")
        st.write(f"**Response:** {response_text}")


def get_user_feedback(index: int, rating: str):
    """Collect user feedback for rejected responses."""
    if rating == "ACCEPT":
        return "", ""

    edited_gt = st.text_area(
        "Provide the correct ground truth response (optional):",
        key=f"ground_truth_{index}",
    )
    sme_feedback = st.text_area(
        "Provide feedback for the data scientist:",
        key=f"feedback_{index}",
        placeholder="Enter your feedback here...",
    )
    return edited_gt, sme_feedback


def update_annotation(
    df: pd.DataFrame, index: int, rating: str, edited_gt: str, sme_feedback: str
):
    """Update the dataframe with user annotations."""
    df.loc[index, "rating"] = rating
    df.loc[index, "sme_feedback"] = sme_feedback
    df.loc[index, "edited_gt"] = edited_gt
    return df


def save_annotations(df: pd.DataFrame):
    """Save the annotated dataframe to a CSV file."""
    if st.button("Submit Annotations"):
        output_path = "./storage/manual_annotations/evaluated_responses_2.csv"
        df.to_csv(output_path, index=False)
        st.success("Annotations submitted successfully!")


def display_and_rate_response(
    index: int, row: pd.Series, is_iteration: bool, df: pd.DataFrame
):
    """Display the response and collect user rating."""
    display_question_and_response(index, row, is_iteration)

    default_index = 0 if row.get("auto_evaluation") == "ACCEPT" else 1
    rating = st.radio(
        f"Rate the response for Question {index + 1}:",
        ("ACCEPT", "REJECT"),
        key=f"rating_{index}",
        index=default_index,
    )

    edited_gt, sme_feedback = get_user_feedback(index, rating)
    df = update_annotation(df, index, rating, edited_gt, sme_feedback)


def process_annotations(uploaded_file):
    """Process the uploaded CSV file and handle annotations."""
    df = pd.read_csv(uploaded_file)
    required_columns = ["question", "response"]

    if not set(required_columns).issubset(df.columns):
        st.error(
            "The uploaded CSV file must contain 'question' and 'response' columns."
        )
        return

    is_iteration = "rationale" in df.columns and "auto_evaluation" in df.columns
    if is_iteration:
        st.write("Displaying responses after prompt iteration")

    highlight_rejects = st.checkbox("Highlight REJECT responses")

    for index, row in df.iterrows():
        use_expander = highlight_rejects and row.get("auto_evaluation") == "ACCEPT"

        if use_expander:
            with st.expander(f"Question {index + 1} (ACCEPT)"):
                display_and_rate_response(index, row, is_iteration, df)
        else:
            display_and_rate_response(index, row, is_iteration, df)

        st.markdown("<hr>", unsafe_allow_html=True)

    save_annotations(df)


def main():
    """Main execution function."""
    setup_page()
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        process_annotations(uploaded_file)
    else:
        st.info("Please upload a CSV file to begin annotating.")


if __name__ == "__main__":
    main()
