import streamlit as st
import pandas as pd
from css.style import apply_snorkel_style

# Helper functions


def setup_page():
    """Set up the page configuration and apply styles."""
    st.set_page_config(
        page_title="Manual Annotations", page_icon=":clipboard:", layout="wide"
    )
    st.markdown(apply_snorkel_style(), unsafe_allow_html=True)
    st.markdown('<h1 class="header">Manual Annotations</h1>', unsafe_allow_html=True)


def display_question_and_response(index, row, is_iteration=False):
    """Display question and response(s) for a given row."""
    icon = "✅" if row.get("auto_evaluation") == "ACCEPT" else "❌"
    st.markdown(
        f"##### **Question {index+1}**: {row['question']} {'➪ Auto Evaluation: ' + icon if is_iteration else ''}",
        help=row.get("rationale", ""),
    )

    if is_iteration:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Old Response:**")
            st.write(row["old_response"])
        with col2:
            st.write("**New Response:**")
            st.write(row["response"])
    else:
        st.write(f"**Response:** {row['response']}")


def get_user_input(index, rating):
    """Get user input for rating, edited ground truth, and SME feedback."""
    edited_gt = ""
    sme_feedback = ""

    if rating == "REJECT":
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


def update_dataframe(df, index, rating, edited_gt, sme_feedback):
    """Update the dataframe with user input."""
    df.loc[index, "rating"] = rating
    df.loc[index, "sme_feedback"] = sme_feedback
    df.loc[index, "edited_gt"] = edited_gt
    return df


def save_annotations(df):
    """Save the annotations to a CSV file."""
    if st.button("Submit Annotations"):
        df.to_csv("./storage/manual_annotations/evaluated_responses.csv", index=False)
        st.success("Annotations submitted successfully!")


def process_uploaded_file(uploaded_file):
    """Process the uploaded CSV file and handle annotations."""
    df = pd.read_csv(uploaded_file)
    required_columns = ["question", "response"]

    if not set(required_columns).issubset(df.columns):
        st.error(
            "The uploaded CSV file must contain at least the 'question' and 'response' columns."
        )
        return

    is_iteration = "rationale" in df.columns and "auto_evaluation" in df.columns

    if is_iteration:
        st.write("Displaying responses after prompt iteration")

    for index, row in df.iterrows():
        display_question_and_response(index, row, is_iteration)

        rating = st.radio(
            f"Rate the response for Question {index+1}:",
            ("ACCEPT", "REJECT"),
            key=f"rating_{index}",
        )

        edited_gt, sme_feedback = get_user_input(index, rating)
        df = update_dataframe(df, index, rating, edited_gt, sme_feedback)

        st.markdown("<hr>", unsafe_allow_html=True)

    save_annotations(df)


# Main execution
setup_page()

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    process_uploaded_file(uploaded_file)
else:
    st.info("Please upload a CSV file to begin annotating.")
