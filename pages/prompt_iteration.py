# Import necessary libraries
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
from helper.logging import get_logger
from openai import OpenAI

# Import custom modules
from css.style import apply_snorkel_style
from prompts.base_prompts import PROMPT_1
from helper.llms import query_gpt4, auto_evaluate_responses

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
        index=1,  # Default to INFO
    )
    logger.setLevel(logging_level)


def load_data():
    """Load and prepare the dataframe."""
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
        st.session_state.df = st.session_state.df.sample(
            n=4, random_state=0
        ).reset_index(drop=True)

    df = st.session_state.df
    df["edited_gt"] = df["edited_gt"].astype(str)
    df["sme_feedback"] = df["sme_feedback"].astype(str)
    return df


def display_evaluated_responses(df):
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


def load_baseline_prompt():
    """Load the baseline prompt from file or use default."""
    try:
        with open("./storage/baseline_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("No baseline prompt found.")
        return PROMPT_1


def display_prompt_dev_box(baseline_prompt, df):
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
                if feedback != "nan":
                    st.markdown(f"- {feedback}")

    return modified_prompt, model


def preview_prompt(df, modified_prompt, model):
    """Preview the modified prompt and generate new responses."""
    if st.button("Preview Prompt"):
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

        auto_eval_df = df.copy()
        auto_eval_df["new_response"] = new_responses
        auto_evaled_df = auto_evaluate_responses(auto_eval_df)

        display_preview_results(auto_evaled_df)
        st.session_state.auto_evaled_df = auto_evaled_df


def display_preview_results(auto_evaled_df):
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
        if st.session_state.auto_evaled_df is not None:
            df_to_save = st.session_state.auto_evaled_df.rename(
                columns={
                    "response": "old_response",
                    "new_response": "response",
                    "rating": "old_rating",
                }
            )
            try:
                df_to_save.to_csv(
                    "./storage/iteration_responses/new_responses.csv", index=False
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
