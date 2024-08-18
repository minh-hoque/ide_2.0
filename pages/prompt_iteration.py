from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from css.style import apply_snorkel_style
from openai import OpenAI
import os
import logging
import colorlog
from prompts.auto_evaluation_prompts import (
    LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT,
    LLM_AS_A_JUDGE_SME_FEEDBACK_PROMPT,
)

# Load environment variables
load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging with color
logger = colorlog.getLogger(__name__)
if not logger.handlers:
    # Disable logging for other libraries
    logging.getLogger().setLevel(logging.WARNING)

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s:%(name)s:%(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent propagation to root logger


# Function to set logging level
def set_logging_level(level):
    logger.setLevel(level)


# Page configuration
st.set_page_config(page_title="Prompt Iteration", page_icon=":pencil2:", layout="wide")

# Apply the Snorkel style
st.markdown(apply_snorkel_style(), unsafe_allow_html=True)

# Main header
st.markdown(
    '<h1 class="header">Prompt Development Workflow</h1>', unsafe_allow_html=True
)

# Add logging level selector
logging_level = st.selectbox(
    "Select Logging Level",
    ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    index=1,  # Default to INFO
)
set_logging_level(logging_level)


# Function to query GPT-4
@st.cache_data
def query_gpt4(prompt, question):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying GPT-4: {str(e)}")
        return f"Error: {str(e)}"


def parse_auto_evaluation_response(result):
    # Parse the auto-evaluation response to extract rational and result
    rational_parts = result.lower().split("rationale:")
    result_parts = result.lower().split("result:")

    parsed_rational = ""
    parsed_result = "UNKNOWN"

    if len(rational_parts) > 1:
        parsed_rational = rational_parts[1].strip()

    if len(result_parts) > 1:
        parsed_result = result_parts[1].strip().upper()
        if parsed_result not in ["ACCEPT", "REJECT"]:
            parsed_result = "UNKNOWN"

    return parsed_rational, parsed_result


def auto_evaluate_responses(df):
    # Implement auto-evaluation logic here
    rational_list = []
    auto_evaluation_results = []
    # Check if new response is equivalent to CORRECT old response utilizing LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT
    # Create a progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for index, row in df.iterrows():
        # Update progress bar
        progress = (index + 1) / len(df)
        progress_bar.progress(progress)
        progress_text.text(f"Auto Evaluation Progress: {int(progress * 100)}%")

        logger.info(f"Processing Question {index + 1}")
        logger.debug(f"Question: {row['question']}")

        if row["rating"] == "ACCEPT":
            formated_prompt = LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT.format(
                old_response=row["response"], new_response=row["new_response"]
            )
            auto_evaluate_response = query_gpt4(formated_prompt, row["question"])

            logger.debug("LLM Response:")
            logger.debug(auto_evaluate_response)

            rational, auto_evaluation = parse_auto_evaluation_response(
                auto_evaluate_response
            )

            rational_list.append(rational)
            auto_evaluation_results.append(auto_evaluation)
            logger.debug("Old Response:")
            logger.debug(row["response"])
            logger.debug("New Response:")
            logger.debug(row["new_response"])
            logger.info(f"Auto Evaluation: {auto_evaluation}")
        else:
            if row["edited_gt"] != "nan":
                # Check if new response is equivalent SME edited_gt CORRECT response
                # Store results in a list
                formated_prompt = LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT.format(
                    old_response=row["edited_gt"], new_response=row["new_response"]
                )
                auto_evaluate_response = query_gpt4(formated_prompt, row["question"])
                logger.debug("LLM Response:")
                logger.debug(auto_evaluate_response)

                rational, auto_evaluation = parse_auto_evaluation_response(
                    auto_evaluate_response
                )

                rational_list.append(rational)
                auto_evaluation_results.append(auto_evaluation)
                logger.debug("Edited Ground Truth:")
                logger.debug(row["edited_gt"])
                logger.debug("New Response:")
                logger.debug(row["new_response"])
                logger.info(f"Auto Evaluation: {auto_evaluation}")

            elif row["sme_feedback"] != "nan":
                # Check if new response if SME feedback is incorporated into the new response
                # Store results in a list
                formated_prompt = LLM_AS_A_JUDGE_SME_FEEDBACK_PROMPT.format(
                    old_response=row["response"],
                    sme_feedback=row["sme_feedback"],
                    new_response=row["new_response"],
                )
                auto_evaluate_response = query_gpt4(
                    formated_prompt,
                    row["question"],
                )
                logger.debug("LLM Response:")
                logger.debug(auto_evaluate_response)

                rational, auto_evaluation = parse_auto_evaluation_response(
                    auto_evaluate_response
                )

                rational_list.append(rational)
                auto_evaluation_results.append(auto_evaluation)
                logger.debug("Old Response:")
                logger.debug(row["response"])
                logger.debug("SME Feedback:")
                logger.debug(row["sme_feedback"])
                logger.debug("New Response:")
                logger.debug(row["new_response"])
                logger.info(f"Auto Evaluation: {auto_evaluation}")

            else:
                # If there's no edited_gt or sme_feedback, we can't evaluate
                auto_evaluation_results.append("N/A")
                logger.warning("No edited ground truth or SME feedback available.")
                logger.info("Auto Evaluation: N/A")

        logger.info("-----------------------------------")

    # Clear the progress bar and text after completion
    progress_bar.empty()
    progress_text.empty()

    # Add the auto-evaluation and rational results to the dataframe
    df["auto_evaluation"] = auto_evaluation_results
    df["rational"] = rational_list

    return df


# # Page configuration
# st.set_page_config(page_title="Prompt Iteration", page_icon=":pencil2:", layout="wide")

# # Apply the Snorkel style
# st.markdown(apply_snorkel_style(), unsafe_allow_html=True)

# # Main header
# st.markdown('<h1 class="header">Prompt Iteration</h1>', unsafe_allow_html=True)

# Load the evaluated responses
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

    # Take 4 random rows from the dataframe
    st.session_state.df = st.session_state.df.sample(n=4, random_state=0)
    st.session_state.df = st.session_state.df.reset_index(drop=True)

df = st.session_state.df

# convert edited_gt and sme_feedback to string
df["edited_gt"] = df["edited_gt"].astype(str)
df["sme_feedback"] = df["sme_feedback"].astype(str)

# Display the evaluated responses
st.subheader("Evaluated Responses")

# Define column configurations
column_config = {
    "question": st.column_config.TextColumn("Question", width="medium"),
    "response": st.column_config.TextColumn("Response", width="medium"),
    "rating": st.column_config.TextColumn("Rating", width="small"),
    "edited_gt": st.column_config.TextColumn("Edited Ground Truth", width="large"),
    "sme_feedback": st.column_config.TextColumn("SME Feedback", width="large"),
}

# Display the dataframe using st.data_editor
st.data_editor(
    df,
    column_config=column_config,
    hide_index=True,
    num_rows="fixed",
    use_container_width=True,
)

# Prompt playground
st.subheader("Prompt Playground")
st.write("Modify the baseline prompt based on the feedback and evaluated responses.")

# Load the baseline prompt (assuming it's stored in a file)
try:
    with open("./storage/baseline_prompt.txt", "r") as f:
        baseline_prompt = f.read()
except FileNotFoundError:
    logger.warning("No baseline prompt found.")
    baseline_prompt = "No baseline prompt found. Please create one."

modified_prompt = st.text_area("Modified Prompt", value=baseline_prompt, height=600)

if st.button("Preview Prompt"):
    # Run the modified prompt through the LLM and use auto-evaluation using edited ground truth and SME feedback to assess the quality of the response
    # Generate responses for all questions
    new_responses = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_rows = len(df)
    for index, row in df.iterrows():
        logger.info(f"Processing index {index}")
        question = row["question"]
        formated_prompt = modified_prompt.format(user_question=question)
        response = query_gpt4(modified_prompt, question)
        new_responses.append(response)

        logger.debug(f"Processed {index + 1} out of {total_rows}")
        progress = float(index + 1) / float(total_rows)
        progress_bar.progress(progress)
        progress_text.text(f"Generating new responses: {index + 1}/{total_rows}")
    progress_bar.empty()
    progress_text.empty()

    # Create a copy of the dataframe for auto-evaluation
    auto_eval_df = df.copy()

    # Add new responses to the dataframe
    auto_eval_df["new_response"] = new_responses

    # Run auto-evaluation on the new responses
    auto_evaled_df = auto_evaluate_responses(auto_eval_df)

    # Display responses in a dataframe
    st.write("Responses generated with the modified prompt:")

    # Create a new dataframe with the columns we want to display
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

    # Define column configuration
    # Define image paths for ACCEPT and REJECT
    accept_image = "images/green_checkmark.png"
    reject_image = "images/red_x.png"

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

    # Function to assign images based on auto_evaluation result
    def get_evaluation_image(result):
        return "✅" if result == "ACCEPT" else "❌"

    # Assign images to auto_evaluation column
    display_df["auto_evaluation"] = display_df["auto_evaluation"].apply(
        get_evaluation_image
    )

    # Display the dataframe
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
