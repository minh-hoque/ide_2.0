import os
import json
import re
from datetime import datetime
from typing import Dict, List, Union
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import html
from css.style import apply_snorkel_style
from helper.llms import query_gpt4, query_structured_gpt4, ExtractionResult
from helper.logging import get_logger
from prompts.extraction_prompts import EXTRACT_PROMPT


# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get a logger for this module
logger = get_logger(__name__)


def setup_page() -> None:
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Extraction Prompt Iteration", page_icon=":mag:", layout="wide"
    )
    st.markdown(apply_snorkel_style(), unsafe_allow_html=True)
    st.markdown(
        '<h1 class="header">Extraction Prompt Development Workflow</h1>',
        unsafe_allow_html=True,
    )


def load_data() -> pd.DataFrame:
    """Load and prepare the dataframe from the CSV file."""
    csv_files = [
        f for f in os.listdir("./storage/extraction_data") if f.endswith(".csv")
    ]

    if not csv_files:
        st.error("No CSV files found in the extraction_data directory.")
        st.stop()

    selected_file = st.selectbox("Select CSV file:", ["-"] + csv_files)

    if selected_file == "-":
        st.error("No file selected. Please try again.")
        st.stop()

    try:
        df = pd.read_csv(os.path.join("./storage/extraction_data", selected_file))
        st.session_state.df = df  # Store df in session state
        return df
    except Exception as e:
        logger.error(f"Error loading file {selected_file}: {str(e)}")
        st.error(f"Error loading file {selected_file}. Please check the file format.")
        st.stop()


def load_prompt() -> str:
    """Load the baseline prompt from file or use default if not found."""
    st.sidebar.markdown("## Load Saved Prompt")
    saved_prompts = load_saved_prompts()
    selected_prompt = st.sidebar.selectbox(
        "Select a saved prompt:", ["Current Baseline"] + saved_prompts, index=0
    )

    if selected_prompt == "Current Baseline":
        return EXTRACT_PROMPT
    else:
        try:
            with open(
                os.path.join("./storage/extraction_prompts", selected_prompt), "r"
            ) as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Selected prompt file {selected_prompt} not found.")
            st.sidebar.error(
                f"Selected prompt file {selected_prompt} not found. Using default prompt."
            )
            return "Extract the following entities from the text: {entities}\n\nText: {text}\n\nExtracted entities:"


def load_saved_prompts():
    """Load the list of saved prompts from the storage directory."""
    prompts_dir = "./storage/extraction_prompts"
    if not os.path.exists(prompts_dir):
        return []

    prompt_files = [f for f in os.listdir(prompts_dir) if f.endswith(".txt")]
    prompt_files.sort(reverse=True)  # Sort files in reverse order (newest first)
    return prompt_files


def display_prompt_dev_box(baseline_prompt: str) -> str:
    """Display the prompt development box for modifying the baseline prompt."""
    st.subheader("Prompt Dev Box")
    st.write("Modify the baseline prompt for entity extraction.")

    modified_prompt = st.text_area("Modified Prompt", value=baseline_prompt, height=300)
    return modified_prompt


def parse_gpt4_output(output: str) -> Dict[str, List[str]]:
    """
    Parse the GPT-4 output from a string that may include markdown JSON formatting.

    Args:
        output (str): The output from GPT-4, potentially including markdown formatting.

    Returns:
        Dict[str, List[str]]: A dictionary where keys are entity types and values are lists of extracted entities.
    """
    # Remove markdown formatting if present
    json_match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = output

    try:
        parsed_output = json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Error parsing GPT-4 output: {output}")
        return {}

    # Ensure all values are lists
    return {k: v if isinstance(v, list) else [v] for k, v in parsed_output.items()}


def generate_extractions(
    df: pd.DataFrame, modified_prompt: str, model: str
) -> List[Dict[str, List[str]]]:
    """Generate extractions using the modified prompt."""
    extractions = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_rows = len(df)

    for index, row in df.iterrows():
        logger.info(f"Extraction for index {index}")
        text = row["text"]
        formatted_prompt = modified_prompt.format(text=text)
        extraction_response = query_gpt4(formatted_prompt)
        print("extraction_response")
        print(extraction_response, type(extraction_response))
        parsed_extraction = parse_gpt4_output(extraction_response)
        print("parsed_extraction")
        print(parsed_extraction, type(parsed_extraction))
        extractions.append(parsed_extraction)

        progress = float(index + 1) / float(total_rows)
        progress_bar.progress(progress)
        progress_text.text(f"Generating extractions: {index + 1}/{total_rows}")

    progress_bar.empty()
    progress_text.empty()
    return extractions


def calculate_example_metrics(
    ground_truth: Dict, extraction: Dict
) -> Dict[str, Dict[str, float]]:
    """Calculate metrics for a single example."""
    entity_metrics = {}
    for entity, gt_values in ground_truth.items():
        ext_values = extraction.get(entity, [])
        tp = len(set(gt_values) & set(ext_values))
        fp = len(set(ext_values) - set(gt_values))
        fn = len(set(gt_values) - set(ext_values))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        entity_metrics[entity] = {"precision": precision, "recall": recall, "f1": f1}
    return entity_metrics


def calculate_metrics(
    ground_truth: List[Dict], extractions: List[Dict]
) -> Dict[str, float]:
    """Calculate metrics for the extractions."""
    entity_metrics = {}
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    entity_count = 0
    # print(ground_truth)
    # print(type(ground_truth))
    for i, (gt_str, ext) in enumerate(zip(ground_truth, extractions)):
        try:
            gt = json.loads(gt_str)
            for entity, gt_values in gt.items():
                if entity not in entity_metrics:
                    entity_metrics[entity] = {"tp": 0, "fp": 0, "fn": 0}

                ext_values = ext.get(entity, [])

                entity_metrics[entity]["tp"] += len(set(gt_values) & set(ext_values))
                entity_metrics[entity]["fp"] += len(set(ext_values) - set(gt_values))
                entity_metrics[entity]["fn"] += len(set(gt_values) - set(ext_values))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for entry {i}:")
            print(f"String: {gt_str}")
            print(f"Error: {str(e)}")
            continue

    for entity, metrics in entity_metrics.items():
        precision = (
            metrics["tp"] / (metrics["tp"] + metrics["fp"])
            if (metrics["tp"] + metrics["fp"]) > 0
            else 0
        )
        recall = (
            metrics["tp"] / (metrics["tp"] + metrics["fn"])
            if (metrics["tp"] + metrics["fn"]) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        entity_metrics[entity]["precision"] = precision
        entity_metrics[entity]["recall"] = recall
        entity_metrics[entity]["f1"] = f1

        total_precision += precision
        total_recall += recall
        total_f1 += f1
        entity_count += 1

    avg_precision = total_precision / entity_count if entity_count > 0 else 0
    avg_recall = total_recall / entity_count if entity_count > 0 else 0
    avg_f1 = total_f1 / entity_count if entity_count > 0 else 0

    return {
        "entity_metrics": entity_metrics,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
    }


def display_metrics(metrics: Dict[str, float]) -> None:
    """Display the calculated metrics."""
    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Precision", f"{metrics['avg_precision']:.2f}")
    col2.metric("Average Recall", f"{metrics['avg_recall']:.2f}")
    col3.metric("Average F1 Score", f"{metrics['avg_f1']:.2f}")

    with st.expander("Entity-level Metrics"):
        for entity, entity_metrics in metrics["entity_metrics"].items():
            st.markdown(f"#### {entity}")
            subcol1, subcol2, subcol3 = st.columns(3)
            subcol1.metric("Precision", f"{entity_metrics['precision']:.2f}")
            subcol2.metric("Recall", f"{entity_metrics['recall']:.2f}")
            subcol3.metric("F1 Score", f"{entity_metrics['f1']:.2f}")


def preview_prompt(df: pd.DataFrame, modified_prompt: str, model: str) -> None:
    """Preview the modified prompt by generating extractions and displaying results."""
    if st.button("Preview Prompt"):
        extractions = generate_extractions(df, modified_prompt, model)
        metrics = calculate_metrics(df["ground_truth"].tolist(), extractions)
        display_metrics(metrics)
        display_preview_results(df, extractions)
        st.session_state.extractions = extractions
        st.session_state.metrics = metrics


def display_preview_results(df: pd.DataFrame, extractions: List[Dict]) -> None:
    """Display the results of the preview."""
    st.write("Extractions generated with the modified prompt:")

    for i, (_, row) in enumerate(df.iterrows()):
        with st.expander(f"Example {i+1}"):
            ground_truth = json.loads(row["ground_truth"])

            metrics = calculate_example_metrics(ground_truth, extractions[i])
            st.markdown("### Entity Metrics")
            table_rows = [
                f"| {entity} | {scores['precision']:.2f} | {scores['recall']:.2f} | {scores['f1']:.2f} |"
                for entity, scores in metrics.items()
            ]
            table_header = "| Entity | Precision | Recall | F1 Score |"
            table_separator = "|--------|-----------|--------|----------|"
            table_content = "\n".join([table_header, table_separator] + table_rows)
            st.markdown(table_content)

            st.write("**Text:**")
            # Decode and unescape HTML entities
            cleaned_text = html.unescape(row["text"])
            st.text(cleaned_text)

            st.write("**Ground Truth:**")
            st.json(ground_truth)

            st.write("**Extracted:**")
            st.json(extractions[i])

            st.write("**Entity-level Metrics:**")


def save_prompt_and_extractions() -> None:
    """Save the latest prompt and generated extractions."""
    if st.button("Save Prompt"):
        if (
            "extractions" in st.session_state
            and "modified_prompt" in st.session_state
            and "df" in st.session_state
        ):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

            # Save the extractions as CSV
            extractions_filename = f"experiment_extractions_{timestamp}.csv"
            extractions_path = os.path.join(
                "./storage/extraction_results", extractions_filename
            )

            # Save the prompt
            prompt_filename = f"extraction_prompt_{timestamp}.txt"
            prompt_path = os.path.join("./storage/extraction_prompts", prompt_filename)

            # Create a mapping entry
            mapping_entry = {
                "timestamp": timestamp,
                "prompt_file": prompt_filename,
                "extractions_file": extractions_filename,
                "metrics": st.session_state.metrics,
            }

            try:
                # Combine original DataFrame with extractions
                combined_df = st.session_state.df.copy()
                combined_df["extractions"] = st.session_state.extractions

                # Save extractions as CSV
                os.makedirs(os.path.dirname(extractions_path), exist_ok=True)
                combined_df.to_csv(extractions_path, index=False)

                # Save prompt
                os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
                with open(prompt_path, "w") as f:
                    f.write(st.session_state.modified_prompt)

                # Update the mapping file
                mapping_file = "./storage/extraction_prompt_mapping.json"
                if os.path.exists(mapping_file):
                    with open(mapping_file, "r") as f:
                        mapping = json.load(f)
                else:
                    mapping = []

                mapping.append(mapping_entry)

                with open(mapping_file, "w") as f:
                    json.dump(mapping, f, indent=2)

                st.success("Prompt and extractions saved successfully!")
                logger.info(f"Saved prompt and extractions with timestamp {timestamp}")
            except Exception as e:
                st.error(f"Error saving prompt and extractions: {str(e)}")
                logger.error(f"Error saving prompt and extractions: {str(e)}")
        else:
            st.error("No extractions or data to save. Please preview the prompt first.")


def display_evaluated_extractions(df: pd.DataFrame) -> None:
    """
    Display the extractions in a dataframe and handle row selection.

    Args:
        df (pd.DataFrame): Dataframe containing extractions and ground truth.
    """
    st.subheader("Extraction Data")
    column_config = {
        "filename": st.column_config.TextColumn("Filename", width="medium"),
        "text": st.column_config.TextColumn("Text", width="medium"),
        "ground_truth": st.column_config.TextColumn("Ground Truth", width="large"),
    }

    if st.session_state.get("selected_row_index", -1) >= 0:
        displayed_columns = ["filename", "text", "ground_truth"]
        filtered_df = st.session_state.filtered_df[displayed_columns]
        st.dataframe(
            filtered_df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
        )
    else:
        displayed_columns = ["filename", "text", "ground_truth"]
        filtered_df = df[displayed_columns]
        selection = st.dataframe(
            filtered_df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            on_select=lambda selection: handle_selection(selection, df),
        )


def handle_selection(selection, df):
    if selection:
        st.session_state.selected_row_index = selection.index[0]
        st.session_state.filtered_df = df.iloc[[st.session_state.selected_row_index]]
        st.rerun()


def main() -> None:
    """Main function to run the Streamlit app for extraction prompt iteration."""
    setup_page()
    df = load_data()
    display_evaluated_extractions(df)  # Add this line
    baseline_prompt = load_prompt()

    modified_prompt = display_prompt_dev_box(baseline_prompt)
    st.session_state.modified_prompt = modified_prompt

    model = st.selectbox(
        "Select a model", ["gpt-4o", "gpt-4o-2024-08-06", "gpt-4-turbo-preview"]
    )

    preview_prompt(df, modified_prompt, model)
    save_prompt_and_extractions()


if __name__ == "__main__":
    main()
