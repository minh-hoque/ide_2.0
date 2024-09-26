import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Union, Optional
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import html
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
import concurrent.futures

from css.style import apply_snorkel_style
from helper.llms import query_llm
from helper.logging import get_logger, setup_logging
from prompts.extraction_prompts import EXTRACT_PROMPT

# Constants for file paths
STORAGE_DIR = "./storage"
EXTRACTION_DATA_DIR = os.path.join(STORAGE_DIR, "extraction_data")
EXTRACTION_PROMPTS_DIR = os.path.join(STORAGE_DIR, "extraction_prompts")
EXTRACTION_RESULTS_DIR = os.path.join(STORAGE_DIR, "extraction_results")
PROMPT_MAPPING_FILE = os.path.join(STORAGE_DIR, "extraction_prompt_mapping.json")

# Load environment variables
load_dotenv()

# Get a logger for this module
logger = get_logger(__name__)


def setup_page() -> None:
    """
    Set up the Streamlit page configuration, including the page title, layout, custom CSS,
    and the sidebar for prompt mode selection.

    Also, initializes the 'prompt_mode' in session state.
    """
    st.set_page_config(
        page_title="Extraction Prompt Iteration", page_icon=":mag:", layout="wide"
    )
    st.markdown(apply_snorkel_style(), unsafe_allow_html=True)
    st.markdown(
        '<h1 class="header">Extraction Prompt Development Workflow</h1>',
        unsafe_allow_html=True,
    )
    st.sidebar.title("Extraction Settings")

    # Prompt mode selection in sidebar with tooltip
    prompt_mode_tooltip = """
    Single: Use one prompt for all entities.
    Multi: Create separate prompts for each entity.
    """
    if "prompt_mode" not in st.session_state:
        st.session_state.prompt_mode = "Single"

    st.session_state.prompt_mode = st.sidebar.radio(
        "Prompt Mode",
        ["Single", "Multi"],
        index=["Single", "Multi"].index(st.session_state.prompt_mode),
        help=prompt_mode_tooltip,
    )

    if st.session_state.prompt_mode == "Multi":
        setup_entity_sidebar()


def setup_entity_sidebar() -> None:
    """
    Set up the sidebar for entity input in Multi prompt mode.

    Allows users to input multiple entities for extraction and stores them in the session state.
    """
    # Initialize entities in session state if not present
    if "entities" not in st.session_state:
        st.session_state.entities = []

    # Display text input for entities
    entities_input = st.sidebar.text_input(
        "Enter entities (comma-separated)",
        value=", ".join(st.session_state.entities),
        key="entities_input",
    )

    # Update entities in session state if the input has changed
    new_entities = [
        entity.strip() for entity in entities_input.split(",") if entity.strip()
    ]
    if new_entities != st.session_state.entities:
        st.session_state.entities = new_entities


def load_data() -> pd.DataFrame:
    """
    Load and prepare the dataframe from a selected CSV file.

    Returns:
        pd.DataFrame: The loaded dataframe.

    Raises:
        SystemExit: If no CSV files are found or no file is selected.
    """
    st.subheader("Load Data")

    data_source = st.radio(
        "Choose data source:", ["Select from existing files", "Upload a CSV file"]
    )

    if data_source == "Select from existing files":
        # List CSV files in the data directory
        csv_files = [f for f in os.listdir(EXTRACTION_DATA_DIR) if f.endswith(".csv")]

        if not csv_files:
            st.error("No CSV files found in the extraction_data directory.")
            st.stop()

        selected_file = st.selectbox("Select CSV file:", ["-"] + csv_files)

        if selected_file == "-":
            st.error("No file selected. Please try again.")
            st.stop()

        file_path = os.path.join(EXTRACTION_DATA_DIR, selected_file)
        try:
            df = pd.read_csv(file_path)
            st.session_state.df = df  # Store df in session state for later use
            return df
        except Exception as e:
            logger.error(f"Error loading file {selected_file}: {str(e)}")
            st.error(
                f"Error loading file {selected_file}. Please check the file format."
            )
            st.stop()
    else:  # Upload a CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df  # Store df in session state for later use
            except Exception as e:
                logger.error(f"Error reading uploaded file: {str(e)}")
                st.error(f"Error reading uploaded file: {str(e)}")
                st.stop()
        else:
            st.info("Please upload a CSV file.")
            st.stop()

    return df


def list_saved_prompts() -> List[str]:
    """
    Load the list of saved prompts from the storage directory based on the current prompt mode.

    Returns:
        List[str]: A list of saved prompt filenames, sorted newest first.
    """
    if not os.path.exists(EXTRACTION_PROMPTS_DIR):
        return []

    file_extension = ".json" if st.session_state.prompt_mode == "Multi" else ".txt"
    prompt_files = [
        f for f in os.listdir(EXTRACTION_PROMPTS_DIR) if f.endswith(file_extension)
    ]
    return sorted(prompt_files, reverse=True)


def load_prompt() -> Union[str, Dict[str, str]]:
    """
    Load the baseline prompt from file or use default if not found.

    Returns:
        Union[str, Dict[str, str]]: The loaded prompt(s).
    """
    st.sidebar.markdown("## Load Saved Prompt")
    saved_prompts = list_saved_prompts()
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
    if "modified_prompts" not in st.session_state:
        load_selected_prompt(selected_prompt)

    return st.session_state.modified_prompts


def load_selected_prompt(selected_prompt: str) -> None:
    """
    Load the selected prompt and update the session state.

    Args:
        selected_prompt (str): The name of the selected prompt file.
    """
    if selected_prompt == "Current Baseline":
        loaded_prompt = (
            EXTRACT_PROMPT
            if st.session_state.prompt_mode == "Single"
            else {entity: EXTRACT_PROMPT for entity in st.session_state.entities}
        )
    else:
        try:
            file_path = os.path.join(EXTRACTION_PROMPTS_DIR, selected_prompt)
            if st.session_state.prompt_mode == "Multi":
                with open(file_path, "r") as f:
                    loaded_data = json.load(f)
                st.session_state.entities = loaded_data["entities"]
                loaded_prompt = loaded_data["prompts"]
            else:
                with open(file_path, "r") as f:
                    loaded_prompt = f.read()

        except FileNotFoundError:
            logger.error(f"Selected prompt file {selected_prompt} not found.")
            st.sidebar.error(
                f"Selected prompt file {selected_prompt} not found. Using default prompt."
            )
            loaded_prompt = "Extract the following entities from the text: {entities}\n\nText: {text}\n\nExtracted entities:"

    # Update the session state with the newly loaded prompt
    st.session_state.modified_prompts = loaded_prompt


def display_prompt_dev_box(prompt: Union[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Display the prompt development box for modifying the baseline prompt.

    Args:
        baseline_prompt (Union[str, Dict[str, str]]): The initial prompt(s) to display.

    Returns:
        Dict[str, str]: A dictionary of modified prompts for each entity.
    """
    st.subheader("Prompt Dev Box")
    st.write("Modify the baseline prompt for entity extraction.")

    if st.session_state.prompt_mode == "Single":
        if isinstance(prompt, str):
            return {
                "all": st.text_area("Modified Prompt", value=prompt, height=600) or ""
            }
        else:
            return {
                "all": st.text_area(
                    "Modified Prompt",
                    value=prompt.get("all", EXTRACT_PROMPT),
                    height=600,
                )
                or ""
            }
    else:
        prompts = {}
        tabs = st.tabs(st.session_state.entities)
        for i, tab in enumerate(tabs):
            with tab:
                entity = st.session_state.entities[i]
                entity_prompt = prompt.get(entity, "")
                prompts[entity] = (
                    st.text_area(
                        f"Modified Prompt for {entity}",
                        value=entity_prompt,
                        height=500,
                    )
                    or ""
                )
        return prompts


def parse_gpt4_output_to_dict(output: Optional[str]) -> Dict[str, List[str]]:
    """
    Parse the GPT-4 output from a string that may include markdown JSON formatting.

    Args:
        output (Optional[str]): The raw output from GPT-4.

    Returns:
        Dict[str, List[str]]: A dictionary of parsed entities and their values.
    """
    if not output:
        return {}

    # Extract JSON content from markdown code block if present
    json_match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
    json_str = json_match.group(1) if json_match else output

    try:
        parsed_output = json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Error parsing GPT-4 output: {output}")
        return {}

    # Ensure that values are lists
    parsed_dict = {
        k: v if isinstance(v, list) else [v] for k, v in parsed_output.items()
    }
    return parsed_dict


def process_batch(
    batch: List[Dict[str, Any]],
    modified_prompts: Dict[str, str],
    model: str,
    prompt_mode: str,
) -> List[Dict[str, List[str]]]:
    extractions = []
    for row in batch:
        extraction = {}
        if prompt_mode == "Single":
            formatted_prompt = modified_prompts["all"].format(text=row["text"])
            extraction_response = query_llm(formatted_prompt)
            extraction = parse_gpt4_output_to_dict(extraction_response or "")
        else:
            for entity, prompt in modified_prompts.items():
                formatted_prompt = prompt.format(text=row["text"])
                extraction_response = query_llm(formatted_prompt, model=model)
                entity_extraction = parse_gpt4_output_to_dict(extraction_response or "")
                extraction.update(entity_extraction)
        extractions.append(extraction)
    return extractions


def generate_extractions(
    df: pd.DataFrame, modified_prompts: Dict[str, str], model: str
) -> List[Dict[str, List[str]]]:
    logger.info("Starting extraction generation process")
    batch_size = 1  # Adjust this based on your needs and rate limits
    num_workers = 5  # Adjust based on your CPU cores and rate limits

    prompt_mode = (
        st.session_state.prompt_mode
    )  # Get prompt_mode from session state here

    batches = [
        df[i : i + batch_size].to_dict("records") for i in range(0, len(df), batch_size)
    ]
    logger.info(f"Created {len(batches)} batches of size {batch_size}")

    progress_bar = st.progress(0)
    progress_text = st.empty()

    all_extractions = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        logger.info(f"Starting ThreadPoolExecutor with {num_workers} workers")

        # Use executor.map to process batches
        batch_results = executor.map(
            process_batch,
            batches,
            [modified_prompts] * len(batches),
            [model] * len(batches),
            [prompt_mode] * len(batches),  # Pass prompt_mode to each batch
        )

        for i, batch_extraction in enumerate(batch_results):
            try:
                all_extractions.extend(batch_extraction)

                logger.info(f"Completed batch {i+1}/{len(batches)}")

                # Update progress bar and text
                progress = (i + 1) / len(batches)
                progress_bar.progress(progress)
                progress_text.text(f"Generating extractions: {i + 1}/{len(batches)}")
            except Exception as exc:
                logger.error(f"Batch {i+1} generated an exception: {exc}")

    progress_bar.empty()
    progress_text.empty()

    logger.info("All batches processed")
    logger.info("Extraction generation process completed")
    return all_extractions


def calculate_metrics(
    ground_truth: List[Dict[str, Any]], extractions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate precision, recall, and F1 score for extractions.

    Args:
        ground_truth (List[Dict[str, Any]]): List of ground truth dictionaries.
        extractions (List[Dict[str, Any]]): List of extraction dictionaries.

    Returns:
        Dict[str, Any]: A dictionary containing overall and entity-level metrics.
    """
    entity_metrics = {}
    total_metrics = {"tp": 0, "fp": 0, "fn": 0}

    for i, (gt_item, ext) in enumerate(zip(ground_truth, extractions)):
        try:
            gt = json.loads(gt_item) if isinstance(gt_item, str) else gt_item
            for entity, gt_values in gt.items():
                if entity not in entity_metrics:
                    entity_metrics[entity] = {"tp": 0, "fp": 0, "fn": 0}
                ext_values = ext.get(entity, [])
                tp = len(set(gt_values) & set(ext_values))
                fp = len(set(ext_values) - set(gt_values))
                fn = len(set(gt_values) - set(ext_values))

                entity_metrics[entity]["tp"] += tp
                entity_metrics[entity]["fp"] += fp
                entity_metrics[entity]["fn"] += fn

                total_metrics["tp"] += tp
                total_metrics["fp"] += fp
                total_metrics["fn"] += fn
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON for entry {i}:")
            logger.error(f"String: {gt_item}")
            logger.error(f"Error: {str(e)}")
            continue

    def calculate_prf(metrics):
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
        return {"precision": precision, "recall": recall, "f1": f1}

    overall_metrics = calculate_prf(total_metrics)
    for entity in entity_metrics:
        entity_metrics[entity].update(calculate_prf(entity_metrics[entity]))

    return {"overall": overall_metrics, "entity_metrics": entity_metrics}


def calculate_single_example_metrics(
    ground_truth: Dict[str, List[str]], extraction: Dict[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for a single example.

    Args:
        ground_truth (Dict[str, List[str]]): The ground truth dictionary.
        extraction (Dict[str, List[str]]): The extraction dictionary.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary of metrics for each entity.
    """
    return calculate_metrics([ground_truth], [extraction])["entity_metrics"]


def display_metrics(metrics: Dict[str, Any]) -> None:
    """
    Display the calculated metrics.

    Args:
        metrics (Dict[str, Any]): The metrics dictionary.
    """
    st.subheader("Evaluation Metrics")

    if st.session_state.prompt_mode == "Single":
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Precision", f"{metrics['overall']['precision']:.2f}")
        col2.metric("Average Recall", f"{metrics['overall']['recall']:.2f}")
        col3.metric("Average F1 Score", f"{metrics['overall']['f1']:.2f}")

        with st.expander("Entity-level Metrics"):
            for entity, entity_metrics in metrics["entity_metrics"].items():
                st.markdown(f"#### {entity}")
                subcol1, subcol2, subcol3 = st.columns(3)
                subcol1.metric("Precision", f"{entity_metrics['precision']:.2f}")
                subcol2.metric("Recall", f"{entity_metrics['recall']:.2f}")
                subcol3.metric("F1 Score", f"{entity_metrics['f1']:.2f}")
    else:
        tabs = st.tabs(st.session_state.entities)
        for i, tab in enumerate(tabs):
            with tab:
                entity = st.session_state.entities[i]
                entity_metrics = metrics["entity_metrics"].get(entity, {})
                col1, col2, col3 = st.columns(3)
                col1.metric("Precision", f"{entity_metrics.get('precision', 0):.2f}")
                col2.metric("Recall", f"{entity_metrics.get('recall', 0):.2f}")
                col3.metric("F1 Score", f"{entity_metrics.get('f1', 0):.2f}")


def preview_prompt(
    df: pd.DataFrame, modified_prompts: Dict[str, str], model: str
) -> None:
    """
    Preview the modified prompt by generating extractions and displaying results.

    Args:
        df (pd.DataFrame): The input dataframe.
        modified_prompts (Dict[str, str]): The modified prompts.
        model (str): The name of the model to use.
    """
    if st.button("Preview Prompt"):
        extractions = generate_extractions(df, modified_prompts, model)
        st.session_state.extractions = extractions
        st.session_state.preview_generated = True


def calculate_and_display_metrics(df: pd.DataFrame, extractions: List[Dict]) -> None:
    """
    Calculate and display metrics for the extractions.

    Args:
        df (pd.DataFrame): The input dataframe.
        extractions (List[Dict]): The list of extraction results.
    """
    metrics = calculate_metrics(df["ground_truth"].tolist(), extractions)
    display_metrics(metrics)
    st.session_state.metrics = metrics


def filter_and_display_results(
    df: pd.DataFrame, extractions: List[Dict], show_only_errors: bool
) -> None:
    """
    Filter results based on errors and display them.

    Args:
        df (pd.DataFrame): The input dataframe.
        extractions (List[Dict]): The list of extraction results.
        show_only_errors (bool): Whether to show only responses with errors.
    """
    filtered_results = []
    for i, (_, row) in enumerate(df.iterrows()):
        ground_truth = row["ground_truth"]
        metrics = calculate_single_example_metrics(ground_truth, extractions[i])
        has_error = any(
            entity_metrics["f1"] < 1.0 for entity_metrics in metrics.values()
        )

        if not show_only_errors or has_error:
            filtered_results.append((i, row, extractions[i], metrics))

    st.session_state.filtered_results = filtered_results
    display_preview_results(filtered_results)


def display_preview_results(filtered_results: List[Tuple]) -> None:
    """
    Display the filtered preview results.

    Args:
        filtered_results (List[Tuple]): List of filtered results to display.
    """
    st.write("Extractions generated with the modified prompt:")

    for i, row, extraction, metrics in filtered_results:
        with st.expander(f"Example {i+1}"):
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
            cleaned_text = html.unescape(row["text"])
            st.text(cleaned_text)

            st.write("**Ground Truth:**")
            st.json(row["ground_truth"])

            st.write("**Extracted:**")
            st.json(extraction)


def save_prompt_and_extractions() -> None:
    """Save the latest prompt(s) and generated extractions."""
    if st.button("Save Prompt"):
        if (
            "extractions" in st.session_state
            and "modified_prompts" in st.session_state
            and "df" in st.session_state
        ):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

            # Save the extractions as CSV
            extractions_filename = f"experiment_extractions_{timestamp}.csv"
            extractions_path = os.path.join(
                EXTRACTION_RESULTS_DIR, extractions_filename
            )

            # Save the prompt(s)
            if st.session_state.prompt_mode == "Single":
                prompt_filename = f"extraction_prompt_{timestamp}.txt"
                prompt_path = os.path.join(EXTRACTION_PROMPTS_DIR, prompt_filename)
            else:
                prompt_filename = f"extraction_prompts_{timestamp}.json"
                prompt_path = os.path.join(EXTRACTION_PROMPTS_DIR, prompt_filename)

            # Create a mapping entry
            mapping_entry = {
                "timestamp": timestamp,
                "prompt_file": prompt_filename,
                "extractions_file": extractions_filename,
                "metrics": st.session_state.metrics,
                "prompt_mode": st.session_state.prompt_mode,
            }

            try:
                # Combine original DataFrame with extractions
                combined_df = st.session_state.df.copy()
                combined_df["extractions"] = st.session_state.extractions

                # Save extractions as CSV
                os.makedirs(os.path.dirname(extractions_path), exist_ok=True)
                combined_df.to_csv(extractions_path, index=False)

                # Save prompt(s)
                os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
                if st.session_state.prompt_mode == "Single":
                    with open(prompt_path, "w") as f:
                        f.write(st.session_state.modified_prompts["all"])
                else:
                    with open(prompt_path, "w") as f:
                        json.dump(
                            {
                                "entities": st.session_state.entities,
                                "prompts": st.session_state.modified_prompts,
                            },
                            f,
                            indent=2,
                        )

                # Update the mapping file
                if os.path.exists(PROMPT_MAPPING_FILE):
                    with open(PROMPT_MAPPING_FILE, "r") as f:
                        mapping = json.load(f)
                else:
                    mapping = []

                mapping.append(mapping_entry)

                with open(PROMPT_MAPPING_FILE, "w") as f:
                    json.dump(mapping, f, indent=2)

                st.success("Prompt(s) and extractions saved successfully!")
                logger.info(
                    f"Saved prompt(s) and extractions with timestamp {timestamp}"
                )
            except Exception as e:
                st.error(f"Error saving prompt(s) and extractions: {str(e)}")
                logger.error(f"Error saving prompt(s) and extractions: {str(e)}")
        else:
            st.error("No extractions or data to save. Please preview the prompt first.")


def display_evaluated_extractions(df: pd.DataFrame) -> None:
    """
    Display the extractions in a dataframe and handle row selection.

    Args:
        df (pd.DataFrame): The input dataframe.
    """
    st.subheader("Extraction Data")
    column_config = {
        "filename": st.column_config.TextColumn("Filename", width="medium"),
        "text": st.column_config.TextColumn("Text", width="medium"),
        "ground_truth": st.column_config.TextColumn("Ground Truth", width="large"),
    }

    displayed_columns = ["filename", "text", "ground_truth"]

    if st.session_state.get("selected_row_index", -1) >= 0:
        filtered_df = st.session_state.filtered_df[displayed_columns]
        st.dataframe(
            filtered_df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
        )
    else:
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


def format_metrics_markdown(precision: float, recall: float, f1: float) -> str:
    """
    Format precision, recall, and F1 score as a markdown table.

    Args:
        precision (float): The precision score.
        recall (float): The recall score.
        f1 (float): The F1 score.

    Returns:
        str: A markdown-formatted string representing the metrics table.
    """
    return f"""
| Metric    | Score |
|-----------|-------|
| Precision | {precision:.2f} |
| Recall    | {recall:.2f} |
| F1 Score  | {f1:.2f} |
"""


def iterate_on_specific_datapoint(
    df: pd.DataFrame, prompts: Union[str, Dict[str, str]]
) -> None:
    """
    Handle the iteration on a specific datapoint when a row is selected.

    Args:
        df (pd.DataFrame): The input dataframe.
        baseline_prompt (Union[str, Dict[str, str]]): The baseline prompt(s).
    """
    selected_row = st.session_state.filtered_df.iloc[0]
    st.subheader(f"Iterating on: {selected_row['filename']}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Text**")
        st.text(selected_row["text"])

        st.write("**Ground Truth:**")
        st.json(json.loads(selected_row["ground_truth"]))

    with col2:
        if st.session_state.prompt_mode == "Single":
            iteration_prompt = st.text_area(
                "Modify the prompt for this specific example:",
                value=prompts["all"],
                height=400,
            )
            run_extraction_for_entity(iteration_prompt, selected_row["text"], "all")
        else:
            tabs = st.tabs(st.session_state.entities)
            for i, tab in enumerate(tabs):
                with tab:
                    entity = st.session_state.entities[i]
                    iteration_prompt = st.text_area(
                        f"Modify the prompt for {entity} in this specific example:",
                        value=prompts.get(entity, ""),
                        height=400,
                    )
                    run_extraction_for_entity(
                        iteration_prompt, selected_row["text"], entity
                    )

    if st.button("Back to All Extractions"):
        st.session_state.selected_row_index = -1
        st.rerun()


def run_extraction_for_entity(prompt: str, text: str, entity: str) -> None:
    if st.button(f"Run Extraction for {entity}"):
        extraction_response = query_llm(prompt.format(text=text))
        parsed_extraction = parse_gpt4_output_to_dict(extraction_response or "")

        st.write(f"**Extracted Entities for {entity}:**")
        st.json(parsed_extraction)

        entity_metrics = calculate_single_example_metrics(
            json.loads(st.session_state.filtered_df.iloc[0]["ground_truth"]),
            parsed_extraction,
        )
        st.markdown(f"#### Metrics for {entity}")
        print("entity_metrics", entity_metrics)
        if entity == "all":
            for entity in entity_metrics:
                st.markdown(f"###### Metrics for {entity}")
                st.markdown(
                    format_metrics_markdown(
                        entity_metrics[entity]["precision"],
                        entity_metrics[entity]["recall"],
                        entity_metrics[entity]["f1"],
                    )
                )
        else:
            st.markdown(
                format_metrics_markdown(
                    entity_metrics[entity]["precision"],
                    entity_metrics[entity]["recall"],
                    entity_metrics[entity]["f1"],
                )
            )


def main() -> None:
    """
    Main function to run the Streamlit app for extraction prompt iteration.
    """
    print("Starting Main")
    setup_page()
    logger, logging_level = setup_logging(__name__)
    df = load_data()
    display_evaluated_extractions(df)

    # Load the prompt (managed via session state)
    load_prompt()

    if st.session_state.get("selected_row_index", -1) >= 0:
        iterate_on_specific_datapoint(df, st.session_state.modified_prompts)
    else:
        # Use the modified prompts from session state in the development box
        modified_prompts = display_prompt_dev_box(st.session_state.modified_prompts)
        st.session_state.modified_prompts = modified_prompts
        print("st.session_state.modified_prompts", st.session_state.modified_prompts)

        model = st.selectbox(
            "Select a model", ["gpt-4o", "gpt-4o-2024-08-06", "gpt-4-turbo-preview"]
        )

        preview_prompt(df, modified_prompts, model)

        # Check if preview has been generated
        if st.session_state.get("preview_generated", False):
            # Calculate and display metrics
            calculate_and_display_metrics(df, st.session_state.extractions)

            # Initialize show_only_errors in session state if not present
            if "show_only_errors" not in st.session_state:
                st.session_state.show_only_errors = False

            # Define a callback function for the checkbox
            def on_checkbox_change():
                st.session_state.show_only_errors = (
                    not st.session_state.show_only_errors
                )

            # Use the callback for the checkbox
            st.checkbox(
                "Show only responses with errors",
                value=st.session_state.show_only_errors,
                on_change=on_checkbox_change,
                key="error_checkbox",
            )

            # Filter and display results
            filter_and_display_results(
                df, st.session_state.extractions, st.session_state.show_only_errors
            )

        save_prompt_and_extractions()


if __name__ == "__main__":
    main()
