# IDE 2.0 - Prompt Development Workflow

This project implements a Streamlit-based application for iterative prompt development and evaluation in AI-powered question-answering and entity extraction systems. It allows users to load evaluated responses, modify prompts, preview new responses, and send them for Subject Matter Expert (SME) evaluation.

The workflow is designed to scale SME expertise and allow data scientists to quickly iterate and develop prompts for their AI systems.

## Features

- Load and display evaluated responses from CSV files
- Modify baseline prompts with a user-friendly interface
- Preview new responses using selected AI models (GPT-4 variants)
- Auto-evaluate generated responses against original responses or SME feedback
- Calculate and display evaluation metrics (acceptance rate, improvements, regressions)
- Save prompts and responses for further iteration
- Iterate on specific questions for focused improvement
- Entity extraction prompt development and evaluation
- Customizable logging levels for debugging and monitoring
- Stylish UI with custom CSS

## Prerequisites

- Python 3.12.4+
- pip (Python package manager)
- OpenAI API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/prompt-development-workflow.git
   cd prompt-development-workflow
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Project Structure

- `main.py`: Main entry point for the Streamlit application
- `pages/`:
  - `prompt_iteration.py`: Prompt development and iteration page for Q&A
  - `prompt_iteration_extraction.py`: Prompt development and iteration page for entity extraction
  - `manual_annotations.py`: Page for manual annotation of responses
- `css/style.py`: Custom CSS styles for improved UI
- `helper/`:
  - `llms.py`: Functions for interacting with language models
  - `logging.py`: Custom logging setup
- `prompts/`:
  - `base_prompts.py`: Default prompts for the Q&A system
  - `extraction_prompts.py`: Default prompts for the entity extraction system
  - `auto_evaluation_prompts.py`: Prompts for auto-evaluation
- `storage/`:
  - `manual_annotations/`: Directory for storing manually annotated responses
  - `iteration_responses/`: Directory for storing responses from Q&A iteration
  - `extraction_data/`: Directory for storing entity extraction datasets
  - `extraction_results/`: Directory for storing entity extraction results
  - `prompts/`: Directory for storing saved prompts
  - `extraction_prompts/`: Directory for storing saved extraction prompts

## Running the Application

To run the full application with all pages:

1. Ensure you're in the project root directory and your virtual environment is activated.

2. Run the Streamlit app using the main entry point:
   ```
   streamlit run main.py
   ```

3. Open your web browser and navigate to the URL displayed in the terminal (usually `http://localhost:8501`).

4. Use the sidebar to navigate between different pages:
   - **Prompt Iteration (Q&A)**
   - **Extraction Prompt Iteration**
   - **Manual Annotation**

## Usage Guide

### Manual Annotations Page

1. **Upload a CSV File:** Upload a CSV file containing questions and responses following the schema outlined in the [CSV Data Formats](#csv-data-formats) section.
2. **Annotate Responses:** Rate each response as `"ACCEPT"` or `"REJECT"`.
3. **Provide Feedback:** For rejected responses, provide feedback and, if possible, an edited ground truth in the provided fields.
4. **Submit Annotations:** Click the submit button to save your annotations for further processing.

### Prompt Iteration Page (Q&A)

1. **Load Evaluated Responses:** Select an evaluated responses CSV file that follows the required schema from the dropdown menu.
2. **Modify Prompt:** Use the "Prompt Dev Box" to modify the baseline prompt. You can view SME feedback to guide your modifications.
3. **Preview Responses:** Click "Preview Prompt" to generate new responses using the modified prompt. The system will auto-evaluate these responses and display metrics.
4. **Iterate on Specific Questions:** Click on a specific row in the evaluated responses table to focus on improving that particular question-answer pair.
5. **Save Prompt and Responses:** After previewing and iterating, save the new prompt and responses for further evaluation.

### Extraction Prompt Iteration Page

1. **Load Extraction Data:** Select an entity extraction dataset CSV file that adheres to the required schema.
2. **Modify Extraction Prompt:** Use the "Prompt Dev Box" to modify the baseline extraction prompt.
3. **Preview Extractions:** Click "Preview Prompt" to generate new extractions using the modified prompt. The system will evaluate these extractions and display metrics.
4. **View Detailed Results:** Expand individual examples to see detailed extraction results and entity-level metrics.
5. **Save Prompt and Extractions:** After previewing and iterating, save the new prompt and extractions for further evaluation.

## CSV Data Formats

To ensure the application works correctly across all pages, the CSV files you upload or load must adhere to specific column schemas. Below are the required columns for each page:

### Manual Annotations Page

When using the **Manual Annotations** page, your CSV file should contain the following columns:

- **question**: The question or prompt presented to the AI system.
- **response**: The AI's generated response to the question.

**Example CSV format:**

| question                           | response                                           |
|------------------------------------|----------------------------------------------------|
| What is the capital of France?     | The capital of France is Paris.                    |
| Explain the theory of relativity.  | The theory of relativity was developed by Einstein.|

---

### Prompt Iteration Page (Q&A)

For the **Prompt Iteration** page, you should load an evaluated responses CSV file with the following columns:

- **question**: The question presented to the AI system.
- **response**: The AI's previous response to the question.
- **rating**: The evaluation rating from the SME, typically `"ACCEPT"` or `"REJECT"`.
- **sme_feedback** (optional): Feedback provided by the SME for rejected responses.
- **edited_gt** (optional): The edited ground truth for the rejected responses provided by the SME.

**Example CSV format:**

| question                          | response                                     | rating  | sme_feedback           | edited_gt                          |
|-----------------------------------|----------------------------------------------|---------|------------------------|------------------------------------|
| What is the capital of France?    | The capital of France is Marseille.          | REJECT  | Incorrect capital city.| The capital of France is Paris.    |
| Explain the theory of relativity. | Energy equals mass times the speed of light squared.| ACCEPT  |                        |                                    |

---

### Extraction Prompt Iteration Page

For the **Extraction Prompt Iteration** page, your entity extraction dataset CSV should have the following columns:

- **text**: The text from which entities are to be extracted.
- **entities**: A dictionary where the keys are the entity types, and the values are lists of extracted entities of that type.

**Example CSV format:**

| text                                                        | entities                                                                                       |
|-------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| "Barack Obama was the 44th President of the United States." | `{"PERSON": ["Barack Obama"], "LOCATION": ["United States"]}`                                  |
| "Apple released the new iPhone in September."               | `{"ORGANIZATION": ["Apple"], "PRODUCT": ["iPhone"], "DATE": ["September"]}`                    |

**Note:** The `entities` column should be a JSON-formatted string representing a dictionary with entity types as keys and lists of extracted entities as values. Ensure that strings are properly escaped and formatted.

---

### General Guidelines for CSV Files

- **Column Names:** Ensure that the column names match exactly as specified, including case sensitivity. Incorrect column names may cause the application to malfunction or fail to load your data.
- **Data Types:** All columns should contain data in the expected format. For example, the `entities` column should contain JSON-formatted strings representing dictionaries.
- **Encoding:** Save your CSV files with UTF-8 encoding to support special characters and symbols.
- **Delimiters:** Use commas to separate fields and enclose fields that contain commas in double quotes.
- **No Extra Commas or Quotes:** Make sure your CSV files are properly formatted and do not contain extra commas, quotes, or malformed data.
- **Empty Fields:** If some columns are optional (like `sme_feedback` and `edited_gt`), you can include them as empty strings if there's no data to provide.

---

## Key Components

### LLM Interaction (`helper/llms.py`)

- `query_gpt4`: Function to query GPT-4 models with various parameters.
- `query_structured_gpt4`: Function to query GPT-4 and parse responses into structured format.
- `auto_evaluate_responses`: Function to automatically evaluate new responses against old ones or SME feedback.

### Logging (`helper/logging.py`)

- Custom colored logging setup for better debugging and monitoring.

### Prompts (`prompts/auto_evaluation_prompts.py`, `prompts/extraction_prompts.py`)

- Contains prompts for auto-evaluation of responses and entity extraction tasks.

### UI Styling (`css/style.py`)

- Custom CSS styles for a more polished and user-friendly interface.

## Contributing

Contributions to this project are welcome! Please feel free to submit issues, fork the repository, and send pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.