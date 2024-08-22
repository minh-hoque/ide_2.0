# Prompt Development Workflow

This project implements a Streamlit-based application for iterative prompt development and evaluation in AI-powered question-answering systems. It allows users to load evaluated responses, modify prompts, preview new responses, and send them for Subject Matter Expert (SME) evaluation.

## Features

- Load and display evaluated responses
- Modify baseline prompts
- Preview new responses using selected AI models
- Auto-evaluate generated responses
- Send responses for SME evaluation

## Prerequisites

- Python 3.7+
- pip (Python package manager)

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
  - `prompt_iteration.py`: Prompt development and iteration page
  - `manual_annotation.py`: Page for manual annotation of responses
  - `auto_evaluation.py`: Page for automatic evaluation of responses
- `css/style.py`: Custom CSS styles
- `helper/`: Helper functions for logging, LLM interactions, etc.
- `prompts/`: Base prompts
- `storage/`: Directory for storing data files

## Running the Application

To run the full application with all pages:

1. Ensure you're in the project root directory and your virtual environment is activated.

2. Run the Streamlit app using the main entry point:
   ```
   streamlit run main.py
   ```

3. Open your web browser and navigate to the URL displayed in the terminal (usually `http://localhost:8501`).

4. Use the sidebar to navigate between different pages:
   - Prompt Iteration
   - Manual Annotation

Each page provides specific functionality for the prompt development workflow.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.