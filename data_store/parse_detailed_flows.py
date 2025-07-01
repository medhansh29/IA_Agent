import pandas as pd
import json
import os
from openai import OpenAI # Import the OpenAI client
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if it exists

def transform_csv_to_json_openai(csv_file_path):
    """
    Transforms a CSV file into a structured JSON questionnaire using OpenAI's GPT-4o model.

    Args:
        csv_file_path (str): The path to the input CSV file.

    Returns:
        dict: A dictionary containing the transformed data in the specified schema.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()

    # Convert DataFrame to a string representation for the LLM
    # Using to_csv ensures column headers are preserved and gives a clear table format.
    csv_content_string = df.to_csv(index=False)

    # --- Define the desired JSON schema for the LLM's response ---
    # This is the exact schema you provided.
    response_schema = {
        "type": "object", # Use "object" for top-level dictionaries
        "properties": {
            "assessment_variables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "var_name": {"type": "string"},
                        "priority": {"type": "integer"},
                        "description": {"type": "string"},
                        "formula": {"type": ["string", "null"]},
                        "type": {"type": "string"},
                        "value": {"type": ["string", "null"]},
                        "project_id": {"type": ["string", "null"]}
                    },
                    "required": ["id", "name", "var_name", "priority", "description", "type"]
                }
            },
            "computational_variables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "var_name": {"type": "string"},
                        "priority": {"type": "integer"},
                        "description": {"type": "string"},
                        "formula": {"type": ["string", "null"]},
                        "type": {"type": "string"},
                        "value": {"type": ["string", "null"]},
                        "project_id": {"type": ["string", "null"]}
                    },
                    "required": ["id", "name", "var_name", "priority", "description", "type"]
                }
            },
            "questionnaire": {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "order": {"type": "integer"},
                                "is_mandatory": {"type": "boolean"},
                                "rationale": {"type": "string"},
                                "core_questions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "text": {"type": "string"},
                                            "type": {"type": "string"},
                                            "variable_name": {"type": "string"},
                                            "triggering_criteria": {"type": ["string", "null"]},
                                            "assessment_variables": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "formula": {"type": ["string", "null"]},
                                            "is_conditional": {"type": ["boolean", "null"]},
                                            "project_id": {"type": ["string", "null"]}
                                        },
                                        "required": ["id", "text", "type", "variable_name", "assessment_variables"]
                                    }
                                },
                                "conditional_questions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "text": {"type": "string"},
                                            "type": {"type": "string"},
                                            "variable_name": {"type": "string"},
                                            "triggering_criteria": {"type": ["string", "null"]},
                                            "assessment_variables": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "formula": {"type": ["string", "null"]},
                                            "is_conditional": {"type": ["boolean", "null"]},
                                            "project_id": {"type": ["string", "null"]}
                                        },
                                        "required": ["id", "text", "type", "variable_name", "assessment_variables"]
                                    }
                                },
                                "triggering_criteria": {"type": ["string", "null"]}
                            },
                            "required": ["title", "description", "order", "is_mandatory", "rationale", "core_questions", "conditional_questions"]
                        }
                    }
                },
                "required": ["sections"]
            }
        },
        "required": ["assessment_variables", "computational_variables", "questionnaire"]
    }

    # Initialize OpenAI client
    # It's recommended to set your API key as an environment variable (OPENAI_API_KEY)
    # or pass it directly.
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # For demonstration, you might temporarily hardcode it, but this is NOT recommended for production.
    # client = OpenAI(api_key="YOUR_OPENAI_API_KEY_HERE")
    try:
        client = OpenAI() # Assumes OPENAI_API_KEY env variable is set
    except Exception as e:
        print(f"Error initializing OpenAI client. Make sure your API key is set correctly. {e}")
        print("Please set your OpenAI API key as an environment variable named OPENAI_API_KEY.")
        return None


    # Construct the prompt for the LLM. Changed from f-string to a regular string with a placeholder.
    prompt = """
    You are an expert in generating loan questionnaires based on provided data.
    Below is raw questionnaire data in CSV format, representing past assessments for low-income, thin-file individuals applying for loans.
    Your task is to intelligently parse this data and transform it into a structured JSON output, adhering strictly to the provided JSON schema.

    Infer the following information based on the CSV data and the context of loan assessments:
    - **Sections**: Divide the questionnaire into logical sections (e.g., "Basic Information", "Business Details", "Financials", "References"). Each section should have a title, description, order, and rationale.
      - A section should be `is_mandatory: True` by default. If a section primarily contains questions with `Triggering Criteria` or if its title implies it's a follow-up, it might be `is_mandatory: False` and have a `triggering_criteria`.
    - **Assessment Variables**: These are the key pieces of information (variables) that the relationship manager needs to determine from the questions.
      - Generate a unique `id` for each assessment variable (camelCase of the variable name if possible, otherwise a UUID).
      - Infer `name`, `var_name`, `priority`, `description`, and `type` (e.g., "string", "integer", "boolean", "array") based on the "Variable Name" and "Question Type" columns. For `description`, provide a concise summary.
      - `formula` and `value` should be `null` for assessment variables.
      - The `project_id` should be extracted from the 'Project id' column in the CSV for each relevant variable or question.
    - **Computational Variables**: These are variables that the credit team must compute from the assessment variables to determine loan worthiness.
      - Invent plausible `computational_variables` (e.g., "Loan Eligibility Score", "Debt-to-Income Ratio", "Profit Margin").
      - Provide a `formula` for each computational variable, expressed in terms of the `var_name` of the *assessment variables*. If an assessment variable isn't explicitly mentioned in the CSV but is common for such calculations, you can infer it as an assessment variable first.
      - Assign `priority` and a concise `description`.
      - `value` should be `null`.
    - **Questions**:
      - Each question needs a unique `id` (use UUIDs for unique identification within the JSON, as text is not unique).
      - `text` is the actual question.
      - `type` is the input type (e.g., "text", "number", "radio", "select", "textArea").
      - `variable_name` maps to an `assessment_variable`'s `var_name`.
      - `triggering_criteria` and `is_conditional` should be inferred from the CSV's 'Triggering Criteria' column. If 'Triggering Criteria' is present, `is_conditional` is `True`.
      - `assessment_variables` should be an array containing the `var_name` of the assessment variable(s) this question is intended to determine.
      - `formula` should be `null` for questions.
      - `project_id` should be extracted from the 'Project id' column.
      - Place questions into `core_questions` or `conditional_questions` arrays within their respective sections based on `is_conditional`.

    **Important Considerations:**
    - The 'Project id' column in the CSV is crucial; it acts as a unique identifier for each questionnaire (or segment of a questionnaire related to a specific occupation). Ensure variables and questions derived from different 'Project id' values are correctly assigned their respective project IDs.
    - Assume `is_mandatory` is `True` for sections unless there's explicit evidence or a strong inference for a `triggering_criteria`.
    - Be intelligent in inferring relationships and missing data points to create a comprehensive and logical questionnaire.
    - Use camelCase for `id` and `var_name` where derived from existing names. For new auto-generated IDs, use UUIDs.
    - If a 'Variable Name' is mentioned multiple times across different 'Project id's, treat them as distinct for their respective project contexts if their 'Question' or 'Question Type' differs, or as the same underlying assessment variable if consistent. For simplicity, the LLM should primarily map questions to the `assessment_variables` it creates.

    CSV Data:
    {csv_content_string}

    Provide the JSON output strictly following this schema. Ensure all fields are populated according to the schema, inferring where necessary. Do not include any conversational text or explanation outside the JSON.
    """

    from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

    messages = [
        ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant designed to output JSON."),
        ChatCompletionUserMessageParam(role="user", content=prompt.format(csv_content_string=csv_content_string))
    ]

    try:
        print("Sending request to OpenAI LLM (gpt-4o)...")
        # Make the API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",  # You can choose other models like "gpt-3.5-turbo"
            messages=messages,
            # For best results with structured output, consider setting response_format={"type": "json_object"}
            # as well, if your client library version supports it alongside response_model.
            # However, response_model is designed for this purpose directly.
            # response_format={"type": "json_object"},
            max_tokens=4000, # Adjust as needed for potentially large outputs
            temperature=0.7 # Adjust creativity as needed
        )

        # The response_model feature directly gives you a Pydantic object
        # which can be converted to a dictionary.
        return response.model_dump()

    except Exception as e:
        print(f"An error occurred during OpenAI API call: {e}")
        return None

if __name__ == "__main__":
    # Ensure your OpenAI API key is set as an environment variable named OPENAI_API_KEY
    # For example: export OPENAI_API_KEY='your_api_key_here' (on Linux/macOS)
    # or $env:OPENAI_API_KEY='your_api_key_here' (on PowerShell)
    # or set OPENAI_API_KEY=your_api_key_here (on Command Prompt)
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key before running the script.")
    else:
        csv_file = 'filtered_flow_data.csv'
        output_json_file = 'questionnaire_output.json' # Define the output file name

        transformed_json_data = transform_csv_to_json_openai(csv_file)

        if transformed_json_data:
            # Write the JSON output to a file
            try:
                with open(output_json_file, 'w', encoding='utf-8') as f:
                    json.dump(transformed_json_data, f, indent=2, ensure_ascii=False)
                print(f"Successfully wrote transformed JSON data to {output_json_file}")
            except Exception as e:
                print(f"Error writing JSON data to file: {e}")
        else:
            print("Failed to transform CSV data to JSON. No output file was created.")
