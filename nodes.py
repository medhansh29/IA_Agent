import os
import json
import uuid
import requests
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import time # Import for sleep function

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Import prompts from the new prompts.py file
from prompts import (
    ASSESSMENT_VARIABLES_PROMPT,
    COMPUTATIONAL_VARIABLES_PROMPT,
    QUESTIONNAIRE_PROMPT,
    VARIABLE_MODIFICATIONS_PROMPT,
    QUESTIONNAIRE_MODIFICATIONS_PROMPT,
    JS_REFINEMENT_PROMPT
)

# Import schemas from the new schemas.py file
from schemas.schemas import (
    GraphState,
    VariableSchema,
    AssessmentVariablesOutput,
    ComputationalVariablesOutput,
    Question,
    Section,
    QuestionnaireOutput,
    VariableModificationsOutput,
    QuestionnaireModificationsOutput,
    RemediationOutput,
    StringOutput
    
)
from rag_implementation import get_rag_chain_and_retriever

load_dotenv()  # Load environment variables from .env file

# --- Supabase Configuration (placeholders) ---
SUPABASE_URL = "https://kvzvonrozcmpiflnzcjy.supabase.co/rest/v1"
SUPABASE_API_KEY = os.getenv("SUPABASE_CLIENT_ANON_KEY", "YOUR_SUPABASE_CLIENT_ANON_KEY")


# Initialize the Language Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Initialize RAG components once at module level
try:
    rag_chain, retriever = get_rag_chain_and_retriever()
except Exception as e:
    print(f"Warning: Could not initialize RAG components: {e}")
    rag_chain = None
    retriever = None

# Helper for refining JS expressions with retry mechanism
def _refine_js_expression(llm_instance: ChatOpenAI, expression_type: str, current_expression: Optional[str],
                          context_question_vars: List[str], target_entity_description: str,
                          is_mandatory_flag: bool = True, max_retries: int = 3) -> str:
    """
    Attempts to refine a given JavaScript expression (triggering_criteria or formula).
    Includes a retry mechanism to force the LLM to generate a meaningful expression.
    If, after max_retries, it still cannot, it will return an explicit error string.
    """
    # If it's mandatory triggering criteria, it should be empty. Enforce this immediately.
    if expression_type == "triggering_criteria" and is_mandatory_flag:
        return "" 

    retries = 0
    
    # If current_expression is already a non-trivial, non-failed expression, return it immediately
    if current_expression and current_expression.strip() != "return true;" and "// LLM FAILED" not in current_expression and "// LLM FAILED TO RESPOND" not in current_expression:
        return current_expression.strip()

    while retries < max_retries:
        if retries > 0:
            print(f"Retrying refinement for {expression_type} on {target_entity_description} (Attempt {retries + 1}/{max_retries})...")
            time.sleep(1) # Small delay before retrying

        refinement_chain = JS_REFINEMENT_PROMPT.partial(
            expression_type=expression_type,
            context_question_vars=', '.join(context_question_vars) if context_question_vars else 'None',
            target_entity_description=target_entity_description
        ) | llm_instance.with_structured_output(StringOutput, method='json_mode')

        generated_expression = None
        raw_response = None
        try:
            raw_response = refinement_chain.invoke({})
            print(f"DEBUG: Raw LLM response for {expression_type} '{target_entity_description}': {raw_response}") # DEBUG PRINT
            
            # KEY FIX: Try to extract using expression_type as key first, then fallback to 'expression'
            if isinstance(raw_response, dict):
                generated_expression = raw_response.get(expression_type)
                if generated_expression is None: # Fallback to 'expression' if expression_type key not found
                    generated_expression = raw_response.get("expression")
            elif hasattr(raw_response, "expression"): # If it has an 'expression' attribute
                generated_expression = raw_response.expression

            print(f"DEBUG: Extracted expression for {expression_type} '{target_entity_description}': {generated_expression}") # DEBUG PRINT
        except Exception as e:
            print(f"Error during LLM refinement attempt {retries + 1} for {expression_type} on {target_entity_description}: {e}")
            generated_expression = None # Treat exception as failure for this attempt

        # If LLM generated *anything*, return it immediately for debugging
        if generated_expression is not None:
            return generated_expression.strip() # Return whatever the LLM generated, even if trivial
        
        # If generated_expression is None, LLM truly failed to respond for this attempt, so increment retry count
        retries += 1

    # If loop finishes, it means all retries were exhausted and LLM returned None every time
    print(f"Failed to get any response from LLM for {expression_type} on {target_entity_description} after {max_retries} attempts.")
    return f"// LLM FAILED TO RESPOND: No expression generated after {max_retries} attempts. Review {target_entity_description}."

def _apply_default_variable_properties(var: Dict, is_assessment_var: bool = True, project_id: Optional[str] = None):
    """
    Helper to apply default properties to a variable (assessment or computational).
    Includes project_id.
    Ensures the 'id' field is unique per project by prepending project_id.
    """
    # Ensure the Supabase ID is unique per project and item
    # Use the existing ID if present, otherwise generate a new UUID for the suffix
    base_id = var.get("id") or str(uuid.uuid4())
    if project_id: # Only prepend if project_id is available
        var["id"] = f"{project_id}_{base_id}"
    else:
        var["id"] = base_id

    var["value"] = None # Ensure value is null
    var["project_id"] = project_id # Add project_id

    if 'name' not in var or not var['name']:
        var['name'] = f"Unnamed Variable {var['id']}"
    
    if 'var_name' not in var or not var['var_name']:
        var['var_name'] = var['name'].lower().replace(' ', '_').replace('-', '_').replace('.', '')
    else:
        var['var_name'] = var['var_name'].lower().replace(' ', '_').replace('-', '_').replace('.', '') # Ensure snake_case if provided

    if 'priority' not in var or not isinstance(var['priority'], int):
        var['priority'] = 5 # Default priority

    if 'description' not in var or not var['description']:
        var['description'] = f"{'Assessment' if is_assessment_var else 'Computational'} variable for {var['name']}"
    
    if is_assessment_var:
        var["formula"] = None # Assessment variables have null formula
        if 'type' not in var or not var['type']:
            var['type'] = "text"
    else: # Computational variable
        if 'formula' not in var or not var['formula']:
            var['formula'] = "// Placeholder formula - please define based on assessment variables"
        if 'type' not in var or not var['type']:
            var['type'] = "float" # Computational variables are often numerical

def _process_question_properties(question: Dict, is_core_question: bool, all_existing_q_vars: set, state_error_ref: GraphState, project_id: Optional[str] = None):
    """
    Helper to apply default properties and refine JS expressions for a question.
    all_existing_q_vars: Set of all question variable names in the current questionnaire for context.
    state_error_ref: A mutable dictionary (or the state dict directly) to update with errors.
    Includes project_id and sets is_conditional.
    Ensures the 'id' field is unique per project by prepending project_id.
    """
    # Ensure the Supabase ID is unique per project and item
    # Use the existing ID if present (e.g., 'q1', 'q2'), otherwise generate a new UUID for the suffix
    base_id = question.get("id") or str(uuid.uuid4())
    if project_id: # Only prepend if project_id is available
        question["id"] = f"{project_id}_{base_id}"
    else:
        question["id"] = base_id

    question["project_id"] = project_id # Add project_id
    
    if 'variable_name' not in question or not question['variable_name']:
        question['variable_name'] = f"q_{uuid.uuid4().hex[:8]}"
    else:
        question['variable_name'] = question['variable_name'].lower().replace(' ', '_').replace('-', '_').replace('.', '')

    # Ensure variable_name uniqueness and add to the set
    if question['variable_name'] in all_existing_q_vars and question['variable_name'] != (question.get('original_variable_name_before_update')): # Avoid checking against itself during update
        print(f"Warning: Duplicate variable_name '{question['variable_name']}' detected. Appending unique suffix.")
        question['variable_name'] = f"{question['variable_name']}_{uuid.uuid4().hex[:4]}"
    all_existing_q_vars.add(question['variable_name']) # Add new/unique var_name to context for subsequent refinements

    if 'type' not in question or not question['type']:
        question['type'] = 'text'
    if 'text' not in question or not question['text']:
        question['text'] = "New Question"

    # Refine formula field
    original_formula = question.get('formula')
    # Pass a list conversion of the set for context_question_vars as _refine_js_expression expects List[str]
    refined_formula = _refine_js_expression(
        llm, "formula", original_formula,
        list(all_existing_q_vars), f"formula for '{question.get('text')}'" 
    )
    question['formula'] = refined_formula
    if "// LLM FAILED TO RESPOND" in refined_formula:
        # Ensure state["error"] is always a string before concatenation
        state_error_ref["error"] = (state_error_ref.get("error") or "") + f"ERROR: Formula for question '{question.get('text')}' is problematic. "

    # Refine triggering_criteria if conditional
    if not is_core_question:
        original_criteria = question.get('triggering_criteria')
        # Pass a list conversion of the set for context_question_vars as _refine_js_expression expects List[str]
        refined_criteria = _refine_js_expression(
            llm, "triggering_criteria", original_criteria,
            list(all_existing_q_vars), f"conditional question '{question.get('text')}'", 
            is_mandatory_flag=False
        )
        question['triggering_criteria'] = refined_criteria
        # Set is_conditional based on whether triggering_criteria is present
        question['is_conditional'] = bool(question['triggering_criteria']) # True if not None/empty string, False otherwise
        if "// LLM FAILED TO RESPOND" in refined_criteria:
            # Ensure state["error"] is always a string before concatenation
            state_error_ref["error"] = (state_error_ref.get("error") or "") + f"ERROR: Conditional question '{question.get('text')}' has problematic triggering_criteria. "
    else:
        question['triggering_criteria'] = None # Core questions have null criteria
        question['is_conditional'] = False # Core questions are never conditional
        
    if 'assessment_variables' not in question or not isinstance(question['assessment_variables'], list):
        question['assessment_variables'] = []

def _process_section_properties(section: Dict, all_existing_q_vars: set, state_error_ref: GraphState, project_id: Optional[str] = None):
    """
    Helper to apply default properties and refine JS expressions for a section.
    all_existing_q_vars: Set of all question variable names in the current questionnaire for context.
    state_error_ref: A mutable dictionary (or the state dict directly) to update with errors.
    Includes project_id.
    """
    section["project_id"] = project_id # Add project_id
    if 'is_mandatory' not in section:
        section['is_mandatory'] = True
    if 'rationale' not in section:
        section['rationale'] = "Generated rationale."
    if 'core_questions' not in section:
        section['core_questions'] = []
    if 'conditional_questions' not in section:
        section['conditional_questions'] = []

    # Refine section triggering_criteria if not mandatory
    if not section['is_mandatory']:
        original_criteria = section.get('triggering_criteria')
        # Pass a list conversion of the set for context_question_vars as _refine_js_expression expects List[str]
        refined_criteria = _refine_js_expression(
            llm, "triggering_criteria", original_criteria,
            list(all_existing_q_vars), f"section '{section.get('title')}'", 
            is_mandatory_flag=False
        )
        section['triggering_criteria'] = refined_criteria
        if "// LLM FAILED TO RESPOND" in refined_criteria:
            # Ensure state["error"] is always a string before concatenation
            state_error_ref["error"] = (state_error_ref.get("error") or "") + f"ERROR: Section '{section.get('title')}' has problematic triggering_criteria. "
    else:
        section['triggering_criteria'] = None # Mandatory sections should have null criteria
    
    if 'data_validation' not in section:
        section['data_validation'] = "return true;" # Default validation


# --- Langraph Node 1: generate_variables ---
def generate_variables(state: GraphState) -> GraphState:
    """
    Generates initial assessment and computational variables based on the user's prompt.
    It prompts an LLM twice: first for assessment variables, then for computational
    variables based on the suggested assessment variables.
    """
    print("---GENERATING INITIAL VARIABLES---")
    # Ensure state["error"] is a string at the start of this node
    state["error"] = state.get("error", "")

    prompt_text = state["prompt"]
    project_id = state.get("project_id") # Get project_id from state
    current_assessment_variables = state.get("assessment_variables", [])
    current_computational_variables = state.get("computational_variables", [])

    # Get RAG context
    context_docs = []
    if retriever:
        try:
            print(f"Retrieving RAG context for prompt: {prompt_text}")
            context_docs = retriever.invoke(prompt_text)
            print(f"Retrieved {len(context_docs)} context documents.")
        except Exception as e:
            print(f"Warning: Could not retrieve RAG context: {e}")
            context_docs = []

    # Step 1: Identify Assessment Variables using LLM
    if not current_assessment_variables:
        print("Generating Assessment Variables...")
        assessment_chain = ASSESSMENT_VARIABLES_PROMPT | llm.with_structured_output(AssessmentVariablesOutput, method='function_calling')

        try:
            llm_response = assessment_chain.invoke({
                "user_input": prompt_text,
                "existing_variables": json.dumps(current_assessment_variables),
                "context": context_docs # Pass RAG context
            })
            suggested_assessment_vars = llm_response.get("assessment_variables", [])

            for var in suggested_assessment_vars:
                _apply_default_variable_properties(var, is_assessment_var=True, project_id=project_id) # Pass project_id

            state["assessment_variables"] = suggested_assessment_vars
            print("\n---Initial Suggested Assessment Variables:---")
            print(json.dumps(suggested_assessment_vars, indent=2))

        except Exception as e:
            state["error"] = (state.get("error") or "") + f"Error generating assessment variables: {e}" # Concatenate
            print(f"Error generating assessment variables: {e}")
            return state # Critical failure, stop workflow

    # Step 2: Create Computational Variables using LLM
    if state["assessment_variables"] and not current_computational_variables:
        print("\nGenerating Computational Variables...")
        assessment_var_names = ", ".join([v["var_name"] for v in state["assessment_variables"]])

        # Use RAG context for computational variables based on assessment var names
        comp_context_docs = []
        if retriever:
            try:
                print(f"Retrieving RAG context for computational variables based on: {assessment_var_names}")
                comp_context_docs = retriever.invoke(assessment_var_names)
                print(f"Retrieved {len(comp_context_docs)} context documents for computational variables.")
            except Exception as e:
                print(f"Warning: Could not retrieve RAG context for computational variables: {e}")
                comp_context_docs = []

        computational_chain = COMPUTATIONAL_VARIABLES_PROMPT | llm.with_structured_output(ComputationalVariablesOutput, method='function_calling')

        try:
            llm_response = computational_chain.invoke({
                "assessment_variables": json.dumps([{"var_name": v["var_name"], "name": v["name"], "type": v["type"]} for v in state["assessment_variables"]]),
                "existing_computational_variables": json.dumps(current_computational_variables),
                "user_input": prompt_text,
                "context": comp_context_docs # Pass RAG context
            })
            suggested_computational_vars = llm_response.get("computational_variables", [])

            for var in suggested_computational_vars:
                _apply_default_variable_properties(var, is_assessment_var=False, project_id=project_id) # Pass project_id

            state["computational_variables"] = suggested_computational_vars
            print("\n---Initial Suggested Computational Variables:---")
            print(json.dumps(suggested_computational_vars, indent=2))

        except Exception as e:
            state["error"] = (state.get("error") or "") + f"Error generating computational variables: {e}" # Concatenate
            print(f"Error generating computational variables: {e}")
            return state # Critical failure, stop workflow

    return state

def modify_variables_llm(state: GraphState) -> GraphState:
    """
    Allows an LLM to modify assessment and computational variables based on a modification prompt.
    The LLM can add, update, or remove variables.
    """
    print("\n---MODIFYING VARIABLES USING LLM---")
    # Ensure state["error"] is a string at the start of this node
    state["error"] = state.get("error", "")

    modification_prompt = state.get("modification_prompt")
    current_assessment_variables = state.get("assessment_variables")
    if current_assessment_variables is None:
        current_assessment_variables = []
    
    current_computational_variables = state.get("computational_variables")
    if current_computational_variables is None:
        current_computational_variables = []

    if not modification_prompt:
        print("No modification prompt provided. Skipping LLM variable modification.")
        return state

    print(f"Applying variable modifications based on: '{modification_prompt}'")
    modification_chain = VARIABLE_MODIFICATIONS_PROMPT | llm.with_structured_output(VariableModificationsOutput, method='function_calling')

    try:
        llm_response = modification_chain.invoke({
            "current_assessment_variables": json.dumps(current_assessment_variables, indent=2),
            "current_computational_variables": json.dumps(current_computational_variables, indent=2),
            "modification_request": modification_prompt
        })
        modifications = llm_response # This is already the VariableModificationsOutput typed dict

        modified_assessment_variables = list(current_assessment_variables)
        modified_computational_variables = list(current_computational_variables)
        project_id = state.get("project_id")

        # Apply Assessment Variable Modifications
        print("\n---Applying Assessment Variable Modifications---")
        # Removals
        if modifications.get("removed_assessment_variable_ids"):
            initial_count = len(modified_assessment_variables)
            # Filter based on the modified (project_id-prefixed) IDs
            removed_base_ids = {f_id.split('_', 1)[-1] for f_id in modifications["removed_assessment_variable_ids"]}
            modified_assessment_variables = [
                var for var in modified_assessment_variables
                if var["id"].split('_', 1)[-1] not in removed_base_ids # Compare only the base ID
            ]
            print(f"Removed {initial_count - len(modified_assessment_variables)} assessment variables.")
        
        # Updates
        if modifications.get("updated_assessment_variables"):
            for update_data in modifications["updated_assessment_variables"]:
                base_id_to_update = update_data["id"] # This ID is expected to be the base ID (e.g., 'av1', not 'project_id_av1')
                found = False
                for i, var in enumerate(modified_assessment_variables):
                    # Compare base ID of existing var with the ID to update
                    if var["id"].split('_', 1)[-1] == base_id_to_update:
                        for key, value in update_data.items():
                            if key != "id": # Don't update the ID itself here, it's used for matching
                                var[key] = value
                        # Re-apply defaults just in case, particularly for var_name consistency
                        _apply_default_variable_properties(var, is_assessment_var=True, project_id=project_id)
                        modified_assessment_variables[i] = var
                        print(f"Updated assessment variable: {base_id_to_update} (name: {var.get('name')})")
                        found = True
                        break
                if not found:
                    print(f"Warning: Assessment variable with base ID '{base_id_to_update}' to update not found.")

        # Additions
        if modifications.get("added_assessment_variables"):
            for new_var_data in modifications["added_assessment_variables"]:
                # Ensure new variable IDs are unique and apply default properties with project_id
                # Check for existing base_id to avoid duplicates if LLM re-generates something
                new_base_id = new_var_data.get("id")
                is_duplicate = False
                if new_base_id:
                    # Check against existing *base* IDs
                    for existing_var in modified_assessment_variables:
                        if existing_var["id"].split('_', 1)[-1] == new_base_id:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    _apply_default_variable_properties(new_var_data, is_assessment_var=True, project_id=project_id)
                    modified_assessment_variables.append(new_var_data)
                    print(f"Added new assessment variable: {new_var_data.get('name')}")
                else:
                    print(f"Warning: Attempted to add duplicate assessment variable with base ID '{new_base_id}'. Skipping.")

        state["assessment_variables"] = modified_assessment_variables

        # Apply Computational Variable Modifications
        print("\n---Applying Computational Variable Modifications---")
        # Removals
        if modifications.get("removed_computational_variable_ids"):
            initial_count = len(modified_computational_variables)
            removed_base_ids = {f_id.split('_', 1)[-1] for f_id in modifications["removed_computational_variable_ids"]}
            modified_computational_variables = [
                var for var in modified_computational_variables
                if var["id"].split('_', 1)[-1] not in removed_base_ids # Compare only the base ID
            ]
            print(f"Removed {initial_count - len(modified_computational_variables)} computational variables.")
        
        # Updates
        if modifications.get("updated_computational_variables"):
            for update_data in modifications["updated_computational_variables"]:
                base_id_to_update = update_data["id"]
                found = False
                for i, var in enumerate(modified_computational_variables):
                    if var["id"].split('_', 1)[-1] == base_id_to_update:
                        for key, value in update_data.items():
                            if key != "id":
                                var[key] = value
                        _apply_default_variable_properties(var, is_assessment_var=False, project_id=project_id)
                        modified_computational_variables[i] = var
                        print(f"Updated computational variable: {base_id_to_update} (name: {var.get('name')})")
                        found = True
                        break
                if not found:
                    print(f"Warning: Computational variable with base ID '{base_id_to_update}' to update not found.")

        # Additions
        if modifications.get("added_computational_variables"):
            for new_var_data in modifications["added_computational_variables"]:
                new_base_id = new_var_data.get("id")
                is_duplicate = False
                if new_base_id:
                    for existing_var in modified_computational_variables:
                        if existing_var["id"].split('_', 1)[-1] == new_base_id:
                            is_duplicate = True
                            break
                if not is_duplicate:
                    _apply_default_variable_properties(new_var_data, is_assessment_var=False, project_id=project_id)
                    modified_computational_variables.append(new_var_data)
                    print(f"Added new computational variable: {new_var_data.get('name')}")
                else:
                    print(f"Warning: Attempted to add duplicate computational variable with base ID '{new_base_id}'. Skipping.")

        state["computational_variables"] = modified_computational_variables

        # Crucially, after modifications, re-evaluate computational variable formulas
        # This will ensure that if an AV was removed, CVs dependent on it are flagged or updated.
        print("\n---Re-evaluating Computational Variable Formulas based on latest AVs---")
        latest_assessment_var_names = {var["var_name"] for var in state["assessment_variables"]}

        for cv in state["computational_variables"]:
            # Only attempt to refine if there's an existing formula to work with
            if cv.get("formula"):
                refined_formula = _refine_js_expression(
                    llm_instance=llm,
                    expression_type="formula",
                    current_expression=cv["formula"],
                    context_question_vars=list(latest_assessment_var_names), # Provide latest AVs as context
                    target_entity_description=f"computational variable '{cv.get('name')}'"
                )
                if "// LLM FAILED" in refined_formula or "// LLM FAILED TO RESPOND" in refined_formula:
                    state["error"] = (state.get("error") or "") + f"ERROR: Computational variable '{cv.get('name')}' has a problematic formula after AV modifications. "
                cv["formula"] = refined_formula
            else:
                 # If no formula, generate a placeholder based on new AVs if relevant
                print(f"Generating initial formula for computational variable '{cv.get('name')}'...")
                generated_formula = _refine_js_expression(
                    llm_instance=llm,
                    expression_type="formula",
                    current_expression=None, # Indicate no existing formula
                    context_question_vars=list(latest_assessment_var_names),
                    target_entity_description=f"computational variable '{cv.get('name')}'"
                )
                if "// LLM FAILED" in generated_formula or "// LLM FAILED TO RESPOND" in generated_formula:
                    state["error"] = (state.get("error") or "") + f"ERROR: Failed to generate initial formula for computational variable '{cv.get('name')}'. "
                cv["formula"] = generated_formula

        state["error"] = None # Clear previous error if successful here

    except Exception as e:
        state["error"] = (state.get("error") or "") + f"ERROR: LLM failed to generate variable modifications: {e}. "
        print(f"Error during LLM variable modification: {e}")

    return state

# --- Langraph Node 3: generate_questionnaire ---
def generate_questionnaire(state: GraphState) -> GraphState:
    """
    Generates a survey questionnaire based on the identified assessment variables.
    Questions are grouped into sections with core and conditional questions.
    """
    print("\n---GENERATING QUESTIONNAIRE---")
    # Ensure state["error"] is a string at the start of this node
    state["error"] = state.get("error", "")
    assessment_variables = state.get("assessment_variables", [])
    computational_variables = state.get("computational_variables")
    if computational_variables is None:
        computational_variables = []
    prompt_context = state.get("prompt", "")
    project_id = state.get("project_id") # Get project_id from state

    if not assessment_variables:
        state["error"] = (state.get("error") or "") + "No assessment variables found to generate questionnaire." # Concatenate
        print(state["error"])
        return state

    # Extract relevant info from assessment variables for the LLM
    # Provide var_name, name, description, and type to help LLM craft questions
    assessment_vars_info = [
        {
            "var_name": var["var_name"],
            "name": var["name"],
            "description": var["description"],
            "type": var["type"]
        } for var in assessment_variables
    ]
    assessment_vars_json = json.dumps(assessment_vars_info, indent=2)

    computational_vars_info = [
        {
            "var_name": var["var_name"],
            "name": var["name"],
            "description": var["description"],
            "type": var["type"]
        } for var in computational_variables
    ]
    computational_vars_json = json.dumps(computational_vars_info, indent=2)

    # Get RAG context for questionnaire generation
    context_docs = []
    if retriever:
        try:
            print(f"Retrieving RAG context for questionnaire generation based on: {prompt_context}")
            context_docs = retriever.invoke(prompt_context)
            print(f"Retrieved {len(context_docs)} context documents for questionnaire.")
        except Exception as e:
            print(f"Warning: Could not retrieve RAG context for questionnaire: {e}")
            context_docs = []

    questionnaire_chain = QUESTIONNAIRE_PROMPT | llm.with_structured_output(QuestionnaireOutput, method='function_calling')

    try:
        llm_response = questionnaire_chain.invoke({
            "user_input": prompt_context,
            "assessment_variables": assessment_vars_json,
            "computational_variables": computational_vars_json,
            "context": context_docs # Pass RAG context
        })
        generated_questionnaire = llm_response

        # Collect initial q_vars from generated_questionnaire to populate all_existing_q_vars_set
        # Use a set to ensure uniqueness from the start
        all_existing_q_vars_set = set()
        for section in generated_questionnaire.get("sections", []):
            for q_list in [section.get('core_questions', []), section.get('conditional_questions', [])]:
                for question in q_list:
                    if question.get('variable_name'):
                        all_existing_q_vars_set.add(question['variable_name'])

        # Post-processing: Ensure IDs, default values, and intelligent JS for sec_idx, section in enumerate(generated_questionnaire.get("sections", [])):
        for sec_idx, section in enumerate(generated_questionnaire.get("sections", [])):
            section['order'] = section.get('order', sec_idx + 1) # Ensure order
            # Pass the main state dict directly for error tracking
            _process_section_properties(section, all_existing_q_vars_set, state, project_id=project_id) # Pass project_id

            for q_list, is_core_q_flag in [
                (section['core_questions'], True),
                (section['conditional_questions'], False)
            ]:
                for q_idx, question in enumerate(q_list):
                    # Pass the main state dict directly for error tracking
                    _process_question_properties(question, is_core_q_flag, all_existing_q_vars_set, state, project_id=project_id) # Pass project_id

        state["questionnaire"] = generated_questionnaire
        print("\n---Generated Questionnaire:---")
        print(json.dumps(generated_questionnaire, indent=2))

    except Exception as e:
        state["error"] = (state.get("error") or "") + f"Error generating questionnaire: {e}" # Concatenate
        print(f"Error generating questionnaire: {e}")

    return state



# --- Langraph Node 4: modify_questionnaire_llm ---
def modify_questionnaire_llm(state: GraphState) -> GraphState:
    """
    Allows an LLM to modify the questionnaire based on a modification prompt.
    The LLM can add, update, or remove sections and questions.
    """
    print("\n---MODIFYING QUESTIONNAIRE USING LLM---")
    # Ensure state["error"] is a string at the start of this node
    state["error"] = state.get("error", "")

    modification_prompt = state.get("modification_prompt")
    current_questionnaire = state.get("questionnaire")
    project_id = state.get("project_id") # Get project_id from state

    if not modification_prompt or not current_questionnaire:
        print("No modification prompt or questionnaire found. Skipping LLM questionnaire modification.")
        return state

    print(f"Applying questionnaire modifications based on: '{modification_prompt}'")

    # Pass assessment variables to LLM for context during questionnaire modification
    assessment_vars_info = [
        {
            "var_name": var["var_name"],
            "name": var["name"],
            "description": var["description"],
            "type": var["type"]
        }
        for var in (state.get("assessment_variables") or [])
    ]
    assessment_vars_json = json.dumps(assessment_vars_info, indent=2)


    modification_chain = QUESTIONNAIRE_MODIFICATIONS_PROMPT | llm.with_structured_output(QuestionnaireModificationsOutput, method='function_calling')

    try:
        llm_response = modification_chain.invoke({
            "questionnaire_json": json.dumps(current_questionnaire, indent=2),
            "modification_prompt": modification_prompt,
            "assessment_vars_json": assessment_vars_json # Pass context
        })

        modifications = llm_response
        modified_questionnaire = current_questionnaire.copy()
        modified_sections = [dict(sec) for sec in modified_questionnaire.get("sections", [])]
        
        # Ensure assessment_variable_calculation exists and is a dict
        if "assessment_variable_calculation" not in modified_questionnaire or modified_questionnaire["assessment_variable_calculation"] is None:
            modified_questionnaire["assessment_variable_calculation"] = {}
        
        modified_av_calc = modified_questionnaire["assessment_variable_calculation"].copy()


        # Handle Section Removals (Process first to simplify updates/additions)
        if modifications.get("removed_section_orders"):
            initial_section_count = len(modified_sections)
            removed_orders = set(modifications["removed_section_orders"])
            modified_sections = [sec for sec in modified_sections if sec['order'] not in removed_orders]
            print(f"Removed {initial_section_count - len(modified_sections)} sections.")
            # Reorder remaining sections to maintain contiguous order
            modified_sections.sort(key=lambda x: x['order'])
            for i, section in enumerate(modified_sections):
                section['order'] = i + 1

        # Helper to get all question variable names in the current (modified) questionnaire structure
        def all_existing_q_vars_in_questionnaire(sections_list) -> set:
            q_vars = set()
            for sec in sections_list:
                for q_type_list in [sec.get('core_questions', []), sec.get('conditional_questions', [])]:
                    for q in q_type_list:
                        if q.get('variable_name'):
                            q_vars.add(q['variable_name'])
            return q_vars

        # Handle Section Updates
        if modifications.get("updated_sections"):
            for updated_sec_data in modifications["updated_sections"]:
                sec_order = updated_sec_data.get("order")
                if sec_order:
                    for i, section in enumerate(modified_sections):
                        if section.get("order") == sec_order:
                            section.update(updated_sec_data)
                            # Pass the main state dict directly for error tracking
                            _process_section_properties(section, all_existing_q_vars_in_questionnaire(modified_sections), state, project_id=project_id) # Pass project_id
                            print(f"Updated section order {sec_order}: {section.get('title')}")
                            break
                    else:
                        print(f"Warning: Section with order '{sec_order}' to update not found.")

        # Handle Section Additions (assign unique orders)
        if modifications.get("added_sections"):
            for new_sec_data in modifications["added_sections"]:
                new_order = new_sec_data.get("order")
                if new_order is None or any(s['order'] == new_order for s in modified_sections):
                    new_order = max([s['order'] for s in modified_sections]) + 1 if modified_sections else 1
                new_sec_data['order'] = new_order
                # Pass the main state dict directly for error tracking
                _process_section_properties(new_sec_data, all_existing_q_vars_in_questionnaire(modified_sections), state, project_id=project_id) # Pass project_id
                modified_sections.append(new_sec_data)
                print(f"Added new section: {new_sec_data.get('title', f'Order {new_order}')}")
            modified_sections.sort(key=lambda x: x['order']) # Re-sort after additions

        # Consolidate all questions for easier lookup during removal/update/add
        all_questions_map = {} # {id (prefixed): {question_obj, section_ref, is_core}}
        for section in modified_sections:
            for q_type, q_list in [('core', section['core_questions']), ('conditional', section['conditional_questions'])]:
                for question in q_list:
                    # Store the project-prefixed ID in the map
                    all_questions_map[question['id']] = { 
                        'obj': question,
                        'section': section,
                        'is_core': (q_type == 'core')
                    }
        # Also need a map for variable_name to id (prefixed) for removals
        question_varname_to_prefixed_id_map = {q_info['obj']['variable_name']: q_id for q_id, q_info in all_questions_map.items()}


        # Handle Question Removals by variable_name
        if modifications.get("removed_question_variable_names"):
            for var_name_to_remove in modifications["removed_question_variable_names"]:
                # Get the prefixed ID from the var_name
                prefixed_q_id_to_remove = question_varname_to_prefixed_id_map.get(var_name_to_remove)
                if prefixed_q_id_to_remove and prefixed_q_id_to_remove in all_questions_map:
                    q_info = all_questions_map[prefixed_q_id_to_remove]
                    containing_section = q_info['section']
                    if q_info['is_core']:
                        containing_section['core_questions'] = [q for q in containing_section['core_questions'] if q['id'] != prefixed_q_id_to_remove]
                    else:
                        containing_section['conditional_questions'] = [q for q in containing_section['conditional_questions'] if q['id'] != prefixed_q_id_to_remove]
                    print(f"Removed question with variable_name: {var_name_to_remove} (ID: {prefixed_q_id_to_remove})")
                    del all_questions_map[prefixed_q_id_to_remove] # Remove from map too
                    del question_varname_to_prefixed_id_map[var_name_to_remove] # Remove from varname map too
                else:
                    print(f"Warning: Question with variable_name '{var_name_to_remove}' to remove not found.")

        # Handle Question Updates by id (LLM provides the base ID, we need to find the prefixed one)
        if modifications.get("updated_questions"):
            for updated_q_data in modifications["updated_questions"]:
                base_q_id = updated_q_data.get("id") # LLM gives the base ID
                if base_q_id:
                    found = False
                    for prefixed_q_id, q_info in all_questions_map.items():
                        # Check if the stored prefixed_q_id ends with the base_q_id from LLM
                        if prefixed_q_id.endswith(f"_{base_q_id}"):
                            q_obj = q_info['obj']
                            
                            # Store original var_name to prevent self-collision during uniqueness check
                            q_obj['original_variable_name_before_update'] = q_obj.get('variable_name')

                            # Update all fields except 'id' as 'id' is already project-prefixed
                            for key, value in updated_q_data.items():
                                if key != "id":
                                    q_obj[key] = value

                            q_obj['value'] = None # Ensure value stays null after update
                            # Re-apply properties to ensure formula/criteria refinement and project_id prefix if needed
                            _process_question_properties(q_obj, q_info['is_core'], all_existing_q_vars_in_questionnaire(modified_sections), state, project_id=project_id)
                            
                            # Clean up the temporary key
                            if 'original_variable_name_before_update' in q_obj:
                                del q_obj['original_variable_name_before_update']

                            # Update varname_to_id map if var_name changed for this question
                            if updated_q_data.get('variable_name'):
                                question_varname_to_prefixed_id_map[q_obj['variable_name']] = q_obj['id'] # Use the new prefixed ID

                            print(f"Updated question: {base_q_id} (text: {q_obj.get('text')})")
                            found = True
                            break
                    if not found:
                        print(f"Warning: Question with base ID '{base_q_id}' to update not found.")

        # Handle Question Additions
        if modifications.get("added_questions"):
            # Update all_existing_q_vars_in_questionnaire dynamically as new questions are added
            existing_q_vars_for_additions = all_existing_q_vars_in_questionnaire(modified_sections)

            for add_q_data in modifications["added_questions"]:
                section_order = add_q_data.get("section_order")
                is_core = add_q_data.get("is_core", True)
                question_data = add_q_data.get("question")

                if section_order and question_data:
                    target_section = next((s for s in modified_sections if s['order'] == section_order), None)
                    if target_section:
                        # Apply properties to the new question data (this will generate its project-prefixed ID)
                        _process_question_properties(question_data, is_core, existing_q_vars_for_additions, state, project_id=project_id) 
                        
                        # Check for duplicate *prefixed* IDs before appending
                        if question_data['id'] in all_questions_map:
                            print(f"Warning: Generated question ID '{question_data['id']}' already exists. Re-generating ID for added question.")
                            # Re-generate a completely new ID if collision occurs
                            question_data['id'] = f"{project_id}_{str(uuid.uuid4())}"
                            _process_question_properties(question_data, is_core, existing_q_vars_for_additions, state, project_id=project_id) # Re-process

                        if is_core:
                            target_section['core_questions'].append(question_data)
                        else:
                            target_section['conditional_questions'].append(question_data)
                        all_questions_map[question_data['id']] = { # Add to main ID map with prefixed ID
                            'obj': question_data,
                            'section': target_section,
                            'is_core': is_core
                        }
                        question_varname_to_prefixed_id_map[question_data['variable_name']] = question_data['id'] # Add to varname map with prefixed ID
                        print(f"Added new question '{question_data.get('text', question_data['variable_name'])}' (ID: {question_data['id']}) to section {section_order}.")
                    else:
                        print(f"Warning: Section with order '{section_order}' not found for adding question.")
                else:
                    print("Warning: Missing 'section_order' or 'question' data for question addition.")

        modified_questionnaire["sections"] = modified_sections
        state["questionnaire"] = modified_questionnaire
        print("\n---LLM Questionnaire Modification Complete.---")

    except Exception as e:
        state["error"] = (state.get("error") or "") + f"Error modifying questionnaire with LLM: {e}" # Concatenate
        print(f"Error modifying questionnaire with LLM: {e}")
        # Do NOT return state here, allow processing to continue to next node
        return state

    return state


def _upsert_single_item(table_name: str, item: Dict[str, Any], headers: Dict[str, str]) -> bool:
    """
    Alternative upsert method using Supabase's native upsert functionality.
    This is more efficient as it does the upsert in a single operation.
    """
    item_id = item.get("id")
    if not item_id:
        print(f"❌ Error: Item for table '{table_name}' missing 'id'. Cannot upsert.")
        return False

    # FIX: Correct the URL construction by removing the redundant "/rest/v1"
    url = f"{SUPABASE_URL}/{table_name}"
    
    # Use Supabase's upsert functionality
    upsert_headers = headers.copy()
    upsert_headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    
    print(f"\n🔄 Advanced upsert for '{table_name}' with ID '{item_id}'")
    print(f"📡 URL: {url}")
    print(f"📝 Payload: {json.dumps(item, indent=2)}")

    try:
        response = requests.post(url, headers=upsert_headers, json=item)
        
        print(f"📊 Status: {response.status_code}")
        print(f"📋 Response: {response.text}")
        
        if response.status_code in [200, 201]:
            print(f"✅ Successfully upserted record '{item_id}' in '{table_name}'")
            return True
        else:
            print(f"❌ Upsert failed with status {response.status_code}")
            print(f"📋 Error details: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during advanced upsert for '{item_id}' in '{table_name}': {e}")
        return False


# --- Langraph Node 5: analyze_questionnaire_impact ---
def analyze_questionnaire_impact(state: GraphState) -> GraphState:
    """
    Analyzes the questionnaire to ensure all assessment variables can be calculated.
    If variables are uncalculable, it attempts to generate new questions to cover them.
    """
    print("\n---ANALYZING QUESTIONNAIRE IMPACT---")
    # Ensure state["error"] is a string at the start of this node
    state["error"] = state.get("error", "")

    # Ensure assessment_variables is always a list at the start of this node
    current_assessment_variables = state.get("assessment_variables")
    if current_assessment_variables is None:
        current_assessment_variables = []
        state["assessment_variables"] = current_assessment_variables # Update state if it was None

    assessment_variables = current_assessment_variables # Use the guaranteed list from now on
    questionnaire = state.get("questionnaire") or {}
    sections = questionnaire.get("sections", [])
    
    # Initialize assessment_variable_calculation if it's None or missing
    if "assessment_variable_calculation" not in questionnaire or questionnaire["assessment_variable_calculation"] is None:
        questionnaire["assessment_variable_calculation"] = {}
    av_calculation_map = questionnaire["assessment_variable_calculation"]


    if not assessment_variables:
        print("No assessment variables to analyze impact.")
        return state
    if not sections:
        print("No questionnaire sections to analyze impact.")
        return state

    # Create a quick lookup for assessment variable objects by var_name
    av_varname_map = {av['var_name']: av for av in assessment_variables}

    # Track which assessment variables are explicitly covered by assessment_variables list in questions
    explicitly_covered_avs = set()
    all_question_vars = set()
    # New set to track AVs referenced by questions but not existing in state['assessment_variables']
    referenced_but_missing_avs = set() 

    for section in sections:
        for q_list in [section.get('core_questions', []), section.get('conditional_questions', [])]:
            for question in q_list:
                all_question_vars.add(question.get('variable_name'))
                for av_name in question.get('assessment_variables', []):
                    explicitly_covered_avs.add(av_name)
                    # Check if this referenced AV exists in our master list
                    if av_name not in av_varname_map:
                        referenced_but_missing_avs.add(av_name)

    # All assessment variables that are supposed to exist based on `state['assessment_variables']`
    existing_assessment_var_names_in_state = {av['var_name'] for av in assessment_variables}

    # Combine uncovered AVs that are *expected* to exist but aren't explicitly covered by questions
    uncovered_assessment_vars_by_question_impact = [
        var for var in assessment_variables
        if var["var_name"] not in explicitly_covered_avs
    ]

    # Add placeholder AVs for those referenced by questions but not existing in state
    if referenced_but_missing_avs:
        print(f"Detected questions referencing missing assessment variables: {', '.join(referenced_but_missing_avs)}")
        for missing_av_name in referenced_but_missing_avs:
            if missing_av_name not in existing_assessment_var_names_in_state: # Only add if truly missing
                print(f"Adding placeholder assessment variable for: {missing_av_name}")
                placeholder_av = {
                    "id": str(uuid.uuid4()),
                    "name": missing_av_name.replace('_', ' ').title(),
                    "var_name": missing_av_name,
                    "priority": 5, # Low priority for auto-added
                    "description": f"Placeholder for {missing_av_name} referenced by a question but not initially defined.",
                    "formula": None,
                    "type": "text", # Default type
                    "value": None,
                    "project_id": state.get("project_id") # Add project_id
                }
                # Use the already guaranteed-to-be-list variable
                current_assessment_variables.append(placeholder_av) 
                av_varname_map[missing_av_name] = placeholder_av # Update map for current run
                existing_assessment_var_names_in_state.add(missing_av_name) # Add to set to prevent re-adding


    # Check for assessment variables whose calculation formula uses non-existent question variables
    problemmatic_calculation_vars = []
    # Identify all question variable names that currently exist in the questionnaire
    existing_question_var_names = set()
    for section in sections:
        for q_list in [section.get('core_questions', []), section.get('conditional_questions', [])]:
            for question in q_list:
                existing_question_var_names.add(question.get('variable_name'))

    for av_var_name, formula in av_calculation_map.items():
        if av_var_name in av_varname_map: # Only check if the AV itself exists
            # Find all 'q_' type variables in the formula
            import re
            formula_question_vars = re.findall(r'\b(q_[a-zA-Z0-9_]+)\b', formula)
            for fq_var in formula_question_vars:
                if fq_var not in existing_question_var_names:
                    problemmatic_calculation_vars.append(av_var_name)
                    print(f"Warning: Assessment variable '{av_var_name}' formula references missing question variable '{fq_var}'.")
                    break # Only need to flag once per AV

    # Combine all unique assessment variables that need attention
    # This list will now include any newly added placeholder AVs if they were referenced by questions
    # and also existing AVs that are not covered or have problematic formulas.
    vars_to_address = list(set(
        [var["var_name"] for var in uncovered_assessment_vars_by_question_impact if var["var_name"] in existing_assessment_var_names_in_state] +
        problemmatic_calculation_vars +
        list(referenced_but_missing_avs) # Include the newly created placeholder AVs here
    ))

    if vars_to_address:
        state["error"] = (state.get("error") or "") + "Warning: Some assessment variables are not fully covered by questionnaire questions or have problematic calculations." # Concatenate
        print(state["error"])
        print(f"Assessment variables needing attention: {', '.join(vars_to_address)}")

        # Attempt to generate new questions for uncovered/problemmatic variables
        print("\n---Attempting to generate new questions for affected assessment variables---")
        # Filter uncovered_vars_info to include newly added placeholder AVs that need questions
        uncovered_vars_info = [av_varname_map[var_name] for var_name in vars_to_address if var_name in av_varname_map]

        if uncovered_vars_info:
            remediation_prompt_template = ChatPromptTemplate.from_messages( # Re-defining here for specific remediation context
                [
                    ("system",
                     "You are an AI assistant tasked with generating missing survey questions. "
                     "A questionnaire has been modified, and some assessment variables are no longer "
                     "adequately captured or their calculation formulas are invalid due to missing question data. "
                     "Your goal is to suggest new, simple, and direct questions for the specified "
                     "assessment variables. "
                     "Output should be a JSON object containing an 'added_questions' array of Question objects, "
                     "and an 'updated_assessment_variable_calculation' dictionary. "
                     "For each question, ensure it has 'id' (unique string), 'text' (simple, clear language), "
                     "'type' (e.g., 'int', 'float', 'text'), 'variable_name' (unique snake_case), "
                     "'triggering_criteria' (null if not conditional), "
                     "and 'assessment_variables' (list of `var_name`s it helps capture). "
                     "**Also include a 'formula' for each added question, describing how its raw answer is interpreted or transformed to contribute.**"
                     "The 'updated_assessment_variable_calculation' mapping should provide new or updated "
                     "JavaScript formulas using the `variable_name`s of the questions you generate. "
                     "You should suggest adding these new questions to the most relevant or first mandatory section if possible."
                    ),
                    ("human",
                     "The following assessment variables need to be captured by new questions (and potentially their calculation mapping updated):\n{uncovered_vars_json}\n\n"
                     "Current questionnaire context (for appropriate placement of new questions and formula updates):\n{questionnaire_json}\n\n"
                     "Please generate suitable new questions for these variables and suggest updates to the assessment_variable_calculation."
                    )
                ]
            )

            remediation_chain = remediation_prompt_template | llm.with_structured_output(RemediationOutput, method='function_calling')

            try:
                remediation_response_raw = remediation_chain.invoke({ # Changed variable name to emphasize raw output
                    "uncovered_vars_json": json.dumps(uncovered_vars_info, indent=2),
                    "questionnaire_json": json.dumps(questionnaire, indent=2)
                })

                print(f"DEBUG: Raw remediation_response_raw from LLM: {remediation_response_raw}")
                print(f"DEBUG: Type of remediation_response_raw: {type(remediation_response_raw)}")

                if not isinstance(remediation_response_raw, dict):
                    print(f"ERROR: Expected remediation_response_raw to be a dict, but got {type(remediation_response_raw)}: {remediation_response_raw}")
                    state["error"] = (state.get("error") or "") + f"LLM remediation output format error: Expected dict, got {type(remediation_response_raw)}."
                    return state # This is a critical failure, stop this node.

                new_questions_data = remediation_response_raw.get("added_questions", [])
                updated_calc_map = remediation_response_raw.get("updated_assessment_variable_calculation", {})

                # Ensure updated_calc_map is indeed a dict before trying to update
                if not isinstance(updated_calc_map, dict):
                    print(f"ERROR: Expected updated_assessment_variable_calculation to be a dict, but got {type(updated_calc_map)}: {updated_calc_map}")
                    # If it's not a dict, default it to an empty dict to prevent further errors
                    updated_calc_map = {}
                    state["error"] = (state.get("error") or "") + "Warning: LLM remediation generated invalid calculation map. Defaulting to empty."

                print(f"DEBUG: Extracted updated_calc_map: {updated_calc_map}")
                print(f"DEBUG: Type of updated_calc_map: {type(updated_calc_map)}")


                if new_questions_data:
                    target_section = None
                    # Try to find an existing mandatory core section
                    for section in sections:
                        if section.get('is_mandatory', False) and section.get('core_questions') is not None:
                            target_section = section
                            break
                    if not target_section and sections: # If no mandatory core, try first existing section
                        target_section = sections[0]
                    
                    if not target_section: # If still no section, create a default one
                        print("No suitable existing section found for remediation. Creating a new 'Remediation' section.")
                        target_section = {
                            "title": "Remediation Questions",
                            "description": "Questions added to cover previously uncaptured assessment variables.",
                            "order": max([s['order'] for s in sections]) + 1 if sections else 1,
                            "is_mandatory": True,
                            "rationale": "Automatically generated to ensure data completeness.",
                            "core_questions": [],
                            "conditional_questions": [],
                            "triggering_criteria": None,
                            "data_validation": "return true;",
                            "project_id": state.get("project_id") # Add project_id
                        }
                        questionnaire["sections"].append(target_section)
                        questionnaire["sections"].sort(key=lambda x: x['order']) # Re-sort sections

                    print(f"Adding {len(new_questions_data)} remediation questions to section: {target_section.get('title', target_section['order'])}")
                    # Consolidate all existing question variable names to ensure uniqueness for new ones
                    existing_q_var_names_after_remediation_add = all_question_vars.copy() # Use a copy of the set

                    for new_q_data in new_questions_data:
                        # Pass the main state dict directly for error tracking
                        _process_question_properties(new_q_data, True, existing_q_var_names_after_remediation_add, state, project_id=state.get("project_id")) # Pass project_id
                        target_section['core_questions'].append(new_q_data)
                
                # Update assessment_variable_calculation map
                if updated_calc_map:
                    questionnaire["assessment_variable_calculation"].update(updated_calc_map)
                    print("Updated assessment variable calculation map with new entries.")

                state["questionnaire"] = questionnaire # Update state with modified questionnaire
                # Update error message to indicate remediation was attempted
                state["error"] = (state.get("error") or "") + f"Warning: Some assessment variables were flagged and an attempt was made to add questions. Review updated questionnaire and calculation map." # Concatenate
                print("\n---Remediation complete. Please review the updated questionnaire and calculation map.---")

            except Exception as e:
                state["error"] = (state.get("error") or "") + f"Error during questionnaire impact remediation: {e}" # Concatenate
                print(f"Error during questionnaire impact remediation: {e}")
                # Do NOT return state here, allow processing to continue to next node
                return state

    else:
        print("All assessment variables are covered by the questionnaire. No impact flagged.")
        state["error"] = None # Clear any previous error if remediation fixed it

    return state

# --- Langraph Node 6: write_to_supabase ---
def write_to_supabase(state: GraphState) -> GraphState:
    """
    Writes the generated (and potentially LLM-modified) assessment and computational variables
    and questionnaire questions to your Supabase tables. This function now performs
    upsert operations (update if exists, insert if new) for individual records.
    """
    print("\n---WRITING TO SUPABASE---")
    # Ensure state["error"] is a string at the start of this node
    state["error"] = state.get("error", "")

    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json"
    }
    error_occurred = False

    # --- Write Assessment Variables ---
    assessment_vars_to_write = state.get("assessment_variables")
    if assessment_vars_to_write:
        print(f"Attempting to upsert {len(assessment_vars_to_write)} assessment variables...")
        for var in assessment_vars_to_write:
            # The 'id' in 'var' is already project-prefixed due to _apply_default_variable_properties
            # Ensure project_id is correctly set in the item payload
            var["project_id"] = state.get("project_id")
            # Using 'asessment_variables' (single 's') as per user's confirmation
            if not _upsert_single_item("asessment_variables", var, headers): 
                error_occurred = True
    else:
        print("No assessment variables found in state to write to Supabase.")

    # --- Write Computational Variables ---
    computational_vars_to_write = state.get("computational_variables")
    if computational_vars_to_write:
        print(f"Attempting to upsert {len(computational_vars_to_write)} computational variables...")
        for var in computational_vars_to_write:
            # The 'id' in 'var' is already project-prefixed
            var["project_id"] = state.get("project_id")
            if not _upsert_single_item("computational_variables", var, headers):
                error_occurred = True
    else:
        print("No computational variables found in state to write to Supabase.")

    # --- Write Questionnaire Questions to 'questions' table ---
    questionnaire_data = state.get("questionnaire")
    # For mapping AVs to ID/Name, ensure we use the project-prefixed IDs from the current state
    assessment_variables_map = {av['var_name']: av for av in (state.get("assessment_variables") or [])}

    questions_to_supabase = []
    if questionnaire_data and questionnaire_data.get("sections"):
        for section in questionnaire_data["sections"]:
            # Process core questions
            for q_type_list, is_core_q_flag in [
                (section.get('core_questions', []), True),
                (section.get('conditional_questions', []), False)
            ]:
                for question in q_type_list:
                    # The is_mandatory field for the question in Supabase should reflect the section's mandatory status.
                    # Individual question's conditional logic is handled by question_triggering_criteria.
                    is_q_mandatory_in_db = section.get('is_mandatory', False)

                    impacted_av_details = []
                    # The `av_var_name` here is the *base* variable name (e.g., 'avg_monthly_income') from the question's assessment_variables list.
                    # We need to find the corresponding assessment variable object in our `state["assessment_variables"]` which now has project-prefixed IDs.
                    # The `assessment_variables_map` already helps us find the full AV object by its `var_name`.
                    for av_var_name in question.get('assessment_variables', []):
                        av_detail = assessment_variables_map.get(av_var_name)
                        if av_detail:
                            impacted_av_details.append({
                                "id": av_detail["id"], # This will now be the project-prefixed ID
                                "name": av_detail["name"]
                            })
                        else:
                            print(f"Warning: Assessment variable '{av_var_name}' not found for question '{question.get('variable_name')}' during Supabase write.")

                    question_entry = {
                        "id": question["id"], # This is already project-prefixed
                        "project_id": state.get("project_id"), # Add project_id
                        "section_number": section["order"],
                        "question_name": question["text"],
                        "section_name": section["title"],
                        "section_description": section["description"],
                        "is_mandatory": is_q_mandatory_in_db, # Updated as per user's clarification
                        "section_triggering_criteria": section["triggering_criteria"],
                        "question_var_name": question["variable_name"],
                        "impacted_assessment_variables": json.dumps(impacted_av_details), # Store as JSON string for jsonb field
                        "question_triggering_criteria": question["triggering_criteria"],
                        "is_conditional": question.get("is_conditional", False), # New: Add is_conditional field
                        "formula": question.get("formula") # Include the new formula field
                    }
                    questions_to_supabase.append(question_entry)
        
        print(f"Attempting to upsert {len(questions_to_supabase)} questions...")
        for q_entry in questions_to_supabase:
            if not _upsert_single_item("questions", q_entry, headers):
                error_occurred = True
    else:
        print("No questionnaire sections found in state to write to Supabase 'questions' table.")

    if error_occurred:
        pass
    else:
        state["error"] = None

    return state