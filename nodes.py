import os
import json
import uuid
import requests
import re
from typing import List, Dict, Optional, Any, cast
from dotenv import load_dotenv
import time # Import for sleep function

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import openai

# Import prompts from the new prompts.py file
from prompts import (
    RAW_INDICATORS_PROMPT,
    DECISION_VARIABLES_PROMPT,
    INTELLIGENT_VARIABLE_MODIFICATIONS_PROMPT,
    DEPENDENCY_ANALYSIS_PROMPT,
    QUESTIONNAIRE_PROMPT,
    JS_REFINEMENT_PROMPT,
    INTELLIGENT_QUESTIONNAIRE_MODIFICATIONS_PROMPT,
    EXPORT_SECTION_CARDS_PROMPT
)

# Import schemas from the new schemas.py file
from schemas.schemas import (
    GraphState,
    VariableSchema,
    RawIndicatorsOutput,
    DecisionVariablesOutput,
    IntelligentVariableModificationsOutput,
    DependencyInfo,
    ImpactAnalysis,
    DependencyGraph,
    IntelligentModificationRequest,
    SynchronizationPlan,
    Question,
    Section,
    QuestionnaireOutput,
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

def _apply_default_variable_properties(var: Dict, is_raw_indicator: bool = True, project_id: Optional[str] = None):
    """
    Helper to apply default properties to a variable (raw indicator or decision variable).
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

    if 'impact_score' not in var or not isinstance(var['impact_score'], int):
        var['impact_score'] = 50 # Default impact_score (midpoint)
    if 'priority' in var:
        del var['priority']

    if 'description' not in var or not var['description']:
        var['description'] = f"{'Raw indicator' if is_raw_indicator else 'Decision variable'} for {var['name']}"
    
    # Add priority_rationale if not present
    if 'priority_rationale' not in var or not var['priority_rationale']:
        var['priority_rationale'] = f"Priority {var.get('impact_score', 50)} assigned based on importance for income assessment"
    
    if is_raw_indicator:
        var["formula"] = None # Raw indicators have null formula
        if 'type' not in var or not var['type']:
            var['type'] = "text"
    else: # Decision variable
        if 'formula' not in var or not var['formula']:
            var['formula'] = "// Placeholder formula - please define based on raw indicators"
        if 'type' not in var or not var['type']:
            var['type'] = "float" # Decision variables are often numerical

def _process_section_properties(section: Dict, all_existing_q_vars: set, state_error_ref: GraphState, project_id: Optional[str] = None):
    """
    Process and validate section properties.
    - is_mandatory defaults to True
    - If is_mandatory is False, triggering_criteria is required
    """
    section["project_id"] = project_id
    
    # Handle is_mandatory (default: True)
    is_mandatory = section.get('is_mandatory', True)
    section['is_mandatory'] = is_mandatory
    
    # Validate triggering criteria
    if not is_mandatory:
        if 'triggering_criteria' not in section or not section['triggering_criteria']:
            state_error_ref["error"] = f"Optional section missing triggering_criteria: {section.get('title', 'Untitled')}"
    elif 'triggering_criteria' in section:
        section['triggering_criteria'] = None  # Clear triggering criteria for mandatory sections
    
    # Ensure both question lists exist
    if 'core_questions' not in section:
        section['core_questions'] = []
    if 'conditional_questions' not in section:
        section['conditional_questions'] = []
    
    return section

def _process_question_properties(question: Dict, is_core_question: bool, all_existing_q_vars: set, state_error_ref: GraphState, project_id: Optional[str] = None):
    """
    Process and validate question properties.
    - is_conditional defaults to False
    - If is_conditional is True, question_triggering_criteria is required
    """
    # Ensure the Supabase ID is unique per project and item
    base_id = question.get("id") or str(uuid.uuid4())
    if project_id:  # Only prepend if project_id is available
        question["id"] = f"{project_id}_{base_id}"
    else:
        question["id"] = base_id

    # Handle is_conditional (default: False)
    is_conditional = question.get('is_conditional', False)
    question['is_conditional'] = is_conditional
    
    # Validate question_triggering_criteria
    if is_conditional:
        if 'question_triggering_criteria' not in question or not question['question_triggering_criteria']:
            state_error_ref["error"] = f"Conditional question missing question_triggering_criteria: {question.get('text', 'Untitled')}"
    elif 'question_triggering_criteria' in question:
        question['question_triggering_criteria'] = None  # Clear triggering criteria for non-conditional questions
    
    # Validate other required fields
    required_fields = ['text', 'type', 'variable_name', 'raw_indicators', 'formula']
    for field in required_fields:
        if field not in question or not question[field]:
            state_error_ref["error"] = f"Question missing required field '{field}': {question.get('text', 'Untitled')}"
            return question

    # Add question's variable name to the set of all question variables
    all_existing_q_vars.add(question['variable_name'])
    
    return question

# --- Langraph Node 1: generate_variables ---
def generate_variables(state: GraphState) -> GraphState:
    """
    Generates initial raw indicators and decision variables based on the user's prompt.
    It prompts an LLM twice: first for raw indicators, then for decision variables
    based on the suggested raw indicators.
    """
    print("---GENERATING INITIAL VARIABLES---")
    # Ensure state["error"] is a string at the start of this node
    state["error"] = state.get("error", "")

    prompt_text = state["prompt"]
    project_id = state.get("project_id") # Get project_id from state
    current_raw_indicators = state.get("raw_indicators", [])
    current_decision_variables = state.get("decision_variables", [])

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

    # Step 1: Identify Raw Indicators using LLM
    if not current_raw_indicators:
        print("Generating Raw Indicators...")
        raw_indicators_chain = RAW_INDICATORS_PROMPT | llm.with_structured_output(RawIndicatorsOutput, method='function_calling')

        try:
            llm_response = raw_indicators_chain.invoke({
                "user_input": prompt_text,
                "existing_variables": json.dumps(current_raw_indicators),
                "context": context_docs # Pass RAG context
            })
            suggested_raw_indicators = llm_response.get("raw_indicators", [])

            for var in suggested_raw_indicators:
                _apply_default_variable_properties(var, is_raw_indicator=True, project_id=project_id) # Pass project_id

            state["raw_indicators"] = suggested_raw_indicators
            print("\n---Initial Suggested Raw Indicators:---")
            print(json.dumps(suggested_raw_indicators, indent=2))

        except Exception as e:
            state["error"] = (state.get("error") or "") + f"Error generating raw indicators: {e}" # Concatenate
            print(f"Error generating raw indicators: {e}")
            return state # Critical failure, stop workflow

    # Step 2: Create Decision Variables using LLM
    if state["raw_indicators"] and not current_decision_variables:
        print("\nGenerating Decision Variables...")
        raw_indicator_names = ", ".join([v["var_name"] for v in state["raw_indicators"]])

        # Use RAG context for decision variables based on raw indicator names
        decision_context_docs = []
        if retriever:
            try:
                print(f"Retrieving RAG context for decision variables based on: {raw_indicator_names}")
                decision_context_docs = retriever.invoke(raw_indicator_names)
                print(f"Retrieved {len(decision_context_docs)} context documents for decision variables.")
            except Exception as e:
                print(f"Warning: Could not retrieve RAG context for decision variables: {e}")
                decision_context_docs = []

        decision_variables_chain = DECISION_VARIABLES_PROMPT | llm.with_structured_output(DecisionVariablesOutput, method='function_calling')

        try:
            llm_response = decision_variables_chain.invoke({
                "raw_indicators": json.dumps([{"var_name": v["var_name"], "name": v["name"], "type": v["type"]} for v in state["raw_indicators"]]),
                "existing_decision_variables": json.dumps(current_decision_variables),
                "user_input": prompt_text,
                "context": decision_context_docs # Pass RAG context
            })
            suggested_decision_vars = llm_response.get("decision_variables", [])

            for var in suggested_decision_vars:
                _apply_default_variable_properties(var, is_raw_indicator=False, project_id=project_id) # Pass project_id

            state["decision_variables"] = suggested_decision_vars
            print("\n---Initial Suggested Decision Variables:---")
            print(json.dumps(suggested_decision_vars, indent=2))

        except Exception as e:
            state["error"] = (state.get("error") or "") + f"Error generating decision variables: {e}" # Concatenate
            print(f"Error generating decision variables: {e}")
            return state # Critical failure, stop workflow

    return state

def modify_variables_intelligent(state: GraphState) -> GraphState:
    """
    Enhanced variable modification with intelligent synchronization.
    """
    print("---INTELLIGENT VARIABLE MODIFICATION---")
    
    modification_prompt = state.get("modification_prompt")
    if not modification_prompt:
        print("No modification prompt provided. Skipping intelligent variable modification.")
        return state
    
    raw_indicators = state.get("raw_indicators", []) or []
    decision_variables = state.get("decision_variables", []) or []
    dependency_graph = state.get("dependency_graph", {})
    project_id = state.get("project_id", "") or ""
    
    if not dependency_graph:
        print("No dependency graph available. Running dependency analysis first...")
        state = analyze_variable_dependencies_node(state)
        dependency_graph = state.get("dependency_graph", {})
    
    try:
        # Prepare the intelligent modification request
        business_context = f"Financial assessment for small business income evaluation. Project ID: {project_id}"
        
        # Determine modification type based on the prompt
        modification_type = determine_modification_type(modification_prompt)
        
        intelligent_request = {
            "primary_modifications": modification_prompt,
            "dependency_analysis": json.dumps(dependency_graph, indent=2),
            "auto_sync_enabled": True,
            "business_context": business_context,
            "modification_type": modification_type
        }
        
        # Use the intelligent modification prompt
        intelligent_chain = INTELLIGENT_VARIABLE_MODIFICATIONS_PROMPT | llm.with_structured_output(
            IntelligentVariableModificationsOutput, method='function_calling'
        )
        
        llm_response = intelligent_chain.invoke({
            "primary_modifications": modification_prompt,
            "dependency_analysis": json.dumps(dependency_graph, indent=2),
            "raw_indicators": json.dumps([{"var_name": ri["var_name"], "name": ri["name"]} for ri in raw_indicators]),
            "business_context": business_context
        })
        # print("[DEBUG] LLM Response from intelligent modification:")
        # print(json.dumps(llm_response, indent=2, default=str))
        
        # Apply the intelligent modifications
        state = apply_intelligent_modifications(state, llm_response, project_id)
        
        # Print only a summary of the modified state
        print("\n--- Modified State After Modification ---")
        print("Raw Indicators:")
        raw_indicators_list = state.get("raw_indicators")
        if raw_indicators_list is None:
            raw_indicators_list = []
        for ri in raw_indicators_list:
            print(f"  - {ri.get('name')} (var_name: {ri.get('var_name')}, type: {ri.get('type')})")
        print("Decision Variables:")
        decision_variables_list = state.get("decision_variables")
        if decision_variables_list is None:
            decision_variables_list = []
        for dv in decision_variables_list:
            print(f"  - {dv.get('name')} (var_name: {dv.get('var_name')}, type: {dv.get('type')}, formula: {dv.get('formula')})")
        if state.get("modification_reasoning"):
            print("Reasoning:")
            print(state.get("modification_reasoning"))
        if state.get("error"):
            print("Error:")
            print(state.get("error"))
        print("--- End of Modification ---\n")
        
        # print("[DEBUG] State after applying intelligent modifications:")
        # print(json.dumps({
        #     "raw_indicators": state.get("raw_indicators"),
        #     "decision_variables": state.get("decision_variables"),
        #     "modification_reasoning": state.get("modification_reasoning"),
        #     "error": state.get("error")
        # }, indent=2, default=str))
        
        print("Intelligent variable modification completed.")
        
    except Exception as e:
        state["error"] = (state.get("error") or "") + f"Error in intelligent variable modification: {e}"
        print(f"Error in intelligent variable modification: {e}")
    
    return state

def determine_modification_type(modification_prompt: str) -> str:
    """
    Determines whether the modification primarily affects raw indicators, decision variables, or both.
    """
    prompt_lower = modification_prompt.lower()
    
    raw_indicator_keywords = ['raw indicator', 'assessment variable', 'data point', 'input variable']
    decision_variable_keywords = ['decision variable', 'computed variable', 'calculated variable', 'formula']
    
    ri_count = sum(1 for keyword in raw_indicator_keywords if keyword in prompt_lower)
    dv_count = sum(1 for keyword in decision_variable_keywords if keyword in prompt_lower)
    
    if ri_count > dv_count:
        return 'raw_indicators'
    elif dv_count > ri_count:
        return 'decision_variables'
    else:
        return 'both'

def apply_intelligent_modifications(state: GraphState, llm_response: Dict, project_id: str) -> GraphState:
    """
    Applies the intelligent modifications returned by the LLM.
    """
    # print("[DEBUG] Entering apply_intelligent_modifications...")
    # print("[DEBUG] LLM Response:")
    # print(json.dumps(llm_response, indent=2, default=str))
    raw_indicators = list(state.get("raw_indicators", []) or [])
    decision_variables = list(state.get("decision_variables", []) or [])
    
    # Apply primary modifications
    primary_mods = llm_response.get("primary_modifications", {})
    # print("[DEBUG] Primary Modifications:")
    # print(json.dumps(primary_mods, indent=2, default=str))
    
    # Apply compensatory modifications
    compensatory_mods = llm_response.get("compensatory_modifications", {})
    # print("[DEBUG] Compensatory Modifications:")
    # print(json.dumps(compensatory_mods, indent=2, default=str))
    
    # Handle removed variables
    removed_vars = llm_response.get("removed_variables", [])
    # print(f"[DEBUG] Variables to remove: {removed_vars}")
    for var_name in removed_vars:
        # Remove from raw indicators
        raw_indicators = [ri for ri in raw_indicators if ri.get('var_name') != var_name]
        # Remove from decision variables
        decision_variables = [dv for dv in decision_variables if dv.get('var_name') != var_name]
    
    # Handle new variables
    new_vars = llm_response.get("new_variables", [])
    # print(f"[DEBUG] New variables to add: {new_vars}")
    for new_var in new_vars:
        if new_var.get('formula'):  # Decision variable
            _apply_default_variable_properties(new_var, is_raw_indicator=False, project_id=project_id)
            decision_variables.append(new_var)
        else:  # Raw indicator
            _apply_default_variable_properties(new_var, is_raw_indicator=True, project_id=project_id)
            raw_indicators.append(new_var)
    
    # Handle formula updates
    updated_formulas = llm_response.get("updated_formulas", {})
    # print(f"[DEBUG] Updated formulas: {updated_formulas}")
    for var_name, new_formula in updated_formulas.items():
        # Update decision variable formula
        for dv in decision_variables:
            if dv.get('var_name') == var_name:
                dv['formula'] = new_formula
                break
    
    # Update state
    state["raw_indicators"] = raw_indicators
    state["decision_variables"] = decision_variables
    
    # Store reasoning for transparency
    reasoning = llm_response.get("reasoning", "")
    if reasoning:
        state["modification_reasoning"] = reasoning
    # print("[DEBUG] State after all modifications:")
    # print(json.dumps({
    #     "raw_indicators": state.get("raw_indicators"),
    #     "decision_variables": state.get("decision_variables"),
    #     "modification_reasoning": state.get("modification_reasoning"),
    #     "error": state.get("error")
    # }, indent=2, default=str))
    
    return state

# --- Langraph Node 3: generate_questionnaire ---
def generate_questionnaire(state: GraphState) -> GraphState:
    """
    Generates a questionnaire based on the raw indicators and decision variables.
    Also generates a title for the questionnaire and stores it in state['questionnaire_title'].
    """
    print("\n---GENERATING QUESTIONNAIRE---")
    state["error"] = state.get("error", "")

    prompt_text = state["prompt"]
    project_id = state.get("project_id")
    raw_indicators = state.get("raw_indicators", [])
    decision_variables = state.get("decision_variables")

    if decision_variables is None:
        decision_variables = []

    if not raw_indicators:
        state["error"] = (state.get("error") or "") + "No raw indicators available for questionnaire generation."
        print("No raw indicators available for questionnaire generation.")
        return state

    prompt_context = f"User request: {prompt_text}. Generate a questionnaire to assess income for small business owners."

    raw_indicators_info = [
        {
            "var_name": var["var_name"],
            "name": var["name"],
            "description": var["description"],
            "type": var["type"]
        } for var in raw_indicators
    ]
    raw_indicators_json = json.dumps(raw_indicators_info, indent=2)

    decision_vars_info = [
        {
            "var_name": var["var_name"],
            "name": var["name"],
            "description": var["description"],
            "type": var["type"]
        } for var in decision_variables
    ]
    decision_vars_json = json.dumps(decision_vars_info, indent=2)

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
            "raw_indicators": raw_indicators_json,
            "decision_variables": decision_vars_json,
            "context": context_docs
        })
        generated_questionnaire = llm_response

        

        # Generate a title for the questionnaire
        title = generated_questionnaire.get("title")
        if not title or not title.strip():
            # Fallback: use the first 3 words of the prompt, title-cased, with '...' at the end
            words = prompt_text.strip().split()
            title = " ".join(words[:6]).title() + "..."
        print(f"[DEBUG] Extracted questionnaire title: '{title}'")  # DEBUG PRINT
        state["questionnaire_title"] = title

        # Collect initial q_vars from generated_questionnaire to populate all_existing_q_vars_set
        all_existing_q_vars_set = set()
        for section in generated_questionnaire.get("sections", []):
            for q_list in [section.get('core_questions', []), section.get('conditional_questions', [])]:
                for question in q_list:
                    if question.get('variable_name'):
                        all_existing_q_vars_set.add(question['variable_name'])

        for sec_idx, section in enumerate(generated_questionnaire.get("sections", [])):
            section['order'] = section.get('order', sec_idx + 1)
            _process_section_properties(section, all_existing_q_vars_set, state, project_id=project_id)
            for q_list, is_core_q_flag in [
                (section['core_questions'], True),
                (section['conditional_questions'], False)
            ]:
                for q_idx, question in enumerate(q_list):
                    _process_question_properties(question, is_core_q_flag, all_existing_q_vars_set, state, project_id=project_id)

        state["questionnaire"] = generated_questionnaire
        print("\n---Generated Questionnaire:---")
        print(json.dumps(generated_questionnaire, indent=2))

    except Exception as e:
        state["error"] = (state.get("error") or "") + f"Error generating questionnaire: {e}"
        print(f"Error generating questionnaire: {e}")

    return state



# --- Langraph Node 4: modify_questionnaire_llm ---
def modify_questionnaire_llm(state: GraphState) -> GraphState:
    """
    Enhanced questionnaire modification with intelligent analysis and reasoning.
    """
    print("\n---MODIFYING QUESTIONNAIRE USING LLM---")
    
    modification_prompt = state.get("modification_prompt")
    if not modification_prompt:
        print("No modification prompt provided. Skipping questionnaire modification.")
        return state
    
    current_questionnaire = state.get("questionnaire")
    if not current_questionnaire:
        state["error"] = "No questionnaire available for modification."
        print("No questionnaire available for modification.")
        return state
    
    raw_indicators = state.get("raw_indicators", []) or []
    project_id = state.get("project_id", "") or ""
    
    try:
        # Prepare business context
        business_context = f"Financial assessment questionnaire for small business income evaluation. Project ID: {project_id}"
        
        # Use the intelligent modification prompt
        intelligent_chain = INTELLIGENT_QUESTIONNAIRE_MODIFICATIONS_PROMPT | llm.with_structured_output(
            QuestionnaireModificationsOutput, method='function_calling'
        )
        
        llm_response = intelligent_chain.invoke({
            "business_context": business_context,
            "raw_indicators": json.dumps([{"var_name": ri["var_name"], "name": ri["name"]} for ri in raw_indicators]),
            "current_questionnaire": json.dumps(current_questionnaire, indent=2),
            "modification_prompt": modification_prompt
        })
        
        # Store the modification reasoning
        if llm_response.get("reasoning"):
            state["modification_reasoning"] = llm_response["reasoning"]
            print("\n--- Modification Reasoning ---")
            print(llm_response["reasoning"])
        
        modifications = llm_response
        modified_questionnaire = current_questionnaire.copy()
        modified_sections = [dict(sec) for sec in modified_questionnaire.get("sections", [])]
        
        # Ensure raw_indicator_calculation exists and is a dict
        if "raw_indicator_calculation" not in modified_questionnaire or modified_questionnaire["raw_indicator_calculation"] is None:
            modified_questionnaire["raw_indicator_calculation"] = {}
        
        modified_ri_calc = modified_questionnaire["raw_indicator_calculation"].copy()
        
        # --- Apply Section Modifications ---
        
        # Remove sections
        removed_section_orders = modifications.get("removed_section_orders", [])
        if removed_section_orders:
            modified_sections = [sec for sec in modified_sections if sec.get("order") not in removed_section_orders]
            print(f"Removed {len(removed_section_orders)} sections.")
        
        # Update existing sections
        updated_sections = modifications.get("updated_sections", [])
        for update in updated_sections:
            for section in modified_sections:
                if section.get("order") == update.get("order"):
                    section.update(update)
        
        # Add new sections
        new_sections = modifications.get("added_sections", [])
        if new_sections:
            # Ensure new sections have unique orders
            existing_orders = {sec.get("order") for sec in modified_sections}
            for new_sec in new_sections:
                while new_sec.get("order") in existing_orders:
                    new_sec["order"] = new_sec.get("order", 1) + 1
                modified_sections.append(new_sec)
                existing_orders.add(new_sec.get("order"))
            print(f"Added {len(new_sections)} new sections.")
        
        # --- Apply Question Modifications ---
        
        # Remove questions
        removed_q_vars = modifications.get("removed_question_variable_names", [])
        if removed_q_vars:
            for section in modified_sections:
                for qtype in ["core_questions", "conditional_questions"]:
                    original_questions = section.get(qtype, [])
                    filtered_questions = [q for q in original_questions if q.get("variable_name") not in removed_q_vars]
                    section[qtype] = filtered_questions
            print(f"Removed {len(removed_q_vars)} questions by variable_name.")
        
        # Update existing questions
        updated_questions = modifications.get("updated_questions", [])
        for update in updated_questions:
            for section in modified_sections:
                for qtype in ["core_questions", "conditional_questions"]:
                    for question in section.get(qtype, []):
                        if question.get("variable_name") == update.get("variable_name"):
                            question.update(update)
        
        # Add new questions
        added_questions = modifications.get("added_questions", [])
        if added_questions:
            # Track existing variable names to prevent duplicates
            existing_var_names = set()
            for section in modified_sections:
                for qtype in ["core_questions", "conditional_questions"]:
                    for q in section.get(qtype, []):
                        existing_var_names.add(q.get("variable_name"))
            
            for new_q in added_questions:
                target_section = None
                section_order = new_q.get("section_order")
                is_core = new_q.get("is_core", True)
                
                # Find the target section
                for section in modified_sections:
                    if section.get("order") == section_order:
                        target_section = section
                        break
                
                if target_section:
                    question = new_q.get("question", {})
                    if question.get("variable_name") not in existing_var_names:
                        _process_question_properties(question, is_core, existing_var_names, state, project_id=project_id)
                        qtype = "core_questions" if is_core else "conditional_questions"
                        if qtype not in target_section:
                            target_section[qtype] = []
                        target_section[qtype].append(question)
                        existing_var_names.add(question.get("variable_name"))
            print(f"Added {len(added_questions)} new questions.")
        
        # Update the questionnaire with modified sections
        modified_questionnaire["sections"] = modified_sections
        state["questionnaire"] = modified_questionnaire
        
        print("\n--- Questionnaire Modification Summary ---")
        print(f"Sections removed: {len(removed_section_orders)}")
        print(f"Sections updated: {len(updated_sections)}")
        print(f"Sections added: {len(new_sections)}")
        print(f"Questions removed: {len(removed_q_vars)}")
        print(f"Questions updated: {len(updated_questions)}")
        print(f"Questions added: {len(added_questions)}")
        if state.get("modification_reasoning"):
            print("\nReasoning:")
            print(state.get("modification_reasoning"))
        
    except Exception as e:
        state["error"] = (state.get("error") or "") + f"Error modifying questionnaire: {e}"
        print(f"Error modifying questionnaire: {e}")
    
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
    Analyzes the questionnaire to ensure all raw indicators can be calculated.
    If variables are uncalculable, it attempts to generate new questions to cover them.
    """
    print("\n---ANALYZING QUESTIONNAIRE IMPACT---")
    # Ensure state["error"] is a string at the start of this node
    state["error"] = state.get("error", "")

    # Ensure raw_indicators is always a list at the start of this node
    current_raw_indicators = state.get("raw_indicators")
    if current_raw_indicators is None:
        current_raw_indicators = []
        state["raw_indicators"] = current_raw_indicators # Update state if it was None

    raw_indicators = current_raw_indicators # Use the guaranteed list from now on
    questionnaire = state.get("questionnaire") or {}
    sections = questionnaire.get("sections", [])
    
    # Initialize raw_indicator_calculation if it's None or missing
    if "raw_indicator_calculation" not in questionnaire or questionnaire["raw_indicator_calculation"] is None:
        questionnaire["raw_indicator_calculation"] = {}
    ri_calculation_map = questionnaire["raw_indicator_calculation"]


    if not raw_indicators:
        print("No raw indicators to analyze impact.")
        return state
    if not sections:
        print("No questionnaire sections to analyze impact.")
        return state

    # Create a quick lookup for raw indicator objects by var_name
    ri_varname_map = {ri['var_name']: ri for ri in raw_indicators}

    # Track which raw indicators are explicitly covered by raw_indicators list in questions
    explicitly_covered_ris = set()
    all_question_vars = set()
    # New set to track RIs referenced by questions but not existing in state['raw_indicators']
    referenced_but_missing_ris = set() 

    for section in sections:
        for q_list in [section.get('core_questions', []), section.get('conditional_questions', [])]:
            for question in q_list:
                all_question_vars.add(question.get('variable_name'))
                for ri_name in question.get('raw_indicators', []):
                    explicitly_covered_ris.add(ri_name)
                    # Check if this referenced RI exists in our master list
                    if ri_name not in ri_varname_map:
                        referenced_but_missing_ris.add(ri_name)

    # All raw indicators that are supposed to exist based on `state['raw_indicators']`
    existing_raw_indicator_names_in_state = {ri['var_name'] for ri in raw_indicators}

    # Combine uncovered RIs that are *expected* to exist but aren't explicitly covered by questions
    uncovered_raw_indicators_by_question_impact = [
        var for var in raw_indicators
        if var["var_name"] not in explicitly_covered_ris
    ]

    # Add placeholder RIs for those referenced by questions but not existing in state
    if referenced_but_missing_ris:
        print(f"Detected questions referencing missing raw indicators: {', '.join(referenced_but_missing_ris)}")
        for missing_ri_name in referenced_but_missing_ris:
            if missing_ri_name not in existing_raw_indicator_names_in_state: # Only add if truly missing
                print(f"Adding placeholder raw indicator for: {missing_ri_name}")
                placeholder_ri = {
                    "id": str(uuid.uuid4()),
                    "name": missing_ri_name.replace('_', ' ').title(),
                    "var_name": missing_ri_name,
                    "impact_score": 50, # Low impact_score for auto-added
                    "priority_rationale": "Auto-generated placeholder for missing raw indicator",
                    "description": f"Placeholder for {missing_ri_name} referenced by a question but not initially defined.",
                    "formula": None,
                    "type": "text", # Default type
                    "value": None,
                    "project_id": state.get("project_id") # Add project_id
                }
                # Use the already guaranteed-to-be-list variable
                current_raw_indicators.append(placeholder_ri) 
                ri_varname_map[missing_ri_name] = placeholder_ri # Update map for current run
                existing_raw_indicator_names_in_state.add(missing_ri_name) # Add to set to prevent re-adding


    # Check for raw indicators whose calculation formula uses non-existent question variables
    problemmatic_calculation_vars = []
    # Identify all question variable names that currently exist in the questionnaire
    existing_question_var_names = set()
    for section in sections:
        for q_list in [section.get('core_questions', []), section.get('conditional_questions', [])]:
            for question in q_list:
                existing_question_var_names.add(question.get('variable_name'))

    for ri_var_name, formula in ri_calculation_map.items():
        if ri_var_name in ri_varname_map: # Only check if the RI itself exists
            # Find all 'q_' type variables in the formula
            import re
            formula_question_vars = re.findall(r'\b(q_[a-zA-Z0-9_]+)\b', formula)
            for fq_var in formula_question_vars:
                if fq_var not in existing_question_var_names:
                    problemmatic_calculation_vars.append(ri_var_name)
                    print(f"Warning: Raw indicator '{ri_var_name}' formula references missing question variable '{fq_var}'.")
                    break # Only need to flag once per RI

    # Combine all unique raw indicators that need attention
    # This list will now include any newly added placeholder RIs if they were referenced by questions
    # and also existing RIs that are not covered or have problematic formulas.
    vars_to_address = list(set(
        [var["var_name"] for var in uncovered_raw_indicators_by_question_impact if var["var_name"] in existing_raw_indicator_names_in_state] +
        problemmatic_calculation_vars +
        list(referenced_but_missing_ris) # Include the newly created placeholder RIs here
    ))

    if vars_to_address:
        state["error"] = (state.get("error") or "") + "Warning: Some raw indicators are not fully covered by questionnaire questions or have problematic calculations." # Concatenate
        print(state["error"])
        print(f"Raw indicators needing attention: {', '.join(vars_to_address)}")

        # Attempt to generate new questions for uncovered/problemmatic variables
        print("\n---Attempting to generate new questions for affected raw indicators---")
        # Filter uncovered_vars_info to include newly added placeholder RIs that need questions
        uncovered_vars_info = [ri_varname_map[var_name] for var_name in vars_to_address if var_name in ri_varname_map]

        if uncovered_vars_info:
            remediation_prompt_template = ChatPromptTemplate.from_messages( # Re-defining here for specific remediation context
                [
                    ("system",
                     "You are an AI assistant tasked with generating missing survey questions. "
                     "A questionnaire has been modified, and some raw indicators are no longer "
                     "adequately captured or their calculation formulas are invalid due to missing question data. "
                     "Your goal is to suggest new, simple, and direct questions for the specified "
                     "raw indicators. "
                     "Output should be a JSON object containing an 'added_questions' array of Question objects, "
                     "and an 'updated_raw_indicator_calculation' dictionary. "
                     "For each question, ensure it has 'id' (unique string), 'text' (simple, clear language), "
                     "'type' (e.g., 'int', 'float', 'text'), 'variable_name' (unique snake_case), "
                     "'triggering_criteria' (null if not conditional), "
                     "and 'raw_indicators' (list of `var_name`s it helps capture). "
                     "**Also include a 'formula' for each added question, describing how its raw answer is interpreted or transformed to contribute.**"
                     "The 'updated_raw_indicator_calculation' mapping should provide new or updated "
                     "JavaScript formulas using the `variable_name`s of the questions you generate. "
                     "You should suggest adding these new questions to the most relevant or first mandatory section if possible."
                    ),
                    ("human",
                     "The following raw indicators need to be captured by new questions (and potentially their calculation mapping updated):\n{uncovered_vars_json}\n\n"
                     "Current questionnaire context (for appropriate placement of new questions and formula updates):\n{questionnaire_json}\n\n"
                     "Please generate suitable new questions for these variables and suggest updates to the raw_indicator_calculation."
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
                updated_calc_map = remediation_response_raw.get("updated_raw_indicator_calculation", {})

                # Ensure updated_calc_map is indeed a dict before trying to update
                if not isinstance(updated_calc_map, dict):
                    print(f"ERROR: Expected updated_raw_indicator_calculation to be a dict, but got {type(updated_calc_map)}: {updated_calc_map}")
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
                    if target_section:
                        # Prevent duplicate variable_name in the section
                        existing_var_names = {q['variable_name'] for q in target_section.get('core_questions', [])}
                        added_count = 0
                        for new_q_data in new_questions_data:
                            if new_q_data['variable_name'] not in existing_var_names:
                                _process_question_properties(new_q_data, True, existing_var_names, state, project_id=state.get("project_id"))
                                target_section['core_questions'].append(new_q_data)
                                existing_var_names.add(new_q_data['variable_name'])
                                added_count += 1
                            else:
                                print(f"Warning: Skipped adding duplicate question with variable_name '{new_q_data['variable_name']}' to section '{target_section.get('title', target_section['order'])}'.")
                        print(f"Adding {added_count} remediation questions to section: {target_section.get('title', target_section['order'])}")

                # Update raw_indicator_calculation map
                if updated_calc_map:
                    questionnaire["raw_indicator_calculation"].update(updated_calc_map)
                    print("Updated raw_indicator_calculation map with new entries.")

                state["questionnaire"] = questionnaire # Update state with modified questionnaire
                # Update error message to indicate remediation was attempted
                state["error"] = (state.get("error") or "") + f"Warning: Some raw indicators were flagged and an attempt was made to add questions. Review updated questionnaire and calculation map."
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
    Writes the generated (and potentially LLM-modified) raw indicators and decision variables
    and questionnaire questions to your Supabase tables. This function now performs
    upsert operations (update if exists, insert if new) for individual records.
    Also saves the prompt, title, and project_id to the 'prompts' table.
    """
    print("\n---WRITING TO SUPABASE---")
    state["error"] = state.get("error", "")

    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json"
    }
    error_occurred = False

    # --- Write Raw Indicators ---
    raw_indicators_to_write = state.get("raw_indicators")
    if raw_indicators_to_write:
        print(f"Attempting to upsert {len(raw_indicators_to_write)} raw indicators...")
        for var in raw_indicators_to_write:
            var["project_id"] = state.get("project_id")
            if not _upsert_single_item("raw_indicators", var, headers): 
                error_occurred = True
    else:
        print("No raw indicators found in state to write to Supabase.")

    # --- Write Decision Variables ---
    decision_vars_to_write = state.get("decision_variables")
    if decision_vars_to_write:
        print(f"Attempting to upsert {len(decision_vars_to_write)} decision variables...")
        for var in decision_vars_to_write:
            var["project_id"] = state.get("project_id")
            if not _upsert_single_item("decision_variables", var, headers):
                error_occurred = True
    else:
        print("No decision variables found in state to write to Supabase.")

    # --- Write Questionnaire Questions to 'questions' table ---
    questionnaire_data = state.get("questionnaire")
    raw_indicators_map = {ri['var_name']: ri for ri in (state.get("raw_indicators") or [])}

    questions_to_supabase = []
    if questionnaire_data and questionnaire_data.get("sections"):
        for section in questionnaire_data["sections"]:
            for q_type_list, is_core_q_flag in [
                (section.get('core_questions', []), True),
                (section.get('conditional_questions', []), False)
            ]:
                for question in q_type_list:
                    is_q_mandatory_in_db = section.get('is_mandatory', False)
                    impacted_ri_details = []
                    for ri_var_name in question.get('raw_indicators', []):
                        ri_detail = raw_indicators_map.get(ri_var_name)
                        if ri_detail:
                            impacted_ri_details.append({
                                "id": ri_detail["id"],
                                "name": ri_detail["name"]
                            })
                        else:
                            print(f"Warning: Raw indicator '{ri_var_name}' not found for question '{question.get('variable_name')}' during Supabase write.")

                    question_entry = {
                        "id": question["id"],
                        "project_id": state.get("project_id"),
                        "section_number": section["order"],
                        "question_name": question["text"],
                        "section_name": section["title"],
                        "section_description": section["description"],
                        "is_mandatory": is_q_mandatory_in_db,
                        "section_triggering_criteria": section.get("triggering_criteria"),
                        "question_var_name": question["variable_name"],
                        "impacted_raw_indicators": json.dumps(impacted_ri_details),
                        "question_triggering_criteria": question.get("triggering_criteria"),
                        "is_conditional": question.get("is_conditional", False),
                        "formula": question.get("formula")
                    }
                    questions_to_supabase.append(question_entry)
        print(f"Attempting to upsert {len(questions_to_supabase)} questions...")
        for q_entry in questions_to_supabase:
            if not _upsert_single_item("questions", q_entry, headers):
                error_occurred = True
    else:
        print("No questionnaire sections found in state to write to Supabase 'questions' table.")

    # --- Write Prompt, Title, and Project ID to 'prompts' table ---
    prompt_entry = {
        "id": state.get("project_id"),  # Use project_id as the unique id
        "project_id": state.get("project_id"),
        "prompt": state.get("prompt"),
        "title": state.get("questionnaire_title", "")
    }
    print(f"Upserting prompt entry to 'prompts' table: {prompt_entry}")
    if not _upsert_single_item("prompts", prompt_entry, headers):
        error_occurred = True

    if error_occurred:
        pass
    else:
        state["error"] = None

    return state

# --- NEW: Dependency Analysis Functions ---

def analyze_variable_dependencies(raw_indicators: List[Dict], decision_variables: List[Dict]) -> Dict[str, Any]:
    """
    Analyzes dependencies between raw indicators and decision variables.
    Returns a comprehensive dependency graph with impact analysis.
    """
    print("---ANALYZING VARIABLE DEPENDENCIES---")
    
    # Extract raw indicator names
    raw_indicator_names = [ri['var_name'] for ri in raw_indicators]
    
    # Analyze decision variable dependencies
    dependency_info_list = []
    breaking_changes = []
    enabling_changes = []
    required_updates = []
    
    for dv in decision_variables:
        formula = dv.get('formula', '')
        var_name = dv.get('var_name', '')
        
        # Parse formula to find raw indicator references
        dependencies = parse_formula_dependencies(formula, raw_indicator_names)
        
        # Determine impact level
        impact_level = determine_impact_level(dependencies, formula)
        
        dependency_info = {
            "variable_name": var_name,
            "depends_on": dependencies,
            "formula": formula,
            "impact_level": impact_level
        }
        dependency_info_list.append(dependency_info)
        
        # Check for potential issues
        if not dependencies and formula.strip():
            breaking_changes.append(var_name)
        elif len(dependencies) == 1 and impact_level == 'critical':
            required_updates.append(var_name)
    
    # Find orphaned raw indicators (not used by any decision variable)
    used_raw_indicators = set()
    for dep_info in dependency_info_list:
        used_raw_indicators.update(dep_info['depends_on'])
    
    orphaned_variables = [ri for ri in raw_indicator_names if ri not in used_raw_indicators]
    
    # Identify potential new decision variables
    enabling_changes = identify_potential_new_decision_variables(raw_indicators, decision_variables)
    
    impact_analysis = {
        "breaking_changes": breaking_changes,
        "enabling_changes": enabling_changes,
        "required_updates": required_updates,
        "orphaned_variables": orphaned_variables
    }
    
    dependency_graph = {
        "raw_indicators": raw_indicator_names,
        "decision_variables": dependency_info_list,
        "impact_analysis": impact_analysis
    }
    
    print(f"Dependency analysis complete. Found {len(dependency_info_list)} decision variables with dependencies.")
    return dependency_graph

def parse_formula_dependencies(formula: str, raw_indicator_names: List[str]) -> List[str]:
    """
    Parses a JavaScript formula to extract raw indicator dependencies.
    """
    if not formula or not formula.strip():
        return []
    
    dependencies = []
    
    # Look for exact matches with raw indicator names
    for ri_name in raw_indicator_names:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(ri_name) + r'\b'
        if re.search(pattern, formula):
            dependencies.append(ri_name)
    
    # Also look for common patterns like q_variable_name
    q_pattern = r'\bq_([a-zA-Z_][a-zA-Z0-9_]*)\b'
    q_matches = re.findall(q_pattern, formula)
    for match in q_matches:
        if match in raw_indicator_names:
            dependencies.append(match)
    
    return list(set(dependencies))  # Remove duplicates

def determine_impact_level(dependencies: List[str], formula: str) -> str:
    """
    Determines the impact level of dependencies on a decision variable.
    """
    if not dependencies:
        return 'critical' if formula.strip() else 'low'
    
    if len(dependencies) == 1:
        # Check if the formula heavily relies on this single dependency
        if re.search(r'\b(return|if|while|for)\b', formula):
            return 'critical'
        else:
            return 'moderate'
    else:
        return 'moderate'

def identify_potential_new_decision_variables(raw_indicators: List[Dict], decision_variables: List[Dict]) -> List[str]:
    """
    Identifies potential new decision variables that could be created.
    """
    existing_dv_names = {dv['var_name'] for dv in decision_variables}
    potential_new = []
    
    # Look for patterns in raw indicators that suggest new decision variables
    ri_names = [ri['var_name'] for ri in raw_indicators]
    
    # Check for expense-related indicators that could form total expenses
    expense_indicators = [ri for ri in ri_names if 'expense' in ri.lower() or 'cost' in ri.lower()]
    if len(expense_indicators) > 1 and 'total_expenses' not in existing_dv_names:
        potential_new.append('total_expenses')
    
    # Check for income-related indicators that could form total income
    income_indicators = [ri for ri in ri_names if 'income' in ri.lower() or 'revenue' in ri.lower() or 'sales' in ri.lower()]
    if len(income_indicators) > 1 and 'total_income' not in existing_dv_names:
        potential_new.append('total_income')
    
    # Check for ratio indicators
    if any('income' in ri.lower() for ri in ri_names) and any('expense' in ri.lower() for ri in ri_names):
        if 'income_expense_ratio' not in existing_dv_names:
            potential_new.append('income_expense_ratio')
    
    return potential_new

# --- NEW: Intelligent Variable Synchronization Functions ---

def analyze_variable_dependencies_node(state: GraphState) -> GraphState:
    """
    Langraph node for analyzing variable dependencies.
    """
    print("---ANALYZING VARIABLE DEPENDENCIES NODE---")
    
    raw_indicators = state.get("raw_indicators", []) or []
    decision_variables = state.get("decision_variables", []) or []
    
    if not raw_indicators and not decision_variables:
        print("No variables to analyze dependencies for.")
        return state
    
    try:
        dependency_graph = analyze_variable_dependencies(raw_indicators, decision_variables)
        state["dependency_graph"] = cast(DependencyGraph, dependency_graph)  # type: ignore
        print("Dependency analysis completed and stored in state.")
    except Exception as e:
        state["error"] = (state.get("error") or "") + f"Error analyzing dependencies: {e}"
        print(f"Error analyzing dependencies: {e}")
    
    return state

def synchronize_variables(state: GraphState) -> GraphState:
    """
    Post-modification synchronization to ensure consistency.
    """
    print("---SYNCHRONIZING VARIABLES---")
    
    raw_indicators = state.get("raw_indicators", []) or []
    decision_variables = state.get("decision_variables", []) or []
    
    if not raw_indicators and not decision_variables:
        return state
    
    try:
        # Re-analyze dependencies after modifications
        dependency_graph = analyze_variable_dependencies(raw_indicators, decision_variables)
        state["dependency_graph"] = cast(DependencyGraph, dependency_graph)  # type: ignore
        
        # Check for consistency issues
        impact_analysis = dependency_graph.get("impact_analysis", {})
        breaking_changes = impact_analysis.get("breaking_changes", [])
        orphaned_variables = impact_analysis.get("orphaned_variables", [])
        
        if breaking_changes or orphaned_variables:
            print(f"Warning: Found {len(breaking_changes)} breaking changes and {len(orphaned_variables)} orphaned variables.")
            state["error"] = (state.get("error") or "") + f"Warning: {len(breaking_changes)} breaking changes and {len(orphaned_variables)} orphaned variables detected after synchronization."
        
        print("Variable synchronization completed.")
        
    except Exception as e:
        state["error"] = (state.get("error") or "") + f"Error in variable synchronization: {e}"
        print(f"Error in variable synchronization: {e}")
    
    return state

# --- Supabase fetch utility ---
def fetch_supabase_tables() -> Dict[str, Any]:
    """
    Fetch all rows from the three Supabase tables: raw_indicators, decision_variables, questionnaire.
    Returns a dict with keys 'raw_indicators', 'decision_variables', 'questionnaire'.
    """
    import requests
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json"
    }
    base_url = SUPABASE_URL
    result = {}
    for table in ["raw_indicators", "decision_variables", "questions", "prompts"]:
        url = f"{base_url}/{table}?select=*"
        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            result[table] = resp.json()
        except Exception as e:
            print(f"Error fetching {table} from Supabase: {e}")
            result[table] = []
    return result

def export_sections_for_card_generator(state):
    questionnaire = state.get("questionnaire")
    if not questionnaire:
        prompt = "No questionnaire found in state."
        state["card_generator_prompt"] = prompt
        return prompt
    title = state.get("prompt", "Business Questionnaire")
    sections = []
    for section in questionnaire.get("sections", []):
        section_obj = {
            "section_title": section.get("title"),
            "section_order": section.get("order"),
            "section_description": section.get("description", ""),
            "core_questions": section.get("core_questions", []),
            "conditional_questions": section.get("conditional_questions", [])
        }
        sections.append(section_obj)
    sections_json = json.dumps(sections, indent=2)
    prompt = EXPORT_SECTION_CARDS_PROMPT.format(title=title, sections_json=sections_json)
    state["card_generator_prompt"] = prompt

    # --- LLM CALL ---
    openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure your key is set in env vars
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.3
    )
    llm_output = response.choices[0].message.content
    state["card_generator_llm_output"] = llm_output
    return llm_output