from typing import List, Dict, Optional, Any, TypedDict

# --- Pydantic/TypedDict Schemas for Structured Output ---
class VariableSchema(TypedDict):
    """Schema for individual assessment or computational variables."""
    id: str
    name: str
    var_name: str
    priority: int
    description: str
    formula: Optional[str] # Null for assessment, JS string for computational
    type: str # New field: text, int, float, dropdown, etc.
    value: Optional[str] # New field: initially null, for user input later
    project_id: Optional[str] # New: Added for project differentiation

class AssessmentVariablesOutput(TypedDict):
    """Schema for the LLM's output when generating assessment variables."""
    assessment_variables: List[VariableSchema]

class ComputationalVariablesOutput(TypedDict):
    """Schema for the LLM's output when generating computational variables."""
    computational_variables: List[VariableSchema]

class Question(TypedDict):
    """Schema for a single survey question."""
    id: str # New field: Unique ID for the question itself
    text: str
    type: str # 'text', 'integer', 'float', 'boolean', 'dropdown'
    variable_name: str # The name of the variable that stores the answer to this question
    triggering_criteria: Optional[str] # JS function string, e.g., "return question_variable_name_from_prev_q === 'Yes';"
    assessment_variables: List[str] # List of var_name of assessment variables this question helps capture
    formula: Optional[str] # JS function string, e.g., "return parseFloat(q_daily_sales);"
    is_conditional: Optional[bool] # New: Indicates if the question is conditional
    project_id: Optional[str] # New: Added for project differentiation

class Section(TypedDict):
    """Schema for a section within the survey questionnaire."""
    title: str
    description: str
    order: int
    is_mandatory: bool
    rationale: str
    core_questions: List[Question]
    conditional_questions: List[Question]
    triggering_criteria: Optional[str] # JS function string for section visibility
    data_validation: str # JS function string for section-level validation
    project_id: Optional[str] # New: Added for project differentiation

class QuestionnaireOutput(TypedDict):
    """Schema for the LLM's output when generating the full questionnaire."""
    sections: List[Section]
    assessment_variable_calculation: Optional[Dict[str, str]] # Maps assessment_var_name to JS formula


# --- Modification Schemas ---
class VariableModification(TypedDict):
    """Base schema for variable modifications."""
    id: str
    # Other fields are optional as they are for partial updates

class VariableModificationsOutput(TypedDict):
    """Schema for the LLM's output when modifying variables."""
    added_assessment_variables: Optional[List[VariableSchema]]
    updated_assessment_variables: Optional[List[VariableModification]]
    removed_assessment_variable_ids: Optional[List[str]]
    added_computational_variables: Optional[List[VariableSchema]]
    updated_computational_variables: Optional[List[VariableModification]]
    removed_computational_variable_ids: Optional[List[str]]

class QuestionModification(TypedDict):
    """Schema for a question modification within a section."""
    id: str
    # Other fields are optional for partial updates

class SectionModification(TypedDict):
    """Schema for a section modification."""
    order: int
    # Other fields are optional for partial updates

class AddedQuestion(TypedDict):
    """Schema for adding a new question, specifying its section and type."""
    section_order: int
    is_core: bool
    question: Question # The full question object

class QuestionnaireModificationsOutput(TypedDict):
    """Schema for the LLM's output when modifying the questionnaire."""
    added_sections: Optional[List[Section]]
    updated_sections: Optional[List[SectionModification]]
    removed_section_orders: Optional[List[int]]
    added_questions: Optional[List[AddedQuestion]] # Dict containing section_order and list of questions to add
    updated_questions: Optional[List[QuestionModification]] # Dict containing question_variable_name and fields to update
    removed_question_variable_names: Optional[List[str]] # List of question_variable_names to remove

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        prompt: User's initial prompt (e.g., "Assess income for a street food vendor").
        modification_prompt: User's prompt for modifying variables or questionnaire.
        assessment_variables: List of dictionaries for assessment variables.
        computational_variables: List[Dict] for computational variables.
        questionnaire: Optional[Dict] # New: Stores the generated survey structure
        error: Any error messages encountered during node execution.
        project_id: Optional[str] # New: Unique ID for the current project workflow
    """
    prompt: str
    modification_prompt: Optional[str]
    assessment_variables: Optional[List[Dict]]
    computational_variables: Optional[List[Dict]]
    questionnaire: Optional[Dict]
    error: Optional[str]
    project_id: Optional[str] # Added project_id to GraphState

# For Remediation Output (used by analyze_questionnaire_impact)
class RemediationOutput(TypedDict):
    added_questions: Optional[List[Question]]
    updated_assessment_variable_calculation: Optional[Dict[str, str]]

# For JS Refinement Output (used by _refine_js_expression)
class StringOutput(TypedDict):
    expression: str
