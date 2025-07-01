from typing import List, Dict, Optional, Any, TypedDict

# --- Pydantic/TypedDict Schemas for Structured Output ---
class VariableSchema(TypedDict):
    """Schema for individual raw indicators or decision variables."""
    id: str
    name: str
    var_name: str
    priority: int
    description: str
    priority_rationale: str  # New field: explanation for why this priority was assigned
    formula: Optional[str] # Null for raw indicators, JS string for decision variables
    type: str # New field: text, int, float, dropdown, etc.
    value: Optional[str] # New field: initially null, for user input later
    project_id: Optional[str] # New: Added for project differentiation

# --- Dependency Analysis Schemas ---
class DependencyInfo(TypedDict):
    """Information about dependencies between variables."""
    variable_name: str
    depends_on: List[str]  # List of raw indicator var_names this decision variable depends on
    formula: str
    impact_level: str  # 'critical', 'moderate', 'low'

class ImpactAnalysis(TypedDict):
    """Analysis of how a modification will impact other variables."""
    breaking_changes: List[str]  # Decision variables that will break
    enabling_changes: List[str]  # New decision variables that could be created
    required_updates: List[str]  # Variables that need formula updates
    orphaned_variables: List[str]  # Raw indicators no longer used by any decision variable

class DependencyGraph(TypedDict):
    """Complete dependency mapping between variables."""
    raw_indicators: List[str]  # All raw indicator var_names
    decision_variables: List[DependencyInfo]  # Decision variables with their dependencies
    impact_analysis: ImpactAnalysis  # Current impact state

class IntelligentModificationRequest(TypedDict):
    """Enhanced modification request with dependency awareness."""
    primary_modifications: str  # User's original modification request
    dependency_analysis: DependencyGraph  # Pre-computed dependency information
    auto_sync_enabled: bool  # Whether to auto-sync related variables
    business_context: str  # Domain context for intelligent decisions
    modification_type: str  # 'raw_indicators', 'decision_variables', or 'both'

class SynchronizationPlan(TypedDict):
    """Plan for synchronizing variables after modifications."""
    primary_changes: List[Dict[str, Any]]  # User's requested changes
    compensatory_changes: List[Dict[str, Any]]  # Changes needed to maintain consistency
    removed_variables: List[str]  # Variables to be removed
    updated_formulas: Dict[str, str]  # Formula updates needed
    new_variables: List[Dict[str, Any]]  # New variables to be added

class RawIndicatorsOutput(TypedDict):
    """Schema for the LLM's output when generating raw indicators."""
    raw_indicators: List[VariableSchema]

class DecisionVariablesOutput(TypedDict):
    """Schema for the LLM's output when generating decision variables."""
    decision_variables: List[VariableSchema]

class IntelligentVariableModificationsOutput(TypedDict):
    """Schema for the LLM's output when making intelligent variable modifications."""
    primary_modifications: Dict[str, List[Dict[str, Any]]]  # Changes to primary variable type
    compensatory_modifications: Dict[str, List[Dict[str, Any]]]  # Changes to other variable type
    removed_variables: List[str]  # Variables to be removed
    updated_formulas: Dict[str, str]  # Formula updates
    new_variables: List[VariableSchema]  # New variables to be added
    reasoning: str  # LLM's reasoning for the changes

class Question(TypedDict):
    """Schema for a single survey question."""
    id: str # New field: Unique ID for the question itself
    text: str
    type: str # 'text', 'integer', 'float', 'boolean', 'dropdown'
    variable_name: str # The name of the variable that stores the answer to this question
    triggering_criteria: Optional[str] # JS function string, e.g., "return question_variable_name_from_prev_q === 'Yes';"
    raw_indicators: List[str] # List of var_name of raw indicators this question helps capture
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
    raw_indicator_calculation: Optional[Dict[str, str]] # Maps raw_indicator_var_name to JS formula


# --- Modification Schemas ---
class VariableModification(TypedDict):
    """Base schema for variable modifications."""
    id: str
    # Other fields are optional as they are for partial updates

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
        modification_history: List of modification requests made.
        raw_indicators: List of dictionaries for raw indicators.
        decision_variables: List[Dict] for decision variables.
        questionnaire: Optional[Dict] # New: Stores the generated survey structure
        error: Any error messages encountered during node execution.
        project_id: Optional[str] # New: Unique ID for the current project workflow
        dependency_graph: Optional[DependencyGraph] # New: Stores dependency analysis results
        modification_reasoning: Optional[str] # New: Stores LLM reasoning for modifications
        status: Optional[str] # New: Tracks the current status of the workflow
        needs_review: Optional[bool] # New: Indicates if modifications need review
    """
    prompt: str
    modification_prompt: Optional[str]
    modification_history: Optional[List[str]]
    raw_indicators: Optional[List[Dict]]
    decision_variables: Optional[List[Dict]]
    questionnaire: Optional[Dict]
    error: Optional[str]
    project_id: Optional[str]
    dependency_graph: Optional[DependencyGraph]
    modification_reasoning: Optional[str]
    status: Optional[str]
    needs_review: Optional[bool]

# For Remediation Output (used by analyze_questionnaire_impact)
class RemediationOutput(TypedDict):
    added_questions: Optional[List[Question]]
    updated_raw_indicator_calculation: Optional[Dict[str, str]]

# For JS Refinement Output (used by _refine_js_expression)
class StringOutput(TypedDict):
    expression: str