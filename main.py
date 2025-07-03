import os
import json
import uuid
from typing import Dict, Any, Optional, List, cast
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import nodes from nodes.py
from nodes import (
    generate_variables,
    modify_variables_intelligent,  # Updated to use intelligent modification
    analyze_variable_dependencies_node,  # New dependency analysis node
    synchronize_variables,  # New synchronization node
    generate_questionnaire,
    modify_questionnaire_llm,
    _refine_js_expression,
    # Additional imports for testing
    analyze_variable_dependencies,
    parse_formula_dependencies,
    determine_impact_level,
    determine_modification_type,
    _apply_default_variable_properties,
    write_to_supabase,
    analyze_questionnaire_impact,  # Add this import
    export_sections_for_card_generator
)

# Import schemas
from schemas.schemas import GraphState

load_dotenv()  # Load environment variables from .env file

# Initialize the Language Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Initialize checkpoint memory
memory = MemorySaver()

def create_workflow() -> StateGraph:
    """
    Creates the main workflow graph for the income assessment system.
    """
    # Create the workflow graph
    workflow = StateGraph(GraphState)
    
    # Add nodes to the graph
    workflow.add_node("generate_variables", generate_variables)
    workflow.add_node("analyze_dependencies", analyze_variable_dependencies_node)
    workflow.add_node("modify_variables_intelligent", modify_variables_intelligent)
    workflow.add_node("synchronize_variables", synchronize_variables)
    workflow.add_node("generate_questionnaire", generate_questionnaire)
    workflow.add_node("modify_questionnaire", modify_questionnaire_llm)
    workflow.add_node("analyze_questionnaire_impact", analyze_questionnaire_impact)
    
    # Define the main workflow edges
    workflow.set_entry_point("generate_variables")
    
    # Always go to dependency analysis after generating variables
    workflow.add_edge("generate_variables", "analyze_dependencies")
    
    # After dependency analysis, conditionally go to modification or questionnaire
    workflow.add_conditional_edges(
        "analyze_dependencies",
        lambda state: "modify_variables_intelligent" if state.get("modification_prompt") else "generate_questionnaire",
        {
            "modify_variables_intelligent": "modify_variables_intelligent",
            "generate_questionnaire": "generate_questionnaire"
        }
    )
    # After modification, always go back to dependency analysis (to allow iterative modifications)
    workflow.add_edge("modify_variables_intelligent", "analyze_dependencies")
    
    # After questionnaire generation, go to impact analysis
    workflow.add_edge("generate_questionnaire", "analyze_questionnaire_impact")
    
    # After questionnaire modification, go to impact analysis
    workflow.add_edge("modify_questionnaire", "analyze_questionnaire_impact")
    
    # After impact analysis, conditionally end or go back to modification
    workflow.add_conditional_edges(
        "analyze_questionnaire_impact",
        lambda state: "modify_questionnaire" if state.get("modification_prompt") else END,
        {
            "modify_questionnaire": "modify_questionnaire",
            END: END
        }
    )
    
    return workflow

def run_workflow(
    prompt: str,
    modification_prompt: Optional[str] = None,
    project_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs the complete workflow for income assessment.
    
    Args:
        prompt: The main prompt for generating variables and questionnaire
        modification_prompt: Optional prompt for modifying existing variables
        project_id: Optional project ID for tracking
    
    Returns:
        Dictionary containing the workflow results
    """
    # Generate project ID if not provided
    if not project_id:
        project_id = str(uuid.uuid4())
    
    # Create the workflow
    app = create_workflow().compile(checkpointer=memory)
    
    # Prepare initial state
    initial_state = {
        "prompt": prompt,
        "modification_prompt": modification_prompt,
        "project_id": project_id,
        "raw_indicators": None,
        "decision_variables": None,
        "questionnaire": None,
        "error": None,
        "dependency_graph": None,
        "modification_reasoning": None
    }
    
    # Run the workflow
    try:
        if modification_prompt:
            # If modification prompt is provided, start with modification
            print("Starting workflow with variable modification...")
            result = app.invoke(initial_state, config={"configurable": {"thread_id": project_id}})
            
            # Continue with questionnaire generation if variables exist
            if result.get("raw_indicators") or result.get("decision_variables"):
                print("Continuing with questionnaire generation...")
                result = app.invoke(result, config={"configurable": {"thread_id": project_id}})
        else:
            # Standard flow: generate variables and questionnaire
            print("Starting standard workflow...")
            result = app.invoke(initial_state, config={"configurable": {"thread_id": project_id}})
        
        return {
            "success": True,
            "project_id": project_id,
            "raw_indicators": result.get("raw_indicators", []),
            "decision_variables": result.get("decision_variables", []),
            "questionnaire": result.get("questionnaire"),
            "dependency_graph": result.get("dependency_graph"),
            "modification_reasoning": result.get("modification_reasoning"),
            "error": result.get("error")
        }
        
    except Exception as e:
        return {
            "success": False,
            "project_id": project_id,
            "error": f"Workflow execution failed: {str(e)}",
            "raw_indicators": [],
            "decision_variables": [],
            "questionnaire": None,
            "dependency_graph": None,
            "modification_reasoning": None
        }

def run_variable_modification_only(
    modification_prompt: str,
    existing_raw_indicators: List[Dict],
    existing_decision_variables: List[Dict],
    project_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs only the variable modification workflow.
    """
    print("[DEBUG] Entering run_variable_modification_only...")
    print(f"[DEBUG] modification_prompt: {modification_prompt}")
    print(f"[DEBUG] existing_raw_indicators: {json.dumps(existing_raw_indicators, indent=2)}")
    print(f"[DEBUG] existing_decision_variables: {json.dumps(existing_decision_variables, indent=2)}")
    if not project_id:
        project_id = str(uuid.uuid4())
    
    app = create_workflow().compile(checkpointer=memory)
    
    # Prepare state with existing variables
    initial_state = {
        "prompt": "Variable modification only",
        "modification_prompt": modification_prompt,
        "project_id": project_id,
        "raw_indicators": existing_raw_indicators,
        "decision_variables": existing_decision_variables,
        "questionnaire": None,
        "error": None,
        "dependency_graph": None,
        "modification_reasoning": None
    }
    print(f"[DEBUG] Initial state: {json.dumps(initial_state, indent=2, default=str)}")
    try:
        print("Running variable modification workflow...")
        result = app.invoke(initial_state, config={"configurable": {"thread_id": project_id}})
        print(f"[DEBUG] State after workflow invoke: {json.dumps(result, indent=2, default=str)}")
        return {
            "success": True,
            "project_id": project_id,
            "raw_indicators": result.get("raw_indicators", []),
            "decision_variables": result.get("decision_variables", []),
            "dependency_graph": result.get("dependency_graph"),
            "modification_reasoning": result.get("modification_reasoning"),
            "error": result.get("error")
        }
        
    except Exception as e:
        return {
            "success": False,
            "project_id": project_id,
            "error": f"Workflow execution failed: {str(e)}",
            "raw_indicators": [],
            "decision_variables": [],
            "dependency_graph": None,
            "modification_reasoning": None
        }

# --- NEW: Run only variable generation and dependency analysis (no questionnaire) ---
def run_generate_variables_only(prompt: str, project_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs only the variable generation and dependency analysis steps.
    Returns a dict with raw_indicators, decision_variables, dependency_graph, etc.
    """
    if not project_id:
        project_id = str(uuid.uuid4())
    initial_state = {
        "prompt": prompt,
        "modification_prompt": None,
        "project_id": project_id,
        "raw_indicators": None,
        "decision_variables": None,
        "questionnaire": None,
        "error": None,
        "dependency_graph": None,
        "modification_reasoning": None
    }
    try:
        # Step 1: generate_variables
        state1 = generate_variables(cast(GraphState, initial_state))
        # Step 2: analyze_dependencies
        state2 = analyze_variable_dependencies_node(cast(GraphState, state1))
        return {
            "success": True,
            "project_id": project_id,
            "raw_indicators": state2.get("raw_indicators", []),
            "decision_variables": state2.get("decision_variables", []),
            "dependency_graph": state2.get("dependency_graph"),
            "error": state2.get("error")
        }
    except Exception as e:
        return {
            "success": False,
            "project_id": project_id,
            "error": f"Variable generation failed: {str(e)}",
            "raw_indicators": [],
            "decision_variables": [],
            "dependency_graph": None
        }

# --- NEW: Run only questionnaire generation (from existing variables) ---
def run_generate_questionnaire_only(prompt: str, raw_indicators, decision_variables, project_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs only the questionnaire generation step, given variables.
    """
    if not project_id:
        project_id = str(uuid.uuid4())
    state = {
        "prompt": prompt,
        "modification_prompt": None,
        "project_id": project_id,
        "raw_indicators": raw_indicators,
        "decision_variables": decision_variables,
        "questionnaire": None,
        "error": None,
        "dependency_graph": None,
        "modification_reasoning": None
    }
    try:
        # Step: generate_questionnaire
        state2 = generate_questionnaire(cast(GraphState, state))
        return {
            "success": True,
            "project_id": project_id,
            "questionnaire": state2.get("questionnaire"),
            "error": state2.get("error")
        }
    except Exception as e:
        return {
            "success": False,
            "project_id": project_id,
            "error": f"Questionnaire generation failed: {str(e)}",
            "questionnaire": None
        }

# --- UPDATED: Interactive test for the complete workflow (multi-modification, dependency-visualizing, with optional Supabase save and auto-fix) ---
def test_complete_workflow():
    """Interactive test for the complete workflow with iterative modifications and questionnaire modification."""
    print("\n=== TESTING COMPLETE WORKFLOW (Strict User Flow) ===")
    prompt = input("Enter your assessment prompt (e.g., 'Assess income for a street food vendor'): ").strip()
    project_id = input("Enter project ID (optional, press Enter for auto-generated): ").strip() or None
    if not project_id:
        project_id = str(uuid.uuid4())

    # Initial workflow state
    state = {
        "prompt": prompt,
        "modification_prompt": None,
        "project_id": project_id,
        "raw_indicators": None,
        "decision_variables": None,
        "questionnaire": None,
        "error": None,
        "dependency_graph": None,
        "modification_reasoning": None
    }

    # Step 1: Generate variables
    state = generate_variables(cast(GraphState, state))

    # Step 2: Analyze dependencies
    state = analyze_variable_dependencies_node(cast(GraphState, state))

    # Step 3: Iterative variable modification loop
    while True:
        print("\nType 'm' to modify variables, 'f' to finalize:", end=' ')
        user_action = input().strip().lower()
        if user_action == 'm':
            mod_prompt = input("Enter variable modification prompt: ").strip()
            if not mod_prompt:
                print("No modification prompt entered. Returning to menu.")
                continue
            state["modification_prompt"] = mod_prompt
            # Run modification node
            state = modify_variables_intelligent(cast(GraphState, state))
            # Clear the modification prompt to avoid infinite loop
            state["modification_prompt"] = None
            # Re-analyze dependencies after modification
            state = analyze_variable_dependencies_node(cast(GraphState, state))
        elif user_action == 'f':
            proceed = input("Proceed to questions? (y/n): ").strip().lower()
            if proceed == 'y':
                # Generate questionnaire
                state = generate_questionnaire(cast(GraphState, state))
                print("\n=== Generated Questionnaire ===")
                print(json.dumps(state.get("questionnaire"), indent=2, default=str))
                break
            else:
                print("You can continue modifying variables.")
        else:
            print("Invalid input. Please type 'm' or 'f'.")

    # Step 4: Questionnaire modification loop
    while True:
        action = input("\nType 'm' to modify questions, 'f' to finalize: ").strip().lower()
        if action == 'f':
            break
        elif action == 'm':
            mod_prompt = input("Enter questionnaire modification prompt: ").strip()
            if not mod_prompt:
                print("No modification entered. Please try again.")
                continue
            state["modification_prompt"] = mod_prompt
            state = modify_questionnaire_llm(cast(GraphState, state))
            state["modification_prompt"] = None
            print("\nQuestions after modification:")
            questionnaire = state.get("questionnaire")
            if questionnaire and questionnaire.get("sections"):
                for sidx, section in enumerate(questionnaire["sections"], 1):
                    print(f"Section {sidx}: {section.get('title', 'Untitled')}")
                    for qtype, qlist in [("Core", section.get('core_questions', [])), ("Conditional", section.get('conditional_questions', []))]:
                        for qidx, q in enumerate(qlist, 1):
                            print(f"  {qtype} Q{qidx}: {q.get('text', 'No text')} (var: {q.get('variable_name')})")
            else:
                print("No questionnaire sections found.")
        else:
            print("Invalid input. Please type 'm' or 'f'.")

    # Step 5: Save to Supabase option
    save_supabase = input("\nSave variables and questionnaire to Supabase? (y/n): ").strip().lower()
    if save_supabase == 'y':
        print("\nSaving to Supabase...")
        state_to_save = {
            "prompt": prompt,
            "modification_prompt": None,
            "project_id": project_id,
            "raw_indicators": state.get("raw_indicators"),
            "decision_variables": state.get("decision_variables"),
            "questionnaire": state.get("questionnaire"),
            "error": None,
            "dependency_graph": state.get("dependency_graph"),
            "modification_reasoning": state.get("modification_reasoning")
        }
        saved_state = write_to_supabase(cast(GraphState, state_to_save))
        if saved_state.get('error'):
            print(f"❌ Error saving to Supabase: {saved_state['error']}")
        else:
            # Export section-by-section card generator prompt using LLM
            card_prompt = export_sections_for_card_generator(saved_state)
            print("\n=== Card Generator LLM Output (copy and paste into your card generator agent) ===\n")
            print(card_prompt)
    else:
        print("Skipped saving to Supabase.")

def test_dependency_analysis():
    """Interactive test for dependency analysis."""
    print("\n=== TESTING DEPENDENCY ANALYSIS ===")
    
    # Get sample data or use defaults
    use_sample = input("Use sample data? (y/n): ").strip().lower()
    
    if use_sample == 'y':
        raw_indicators = [
            {
                "id": "ri1",
                "name": "Daily Sales",
                "var_name": "daily_sales",
                "priority": 1,
                "description": "Average daily sales amount",
                "priority_rationale": "Critical for income assessment",
                "formula": None,
                "type": "float",
                "value": None,
                "project_id": "test_project"
            },
            {
                "id": "ri2", 
                "name": "Operating Days",
                "var_name": "operating_days",
                "priority": 2,
                "description": "Number of days business operates per week",
                "priority_rationale": "Important for weekly calculation",
                "formula": None,
                "type": "integer",
                "value": None,
                "project_id": "test_project"
            }
        ]
        
        decision_variables = [
            {
                "id": "dv1",
                "name": "Weekly Revenue",
                "var_name": "weekly_revenue",
                "priority": 1,
                "description": "Calculated weekly revenue",
                "priority_rationale": "Key metric for income assessment",
                "formula": "return daily_sales * operating_days;",
                "type": "float",
                "value": None,
                "project_id": "test_project"
            }
        ]
    else:
        print("Please provide your own data (for now, using sample data)")
        raw_indicators = []
        decision_variables = []
    
    print(f"\nAnalyzing dependencies for:")
    print(f"Raw Indicators: {len(raw_indicators)}")
    print(f"Decision Variables: {len(decision_variables)}")
    
    try:
        dependency_graph = analyze_variable_dependencies(raw_indicators, decision_variables)
        
        print("\n✅ Dependency Analysis Results:")
        print(f"Raw Indicators: {dependency_graph['raw_indicators']}")
        print(f"Decision Variables: {len(dependency_graph['decision_variables'])}")
        
        impact_analysis = dependency_graph['impact_analysis']
        print(f"\nImpact Analysis:")
        print(f"  Breaking Changes: {impact_analysis['breaking_changes']}")
        print(f"  Enabling Changes: {impact_analysis['enabling_changes']}")
        print(f"  Required Updates: {impact_analysis['required_updates']}")
        print(f"  Orphaned Variables: {impact_analysis['orphaned_variables']}")
        
        show_full = input("\nShow full dependency graph? (y/n): ").strip().lower()
        if show_full == 'y':
            print("\n=== FULL DEPENDENCY GRAPH ===")
            print(json.dumps(dependency_graph, indent=2, default=str))
            
    except Exception as e:
        print(f"❌ Dependency analysis failed: {e}")
        import traceback
        traceback.print_exc()

def test_intelligent_modification():
    """Interactive test for intelligent variable modification."""
    print("\n=== TESTING INTELLIGENT VARIABLE MODIFICATION ===")
    
    # Get modification prompt
    modification_prompt = input("Enter modification prompt (e.g., 'Remove Operating Days and update related variables'): ").strip()
    if not modification_prompt:
        modification_prompt = "Remove the 'Operating Days' raw indicator and update related decision variables"
    
    # Use sample data for testing
    existing_raw_indicators = [
        {
            "id": "ri1",
            "name": "Daily Sales",
            "var_name": "daily_sales",
            "priority": 1,
            "description": "Average daily sales amount",
            "priority_rationale": "Critical for income assessment",
            "formula": None,
            "type": "float",
            "value": None,
            "project_id": "test_project"
        },
        {
            "id": "ri2", 
            "name": "Operating Days",
            "var_name": "operating_days",
            "priority": 2,
            "description": "Number of days business operates per week",
            "priority_rationale": "Important for weekly calculation",
            "formula": None,
            "type": "integer",
            "value": None,
            "project_id": "test_project"
        }
    ]
    
    existing_decision_variables = [
        {
            "id": "dv1",
            "name": "Weekly Revenue",
            "var_name": "weekly_revenue",
            "priority": 1,
            "description": "Calculated weekly revenue",
            "priority_rationale": "Key metric for income assessment",
            "formula": "return daily_sales * operating_days;",
            "type": "float",
            "value": None,
            "project_id": "test_project"
        }
    ]
    
    project_id = input("Enter project ID (optional, press Enter for auto-generated): ").strip()
    if not project_id:
        project_id = None
    
    print(f"\nRunning intelligent modification with:")
    print(f"Modification: {modification_prompt}")
    print(f"Existing Raw Indicators: {len(existing_raw_indicators)}")
    print(f"Existing Decision Variables: {len(existing_decision_variables)}")
    print(f"Project ID: {project_id or 'Auto-generated'}")
    
    proceed = input("\nProceed? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Modification cancelled.")
        return
    
    try:
        # Run the variable modification workflow
        result = run_variable_modification_only(
            modification_prompt, 
            existing_raw_indicators, 
            existing_decision_variables, 
            project_id
        )
        
        if result["success"]:
            print(f"\n✅ Intelligent modification completed successfully!")
            print(f"Project ID: {result['project_id']}")
            print(f"Updated Raw Indicators: {len(result['raw_indicators'])}")
            print(f"Updated Decision Variables: {len(result['decision_variables'])}")
            if result['modification_reasoning']:
                print(f"📝 Reasoning: {result['modification_reasoning']}")
            if result['dependency_graph']:
                print("✅ Dependency graph updated")
        else:
            print(f"❌ Modification failed: {result['error']}")
        
        # Show detailed results
        show_detailed = input("\nShow detailed results? (y/n): ").strip().lower()
        if show_detailed == 'y':
            print("\n=== DETAILED RESULTS ===")
            print(json.dumps(result, indent=2, default=str))
            
    except Exception as e:
        print(f"❌ Intelligent modification failed: {e}")
        import traceback
        traceback.print_exc()

def test_synchronization():
    """Interactive test for variable synchronization."""
    print("\n=== TESTING VARIABLE SYNCHRONIZATION ===")
    
    # Create test state with inconsistencies
    raw_indicators = [
        {
            "id": "ri1",
            "name": "Daily Sales",
            "var_name": "daily_sales",
            "priority": 1,
            "description": "Average daily sales amount",
            "priority_rationale": "Critical for income assessment",
            "formula": None,
            "type": "float",
            "value": None,
            "project_id": "test_project"
        }
    ]
    
    decision_variables = [
        {
            "id": "dv1",
            "name": "Weekly Revenue",
            "var_name": "weekly_revenue",
            "priority": 1,
            "description": "Calculated weekly revenue",
            "priority_rationale": "Key metric for income assessment",
            "formula": "return daily_sales * operating_days;",  # References non-existent operating_days
            "type": "float",
            "value": None,
            "project_id": "test_project"
        }
    ]
    
    print(f"\nTesting synchronization with inconsistent data:")
    print(f"Raw Indicators: {len(raw_indicators)}")
    print(f"Decision Variables: {len(decision_variables)}")
    print("Note: Decision variable formula references non-existent 'operating_days'")
    
    proceed = input("\nProceed with synchronization? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Synchronization cancelled.")
        return
    
    try:
        state = cast(GraphState, {
            "prompt": "Test income assessment",
            "modification_prompt": None,
            "project_id": "test_project",
            "raw_indicators": raw_indicators,
            "decision_variables": decision_variables,
            "questionnaire": None,
            "error": None,
            "dependency_graph": None,
            "modification_reasoning": None
        })
        
        synchronized_state = synchronize_variables(state)
        
        print("\n✅ Synchronization Results:")
        print(f"Error: {synchronized_state.get('error', 'None')}")
        print(f"Dependency Graph: {'Generated' if synchronized_state.get('dependency_graph') else 'None'}")
        
        dependency_graph = synchronized_state.get('dependency_graph')
        if dependency_graph and isinstance(dependency_graph, dict):
            impact_analysis = dependency_graph.get('impact_analysis', {})
            print(f"\nImpact Analysis:")
            print(f"  Breaking Changes: {impact_analysis.get('breaking_changes', [])}")
            print(f"  Orphaned Variables: {impact_analysis.get('orphaned_variables', [])}")
        
        show_full = input("\nShow full synchronized state? (y/n): ").strip().lower()
        if show_full == 'y':
            print("\n=== FULL SYNCHRONIZED STATE ===")
            print(json.dumps(synchronized_state, indent=2, default=str))
            
    except Exception as e:
        print(f"❌ Synchronization failed: {e}")
        import traceback
        traceback.print_exc()

def test_individual_components():
    """Interactive test for individual components."""
    print("\n=== TESTING INDIVIDUAL COMPONENTS ===")
    
    while True:
        print("\nIndividual Component Tests:")
        print("1. Test Formula Parsing")
        print("2. Test Impact Level Determination")
        print("3. Test Modification Type Detection")
        print("4. Test Variable Property Application")
        print("5. Back to Main Menu")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            test_formula_parsing()
        elif choice == "2":
            test_impact_level_determination()
        elif choice == "3":
            test_modification_type_detection()
        elif choice == "4":
            test_variable_property_application()
        elif choice == "5":
            break
        else:
            print("Invalid choice. Please enter a number between 1-5.")

def test_formula_parsing():
    """Test the formula parsing functionality."""
    print("\n=== TESTING FORMULA PARSING ===")
    
    formula = input("Enter a JavaScript formula to parse (e.g., 'return daily_sales * operating_days;'): ").strip()
    if not formula:
        formula = "return daily_sales * operating_days;"
    
    raw_indicator_names = input("Enter raw indicator names (comma-separated, e.g., 'daily_sales,operating_days'): ").strip()
    if not raw_indicator_names:
        raw_indicator_names = "daily_sales,operating_days,monthly_expenses"
    
    raw_indicator_list = [name.strip() for name in raw_indicator_names.split(",")]
    
    print(f"\nParsing formula: {formula}")
    print(f"Available raw indicators: {raw_indicator_list}")
    
    try:
        dependencies = parse_formula_dependencies(formula, raw_indicator_list)
        
        print(f"\n✅ Parsing Results:")
        print(f"Dependencies found: {dependencies}")
        print(f"Number of dependencies: {len(dependencies)}")
        
    except Exception as e:
        print(f"❌ Formula parsing failed: {e}")
        import traceback
        traceback.print_exc()

def test_impact_level_determination():
    """Test the impact level determination functionality."""
    print("\n=== TESTING IMPACT LEVEL DETERMINATION ===")
    
    dependencies_input = input("Enter dependencies (comma-separated, e.g., 'daily_sales,operating_days'): ").strip()
    if not dependencies_input:
        dependencies_input = "daily_sales"
    
    dependencies = [dep.strip() for dep in dependencies_input.split(",") if dep.strip()]
    
    formula = input("Enter formula (e.g., 'return daily_sales * 7;'): ").strip()
    if not formula:
        formula = "return daily_sales * 7;"
    
    print(f"\nDetermining impact level for:")
    print(f"Dependencies: {dependencies}")
    print(f"Formula: {formula}")
    
    try:
        impact_level = determine_impact_level(dependencies, formula)
        
        print(f"\n✅ Impact Level: {impact_level}")
        print(f"Explanation:")
        if impact_level == "critical":
            print("  - Variable cannot function without this dependency")
        elif impact_level == "moderate":
            print("  - Variable can be adapted or has alternatives")
        else:
            print("  - Variable has minimal dependency on this raw indicator")
        
    except Exception as e:
        print(f"❌ Impact level determination failed: {e}")
        import traceback
        traceback.print_exc()

def test_modification_type_detection():
    """Test the modification type detection functionality."""
    print("\n=== TESTING MODIFICATION TYPE DETECTION ===")
    
    modification_prompt = input("Enter modification prompt: ").strip()
    if not modification_prompt:
        modification_prompt = "Add new raw indicators for seasonal variations"
    
    print(f"\nDetecting modification type for: {modification_prompt}")
    
    try:
        modification_type = determine_modification_type(modification_prompt)
        
        print(f"\n✅ Modification Type: {modification_type}")
        print(f"Explanation:")
        if modification_type == "raw_indicators":
            print("  - Primarily affects raw indicators")
        elif modification_type == "decision_variables":
            print("  - Primarily affects decision variables")
        else:
            print("  - Affects both raw indicators and decision variables")
        
    except Exception as e:
        print(f"❌ Modification type detection failed: {e}")
        import traceback
        traceback.print_exc()

def test_variable_property_application():
    """Test the variable property application functionality."""
    print("\n=== TESTING VARIABLE PROPERTY APPLICATION ===")
    
    var_type = input("Test raw indicator or decision variable? (raw/decision): ").strip().lower()
    if var_type not in ["raw", "decision"]:
        var_type = "raw"
    
    is_raw_indicator = var_type == "raw"
    
    # Create a test variable
    test_var = {
        "id": "test_var",
        "name": "Test Variable",
        "var_name": "test_var",
        "priority": 3,
        "description": "A test variable",
        "priority_rationale": "For testing purposes",
        "formula": None if is_raw_indicator else "return 100;",
        "type": "text" if is_raw_indicator else "float",
        "value": None,
        "project_id": "test_project"
    }
    
    print(f"\nTesting property application for {'raw indicator' if is_raw_indicator else 'decision variable'}")
    print(f"Initial variable: {json.dumps(test_var, indent=2)}")
    
    try:
        _apply_default_variable_properties(test_var, is_raw_indicator=is_raw_indicator, project_id="test_project")
        
        print(f"\n✅ After property application:")
        print(json.dumps(test_var, indent=2))
        
    except Exception as e:
        print(f"❌ Property application failed: {e}")
        import traceback
        traceback.print_exc()

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print("Income Assessment System - Interactive Testing Mode")
    print("=" * 60)
    
    while True:
        print("\nAvailable Test Options:")
        print("1. Test Complete Workflow")
        print("2. Test Dependency Analysis")
        print("3. Test Intelligent Variable Modification")
        print("4. Test Variable Synchronization")
        print("5. Test Individual Components")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            test_complete_workflow()
        elif choice == "2":
            test_dependency_analysis()
        elif choice == "3":
            test_intelligent_modification()
        elif choice == "4":
            test_synchronization()
        elif choice == "5":
            test_individual_components()
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1-6.")
