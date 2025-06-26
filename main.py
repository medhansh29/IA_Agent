import json
import uuid # Import uuid for generating project_id
from langgraph.graph import StateGraph, END

# Import the GraphState and all the node functions from nodes.py
from nodes import (
    GraphState,
    generate_variables,
    generate_questionnaire,
    modify_variables_llm, # This function will be used for both AV and CV mods
    modify_questionnaire_llm,
    analyze_questionnaire_impact,
    write_to_supabase # We will use this function for multiple save points
)

# Define the Langraph workflow graph
workflow = StateGraph(GraphState)

# Add all the defined nodes to the workflow graph
workflow.add_node("generate_variables_node", generate_variables) # Generates both initial AVs and CVs
workflow.add_node("modify_variables_llm_node", modify_variables_llm) # Used for both AV and CV modifications
workflow.add_node("generate_questionnaire_node", generate_questionnaire)
workflow.add_node("modify_questionnaire_llm_node", modify_questionnaire_llm)
workflow.add_node("analyze_questionnaire_impact_node", analyze_questionnaire_impact)

# Add distinct nodes for saving at different stages
workflow.add_node("save_assessment_variables_node", write_to_supabase) # Save only AVs (and initial CVs if present)
workflow.add_node("save_computational_variables_node", write_to_supabase) # Save AVs and CVs
workflow.add_node("save_questionnaire_node", write_to_supabase) # Save final questionnaire (and all variables)


# Set the entry point for the graph
workflow.set_entry_point("generate_variables_node")

# Define the edges (transitions) between the nodes to control the flow

# 1. After initial variable generation, allow modification of variables
#    (Frontend will control whether this is AV or CV mod by calling specific API)
workflow.add_edge("generate_variables_node", "modify_variables_llm_node")

# 2. After variable modification (e.g., for Assessment Variables), save them
workflow.add_edge("modify_variables_llm_node", "save_assessment_variables_node")

# 3. After saving assessment variables, the workflow can logically proceed to generating questionnaire
#    (Note: In the full workflow, generate_variables_node will ensure CVs are generated if AVs are present)
workflow.add_edge("save_assessment_variables_node", "generate_questionnaire_node")

# 4. After questionnaire generation, allow modification of questionnaire
workflow.add_edge("generate_questionnaire_node", "modify_questionnaire_llm_node")

# 5. After questionnaire modification, analyze its impact
workflow.add_edge("modify_questionnaire_llm_node", "analyze_questionnaire_impact_node")

# 6. Finally, save the entire state (including questionnaire) to DB
workflow.add_edge("analyze_questionnaire_impact_node", "save_questionnaire_node")

# The workflow ends after the final save
workflow.add_edge("save_questionnaire_node", END)


# Compile the workflow graph into an executable application
app = workflow.compile()

# Re-export individual node functions so api.py can directly access them
__all__ = [
    "app", # The compiled full workflow
    "GraphState", # The state definition
    "generate_variables",
    "generate_questionnaire",
    "modify_variables_llm",
    "modify_questionnaire_llm",
    "analyze_questionnaire_impact",
    "write_to_supabase"
]

if __name__ == "__main__":
    # --- Running Test: Full Workflow - Generation, LLM Modification, Impact Analysis ---
    print("\n--- Running Full Test Workflow: Generation, LLM Modification, Impact Analysis ---")

    test_initial_prompt = "Assess income for a local tailor shop owner."
    # For a full end-to-end test, we'll provide a comprehensive modification prompt
    # that LLM's modify_variables_llm and modify_questionnaire_llm can act on.
    test_modification_prompt = (
        "For assessment variables: Add a new assessment variable named 'Customer Loyalty Score' (var_name: 'customer_loyalty_score') with priority 2, description 'A score indicating customer retention and repeat business based on interviews.', and type 'int'. "
        "For computational variables: Remove 'customer_acquisition_cost'. "
        "For the questionnaire: Update the description of the section with title 'Business Overview' to 'Questions about the fundamentals of your tailor shop.' "
        "Also, add a new core question to the 'Customer Insights' section: 'How many unique customers do you serve per week?' (type: 'integer', var_name: 'q_weekly_unique_customers', impacts 'customer_base_size')."
        "Remove the question with variable_name 'q_experience_in_tailoring_2727'." # This will test impact analysis as 'experience_in_tailoring' will no longer be directly covered
    )

    # Generate a unique project_id for this test run
    test_project_id = str(uuid.uuid4())
    print(f"Starting workflow with Project ID: {test_project_id}")

    initial_test_state: GraphState = {
        "prompt": test_initial_prompt,
        "modification_prompt": test_modification_prompt,
        "assessment_variables": None,
        "computational_variables": None,
        "questionnaire": None,
        "error": None,
        "project_id": test_project_id # Pass the generated project_id
    }

    try:
        # Invoke the compiled Langraph app with the initial state
        final_state = app.invoke(initial_test_state)

        print("\n--- Final State of the Langraph Workflow ---")
        print(f"Workflow completed for Project ID: {final_state.get('project_id')}")
        print(f"Final Error State: {final_state.get('error')}")
        print("\n--- Final Questionnaire ---")
        print(json.dumps(final_state.get("questionnaire", {}), indent=2))
        print("\n--- Final Assessment Variables ---")
        print(json.dumps(final_state.get("assessment_variables", []), indent=2))
        print("\n--- Final Computational Variables ---")
        print(json.dumps(final_state.get("computational_variables", []), indent=2))

        if final_state.get("error"):
            print(f"\nWorkflow Completed with Errors/Warnings: {final_state['error']}")
        else:
            print("\nWorkflow Completed Successfully. Data processed and sent to Supabase (simulated).")

    except Exception as e:
        print(f"\nAn unexpected error occurred during workflow execution: {e}")

