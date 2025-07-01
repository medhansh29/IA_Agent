from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
from typing import Optional, Dict, List, Any, cast # Import 'cast'
import uuid # Import uuid for generating project_id
from fastapi.middleware.cors import CORSMiddleware

# Import GraphState and individual node functions from nodes.py
# Now importing the internal, granular generation functions
from nodes import (
    generate_variables,
    generate_questionnaire,
    modify_variables_intelligent,
    modify_questionnaire_llm,
    analyze_questionnaire_impact,
    write_to_supabase, # The single function used for multiple save points
    fetch_supabase_tables,
    analyze_variable_dependencies
)

# Import the GraphState schema
from schemas.schemas import GraphState

# Initialize the FastAPI application
api_app = FastAPI(
    title="Langraph Financial Assessment API (Step-by-Step with Finalize)",
    description="API to run the Langraph workflow with explicit finalization steps for variables and questionnaire.",
    version="1.0.0"
)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for API Request/Response ---

class InitialWorkflowRequest(BaseModel):
    """Schema for the request to start the workflow."""
    prompt: str

class SharedWorkflowState(BaseModel):
    """
    Represents the full GraphState that will be passed between API calls.
    All Optional fields indicate they might be None at certain stages.
    """
    prompt: str
    modification_prompt: Optional[str] = None # This will be set by specific modification endpoints
    modification_history: Optional[List[str]] = None # History of modifications, if needed
    raw_indicators: Optional[List[Dict[str, Any]]] = None
    decision_variables: Optional[List[Dict[str, Any]]] = None
    questionnaire: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    project_id: Optional[str] = None # New: To differentiate projects in DB
    dependency_graph: Optional[Any] = None # New: Stores dependency analysis results
    modification_reasoning: Optional[str] = None # New: Stores LLM reasoning for modifications
    status: Optional[str] = None  # New: Track workflow status
    needs_review: Optional[bool] = None  # New: Indicates if modifications need review


class ModificationRequest(BaseModel):
    """Schema for requests that involve a modification prompt."""
    current_state: SharedWorkflowState
    modification_prompt: str = ""


class ProjectIdRequest(BaseModel):
    """Schema for requests that require a project_id."""
    project_id: str


# --- API Endpoints for Step-by-Step Workflow ---

@api_app.post("/step/generate-variables", response_model=SharedWorkflowState, summary="Step 1: Generate Initial Variables")
async def step_generate_variables(request: InitialWorkflowRequest):
    """
    Initiates the workflow by generating initial raw indicators and decision variables based on the provided prompt.
    A unique `project_id` is generated for this workflow run.
    Returns the initial state with both raw indicators and decision variables populated.
    """
    try:
        new_project_id = str(uuid.uuid4()) # Generate a new UUID for the project

        # Create an initial GraphState for the first node
        initial_state = cast(GraphState, {
            "prompt": request.prompt,
            "modification_prompt": None,
            "raw_indicators": None, # Ensure RIs are explicitly None to trigger generation
            "decision_variables": None, # Ensure DVs are explicitly None to trigger generation
            "questionnaire": None,
            "error": None,
            "project_id": new_project_id, # Set the new project_id
            "status": "variables_generated",
            "needs_review": False
        })
        
        # Use the existing generate_variables function which handles both types
        updated_state = generate_variables(initial_state)

        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/generate-variables: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/step/modify-variables", response_model=SharedWorkflowState, summary="Step 2: Modify Variables")
async def step_modify_variables(request: ModificationRequest):
    """
    Applies LLM-driven modifications to raw indicators and/or decision variables based on the `modification_prompt`.
    Returns the updated state with modified variables.
    """
    try:
        current_state = cast(GraphState, request.current_state.model_dump())
        if current_state.get("modification_history") is None:
            current_state["modification_history"] = []
        if not isinstance(current_state["modification_history"], list):
            current_state["modification_history"] = []
        current_state["modification_history"].append(request.modification_prompt)
        current_state["modification_prompt"] = request.modification_prompt
        current_state["status"] = "variables_modified"
        updated_state = modify_variables_intelligent(current_state)
        return SharedWorkflowState(**updated_state)
    except Exception as e:
        print(f"Error in /step/modify-variables: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/step/generate-questionnaire", response_model=SharedWorkflowState, summary="Step 3: Generate Questionnaire")
async def step_generate_questionnaire(request: SharedWorkflowState):
    """
    Generates the questionnaire based on the finalized raw indicators and decision variables.
    Returns the updated state with the questionnaire populated.
    """
    try:
        current_state = cast(GraphState, request.model_dump())
        
        current_state["status"] = "questionnaire_generated"
        updated_state = generate_questionnaire(current_state)
        
        # Always analyze impact after generation
        updated_state = analyze_questionnaire_impact(updated_state)
        
        # Set needs_review based on impact analysis
        updated_state["needs_review"] = bool(updated_state.get("error"))
        
        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/generate-questionnaire: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/step/modify-questionnaire", response_model=SharedWorkflowState, summary="Step 4: Modify Questionnaire")
async def step_modify_questionnaire(request: ModificationRequest):
    """
    Applies LLM-driven modifications to the questionnaire structure based on the `modification_prompt`.
    Returns the updated state with the modified questionnaire and detailed reasoning.
    """
    try:
        current_state = cast(GraphState, request.current_state.model_dump())
        if current_state.get("modification_history") is None:
            current_state["modification_history"] = []
        if not isinstance(current_state["modification_history"], list):
            current_state["modification_history"] = []
        
        # Store modification history
        current_state["modification_history"].append(request.modification_prompt)
        current_state["modification_prompt"] = request.modification_prompt
        current_state["status"] = "questionnaire_modified"
        
        # Apply modifications with intelligent reasoning
        updated_state = modify_questionnaire_llm(current_state)
        
        # Always analyze impact after modification
        updated_state = analyze_questionnaire_impact(updated_state)
        
        # Set needs_review based on impact analysis
        updated_state["needs_review"] = bool(updated_state.get("error"))
        
        # Print reasoning for transparency
        if updated_state.get("modification_reasoning"):
            print("\n--- Questionnaire Modification Reasoning ---")
            print(updated_state["modification_reasoning"])
        
        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/modify-questionnaire: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/step/analyze-impact", response_model=SharedWorkflowState, summary="Step 5: Analyze Questionnaire Impact")
async def step_analyze_impact(request: SharedWorkflowState):
    """
    Analyzes the questionnaire for completeness and consistency with assessment variables.
    Attempts to remediate by adding missing questions if needed.
    Returns the state after analysis and potential remediation.
    """
    try:
        current_state = cast(GraphState, request.model_dump())
        
        current_state["status"] = "impact_analyzed"
        updated_state = analyze_questionnaire_impact(current_state)
        
        # Set needs_review based on impact analysis
        updated_state["needs_review"] = bool(updated_state.get("error"))
        
        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/analyze-impact: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/step/save-questionnaire", response_model=SharedWorkflowState, summary="Step 6: Save Final Questionnaire to DB")
async def step_save_questionnaire(request: SharedWorkflowState):
    """
    Saves all finalized raw indicators, decision variables, and the questionnaire to Supabase. This is the ONLY endpoint that writes to the database.
    Returns the final state with potential errors from the save operation.
    """
    try:
        current_state = cast(GraphState, request.model_dump())
        
        # Run one final impact analysis
        current_state = analyze_questionnaire_impact(current_state)
        
        if current_state.get("error"):
            raise HTTPException(
                status_code=400, 
                detail="Cannot save questionnaire with pending issues. Please resolve all issues first."
            )
        
        current_state["status"] = "saved"
        current_state["needs_review"] = False
        updated_state = write_to_supabase(current_state)
        return SharedWorkflowState(**updated_state)
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in /step/save-questionnaire: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/api/generate-assessment", response_model=SharedWorkflowState, summary="Generate a complete income assessment with raw indicators, decision variables, and questionnaire.")
async def generate_assessment(request: InitialWorkflowRequest):
    """
    Generate a complete income assessment with raw indicators, decision variables, and questionnaire.
    """
    try:
        prompt = request.prompt
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # For now, return a placeholder response since run_workflow is not available
        return SharedWorkflowState(
            prompt=prompt,
            project_id=str(uuid.uuid4()),
            raw_indicators=[],
            decision_variables=[],
            error="Workflow generation not yet implemented"
        )
        
    except Exception as e:
        print(f"Error in /api/generate-assessment: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/api/modify-variables", response_model=SharedWorkflowState, summary="Modify existing raw indicators and decision variables using intelligent synchronization.")
async def modify_variables(request: ModificationRequest):
    """
    Modify existing raw indicators and decision variables using intelligent synchronization.
    """
    try:
        current_state = cast(GraphState, request.current_state.model_dump())
        modification_prompt = request.modification_prompt
        existing_raw_indicators = current_state.get("raw_indicators", [])
        existing_decision_variables = current_state.get("decision_variables", [])
        project_id = current_state.get("project_id")
        
        if not modification_prompt:
            raise HTTPException(status_code=400, detail="Modification prompt is required")
        
        if not existing_raw_indicators and not existing_decision_variables:
            raise HTTPException(status_code=400, detail="At least one raw indicator or decision variable is required")
        
        # For now, return a placeholder response since run_variable_modification_only is not available
        return SharedWorkflowState(
            prompt=current_state.get("prompt", ""),
            modification_prompt=modification_prompt,
            project_id=project_id,
            raw_indicators=existing_raw_indicators,
            decision_variables=existing_decision_variables,
            error="Variable modification not yet implemented"
        )
        
    except Exception as e:
        print(f"Error in /api/modify-variables: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/api/analyze-dependencies", response_model=Dict[str, Any], summary="Analyze dependencies between raw indicators and decision variables.")
async def analyze_dependencies(request: ModificationRequest):
    """
    Analyze dependencies between raw indicators and decision variables.
    """
    try:
        current_state = cast(GraphState, request.current_state.model_dump())
        raw_indicators = current_state.get("raw_indicators", []) or []
        decision_variables = current_state.get("decision_variables", []) or []
        
        if not raw_indicators and not decision_variables:
            raise HTTPException(status_code=400, detail="At least one raw indicator or decision variable is required")
        
        # Run dependency analysis
        dependency_graph = analyze_variable_dependencies(raw_indicators, decision_variables)
        
        return {
            "success": True,
            "dependency_graph": dependency_graph
        }
        
    except Exception as e:
        print(f"Error in /api/analyze-dependencies: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.get("/api/fetch-supabase-tables", response_model=Dict[str, Any], summary="Fetch all rows from raw_indicators, decision_variables, and questionnaire tables in Supabase.")
def fetch_supabase_tables_api():
    """
    Fetch all rows from the three Supabase tables for display in the saved questionnaires section.
    """
    try:
        result = fetch_supabase_tables()
        return {"success": True, "data": result}
    except Exception as e:
        print(f"Error in /api/fetch-supabase-tables: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.get("/api/fetch-assessment/{project_id}", response_model=SharedWorkflowState)
async def fetch_assessment(project_id: str):
    """Fetch a specific assessment by project_id"""
    try:
        tables = fetch_supabase_tables()
        # Find the assessment with matching project_id
        for assessment in tables:
            if isinstance(assessment, dict) and assessment.get("project_id") == project_id:
                # Convert the assessment to a proper SharedWorkflowState
                state = SharedWorkflowState(
                    prompt=assessment.get("prompt", ""),
                    project_id=project_id,
                    raw_indicators=assessment.get("raw_indicators"),
                    decision_variables=assessment.get("decision_variables"),
                    questionnaire=assessment.get("questionnaire"),
                    status=assessment.get("status", "unknown"),
                    modification_history=assessment.get("modification_history", []),
                    dependency_graph=assessment.get("dependency_graph"),
                    modification_reasoning=assessment.get("modification_reasoning"),
                    needs_review=False  # Saved assessments don't need review
                )
                return state
        raise HTTPException(status_code=404, detail=f"Assessment with project_id {project_id} not found")
    except Exception as e:
        print(f"Error in /api/fetch-assessment: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.get("/api/status/{project_id}")
async def check_status(project_id: str):
    """Check the status of an assessment workflow"""
    try:
        tables = fetch_supabase_tables()
        for assessment in tables:
            if isinstance(assessment, dict) and assessment.get("project_id") == project_id:
                return {
                    "project_id": project_id,
                    "status": assessment.get("status", "unknown"),
                    "needs_review": assessment.get("needs_review", False),
                    "last_modified": assessment.get("last_modified")
                }
        raise HTTPException(status_code=404, detail=f"Assessment with project_id {project_id} not found")
    except Exception as e:
        print(f"Error in /api/status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run("api:api_app", host="0.0.0.0", port=8000, reload=True)

