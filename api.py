from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
from typing import Optional, Dict, List, Any, cast # Import 'cast'
import uuid # Import uuid for generating project_id

# Import GraphState and individual node functions from nodes.py
# Now importing the internal, granular generation functions
from nodes import (
    GraphState,
    generate_variables,
    generate_questionnaire,
    modify_variables_llm,
    modify_questionnaire_llm,
    analyze_questionnaire_impact,
    write_to_supabase # The single function used for multiple save points
)

# Initialize the FastAPI application
api_app = FastAPI(
    title="Langraph Financial Assessment API (Step-by-Step with Finalize)",
    description="API to run the Langraph workflow with explicit finalization steps for variables and questionnaire.",
    version="1.0.0"
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
    assessment_variables: Optional[List[Dict[str, Any]]] = None
    computational_variables: Optional[List[Dict[str, Any]]] = None
    questionnaire: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    project_id: Optional[str] = None # New: To differentiate projects in DB


class ModificationRequest(BaseModel):
    """Schema for requests that involve a modification prompt."""
    current_state: SharedWorkflowState
    modification_prompt: str = ""


# --- API Endpoints for Step-by-Step Workflow ---

@api_app.post("/step/generate-variables", response_model=SharedWorkflowState, summary="Step 1: Generate Initial Variables")
async def step_generate_variables(request: InitialWorkflowRequest):
    """
    Initiates the workflow by generating initial assessment and computational variables based on the provided prompt.
    A unique `project_id` is generated for this workflow run.
    Returns the initial state with both assessment and computational variables populated.
    """
    try:
        new_project_id = str(uuid.uuid4()) # Generate a new UUID for the project

        # Create an initial GraphState for the first node
        initial_state = cast(GraphState, {
            "prompt": request.prompt,
            "modification_prompt": None,
            "assessment_variables": None, # Ensure AVs are explicitly None to trigger generation
            "computational_variables": None, # Ensure CVs are explicitly None to trigger generation
            "questionnaire": None,
            "error": None,
            "project_id": new_project_id # Set the new project_id
        })
        
        # Use the existing generate_variables function which handles both types
        updated_state = generate_variables(initial_state)

        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/generate-variables: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/step/modify-assessment-variables", response_model=SharedWorkflowState, summary="Step 2: Modify Assessment Variables")
async def step_modify_assessment_variables(request: ModificationRequest):
    """
    Applies LLM-driven modifications to the assessment variables based on the `modification_prompt`.
    The `modification_prompt` should specifically target assessment variables.
    Returns the updated state with modified assessment variables.
    """
    try:
        current_state = cast(GraphState, request.current_state.model_dump())
        # ADD THIS: Track modification history
        if current_state.get("modification_history") is None:
            current_state["modification_history"] = []
        
        # Add the new modification to history
        # Ensure modification_history is a list before appending
        if not isinstance(current_state["modification_history"], list):
            current_state["modification_history"] = []
        current_state["modification_history"].append(request.modification_prompt)
        
        # Still update the current modification prompt
        current_state["modification_prompt"] = request.modification_prompt
        
        updated_state = modify_variables_llm(current_state)
        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/modify-assessment-variables: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/step/finalize-assessment-variables", response_model=SharedWorkflowState, summary="Step 3: Finalize and Save Assessment Variables to DB")
async def step_finalize_assessment_variables(request: SharedWorkflowState):
    """
    Finalizes the assessment variables by saving them to the database.
    This step primarily ensures assessment variables are persisted before proceeding.
    Returns the state after the save operation.
    """
    try:
        current_state = cast(GraphState, request.model_dump())
        
        updated_state = write_to_supabase(current_state)
        
        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/finalize-assessment-variables: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/step/modify-computational-variables", response_model=SharedWorkflowState, summary="Step 4: Modify Computational Variables")
async def step_modify_computational_variables(request: ModificationRequest):
    """
    Applies LLM-driven modifications to the computational variables based on the `modification_prompt`.
    The `modification_prompt` should specifically target computational variables.
    Returns the updated state with modified computational variables.
    """
    try:
        current_state = cast(GraphState, request.current_state.model_dump())
        current_state["modification_prompt"] = request.modification_prompt
        
        updated_state = modify_variables_llm(current_state)
        
        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/modify-computational-variables: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/step/finalize-computational-variables", response_model=SharedWorkflowState, summary="Step 5: Finalize and Save Computational Variables to DB")
async def step_finalize_computational_variables(request: SharedWorkflowState):
    """
    Finalizes the computational variables by saving them to the database.
    This step ensures all variables are persisted before questionnaire generation.
    Returns the state after the save operation.
    """
    try:
        current_state = cast(GraphState, request.model_dump())
        
        updated_state = write_to_supabase(current_state)
        
        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/finalize-computational-variables: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/step/generate-questionnaire", response_model=SharedWorkflowState, summary="Step 6: Generate Questionnaire")
async def step_generate_questionnaire(request: SharedWorkflowState):
    """
    Generates the questionnaire based on the finalized assessment and computational variables.
    Returns the updated state with the questionnaire populated.
    """
    try:
        current_state = cast(GraphState, request.model_dump())
        
        updated_state = generate_questionnaire(current_state)
        
        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/generate-questionnaire: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/step/modify-questionnaire", response_model=SharedWorkflowState, summary="Step 7: Modify Questionnaire")
async def step_modify_questionnaire(request: ModificationRequest):
    """
    Applies LLM-driven modifications to the questionnaire structure based on the `modification_prompt`.
    Returns the updated state with the modified questionnaire.
    """
    try:
        current_state = cast(GraphState, request.current_state.model_dump())
        current_state["modification_prompt"] = request.modification_prompt
        
        updated_state = modify_questionnaire_llm(current_state)
        
        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/modify-questionnaire: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/step/analyze-impact", response_model=SharedWorkflowState, summary="Step 8: Analyze Questionnaire Impact and Remediate")
async def step_analyze_impact(request: SharedWorkflowState):
    """
    Analyzes the questionnaire for completeness and consistency with assessment variables.
    Attempts to remediate by adding missing questions if needed.
    Returns the state after analysis and potential remediation.
    """
    try:
        current_state = cast(GraphState, request.model_dump())
        
        updated_state = analyze_questionnaire_impact(current_state)
        
        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/analyze-impact: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_app.post("/step/save-questionnaire", response_model=SharedWorkflowState, summary="Step 9: Save Final Questionnaire to DB")
async def step_save_questionnaire(request: SharedWorkflowState):
    """
    Saves the final questionnaire to Supabase. This will also upsert any variables again.
    Returns the final state with potential errors from the save operation.
    """
    try:
        current_state = cast(GraphState, request.model_dump())
        
        updated_state = write_to_supabase(current_state)
        
        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/save-questionnaire: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# To run this FastAPI application, save it as `api.py` and run:
# uvicorn api:api_app --host 0.0.0.0 --port 8000 --reload
# (Use --reload for development to automatically restart on code changes)

# You can then access the API documentation at http://127.0.0.1:8000/docs
# and make POST requests to the new endpoints (e.g., http://127.0.0.1:8000/step/generate-assessment-variables)
    """
    Saves the final questionnaire to Supabase. This will also upsert any variables again.
    Returns the final state with potential errors from the save operation.
    """
    try:
        current_state = cast(GraphState, request.model_dump())
        
        updated_state = write_to_supabase(current_state)
        
        return SharedWorkflowState(**updated_state)

    except Exception as e:
        print(f"Error in /step/save-questionnaire: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# To run this FastAPI application, save it as `api.py` and run:
# uvicorn api:api_app --host 0.0.0.0 --port 8000 --reload
# (Use --reload for development to automatically restart on code changes)

# You can then access the API documentation at http://127.0.0.1:8000/docs
# and make POST requests to the new endpoints (e.g., http://127.0.0.1:8000/step/generate-assessment-variables)

