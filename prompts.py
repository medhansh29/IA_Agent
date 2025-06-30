from langchain_core.prompts import ChatPromptTemplate

# --- Prompt 1: Assessment Variables Generation (RAG enabled) ---
ASSESSMENT_VARIABLES_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant for a fintech company lending to subprime customers with thin credit files. "
         "Your role is to help relationship managers generate user interview questions to assess and gather "
         "insights about small business owners. These interviews can be delivered via omnichannel interfaces "
         "(in-person, web, app, WhatsApp, voice). The gathered insights will also help refine customer "
         "segmentation for future growth experiments. "
         "Your task is to identify the most impactful trade or business specific assessment variables "
         "that will come from these interviews to accurately assess their income. "
         "The clients are small businesses with no steady source of income like a salary. Their owners are "
         "usually uneducated or marginally educated, They usually fall below the poverty line or are classified as low-income. "
         "For each variable, provide an 'id' (a unique string), 'name' (display name), "
         "'var_name' (snake_case for coding, e.g., 'avg_daily_customers'), "
         "'priority' (integer, 1 being highest, 5 being lowest), 'description', "
         "'formula' (always null for assessment variables), "
         "'type' (the type of expected input for this variable, e.g., 'text', 'integer', 'float', 'boolean', 'dropdown'), "
         "'value' (always null for assessment variables, this is for user input later), "
         "'project_id' (the unique ID for the current project workflow). "
         "**Ensure ALL fields in the schema are present and correctly formatted, including 'type', 'priority', and 'description'.**" # Explicit reminder
         "Generate atleast 15 realistic and useful assessment variables that are relevant to small business income assessment. "
         "\n\n--- Supplementary Context from Historical Data (use to inspire and refine, but prioritize main task and schema adherence) ---\n{context}\n----------------------------------------------------------------------" # Moved to end, clarified role
        ),
        ("human", "Based on the following user input and any existing variables, generate a list of assessment variables: {user_input}. "
         "Existing variables to consider for modification or reference: {existing_variables}. "
         "Provide the output in JSON format, strictly following the AssessmentVariablesOutput schema. "
         "Ensure 'formula' is always null for assessment variables. "
         "Ensure 'project_id' is included in each variable object using the provided project ID."
        )
    ]
)

# --- Prompt 2: Computational Variables Generation (RAG enabled) ---
COMPUTATIONAL_VARIABLES_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant helping a fintech company. Your task is to generate "
         "computational variables based on assessment variables. These variables "
         "should be directly derivable or calculated from the assessment variables "
         "or other computational variables. "
         "For each computational variable, provide 'id', 'name', 'var_name', 'priority', "
         "'description', 'type' (e.g., 'float'), and a 'formula' (JavaScript string) "
         "that calculates its value based on other 'var_name's. 'value' should be null. "
         "Ensure 'project_id' is included. "
         "**Generate meaningful and accurate JavaScript formulas which tell how the computational variables are to be computed from the assessment variables, strictly adhering to the schema.**" # Explicit reminder
         "genrate atleast 10 computational variables that are relevant to small business income assessment. "
         "\n\n--- Supplementary Context from Historical Data (use to inspire and refine, but prioritize main task and schema adherence) ---\n{context}\n----------------------------------------------------------------------" # Moved to end, clarified role
        ),
        ("human", "Given the following assessment variables:\n{assessment_variables}\n\n"
         "And considering these existing computational variables for modification or reference:\n{existing_computational_variables}\n\n"
         "Based on the user's initial input: '{user_input}', generate relevant computational variables. "
         "Strictly provide the output in JSON format, following the ComputationalVariablesOutput schema. "
         "Ensure all 'formula' values are valid JavaScript expressions that can be evaluated. "
         "Example formula: 'return q_daily_sales * q_num_days_week * 4;' if q_daily_sales is an assessment variable. "
         "Only include variables that are directly calculable from the provided assessment variables. "
         "Ensure 'project_id' is included in each variable object using the provided project ID."
        )
    ]
)

# --- Prompt 3: Questionnaire Generation (RAG enabled) ---
QUESTIONNAIRE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant that designs comprehensive and logical questionnaires "
         "for small business financial assessment. Your task is to generate a questionnaire "
         "structure, organized into sections. Each `Section` must explicitly contain two lists of questions: "
         "`core_questions` and `conditional_questions`. "
         "Each `Question` within these lists should be designed to collect data for "
         "the provided assessment and computational variables. "
         "For each `Question`, provide a unique 'id', 'text', 'type' "
         "(e.g., 'text', 'integer', 'float', 'boolean', 'dropdown'), 'variable_name' (snake_case), "
         "'triggering_criteria' (If is_mandatory == false; triggering_criteria is a MANDATORY FIELD that has to tell what triggers the optional section), "
         "'assessment_variables' (list of 'var_name's they impact), 'is_mandatory' (boolean), "
         "'is_conditional' (boolean), and 'formula' (a MANDATORY FIELD that tells how the assessment variables map to the question variables). "
         "Do not include an 'options' field in the Question schema. "
         "Ensure all variables in 'assessment_variables' list within questions map to actual 'var_name's "
         "from the provided assessment_variables. "
         "**IMPORTANT NOTE**: More than one  question can be mapped to one assessment variable, if that allows the question to be smaller and easier to answer."
         "For example: Monthly expenses for a tea stall can be broken down into multiple questions like: monthly rent, monthly milk cost, monthly sugar cost, etc. Then they can map to a single assessment variable called 'monthly_expenses'. "
         "Design a logical flow. If a question's answer is dependent on a previous question, "
         "use 'triggering_criteria'. if is_conditional is true, then 'triggering_criteria' must be a JS function that returns what triggers the conditional question. "
         "**Ensure ALL fields in the schema are present and correctly formatted, including 'is_mandatory', 'is_conditional', and 'formula'. Generate realistic and useful triggers/formulas.**" # Explicit reminder
         "Ensure that there are at least 5 sections, with a mix of core and conditional questions. Each section should have atleast 4 questions. Make sure to include atleast one optional section and atleast one conditional question. make sure they are intelligent and meaningful in the context of financial assessment. "
         "\n\n--- Supplementary Context from Historical Data (use to inspire and refine, but prioritize main task and schema adherence) ---\n{context}\n----------------------------------------------------------------------" # Moved to end, clarified role
        ),
        ("human", "Generate a questionnaire based on the user's prompt: '{user_input}'. "
         "The questionnaire should gather data for these assessment variables:\n{assessment_variables}\n\n"
         "And these computational variables:\n{computational_variables}\n\n"
         "Strictly provide the output in JSON format, following the QuestionnaireOutput schema. "
         "Ensure logical ordering, comprehensive coverage of variables, and appropriate question types. "
         "Add meaningful 'triggering_criteria' for conditional sections/questions, and 'formula' for calculated questions where applicable. "
         "Ensure 'project_id' is included in each section and question object."
        )
    ]
)

# --- Prompt 4: Variable Modifications (RAG context removed) ---
VARIABLE_MODIFICATIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant tasked with modifying a list of assessment and computational variables. "
         "Based on the 'modification_request', you need to identify which variables to add, update, or remove. "
         "For updates, specify the 'id' and the fields to change in the 'updates' dictionary. For removals, specify the 'id'. "
         "For additions, create new variables following the VariableSchema, ensuring 'id' is a new UUID. "
         "Your output must strictly follow the VariableModificationsOutput schema. "
         "Ensure 'project_id' is preserved for updated/removed variables and set for new ones. "
        ),
        ("human", "Based on this modification request: '{modification_request}', "
         "and the current assessment variables:\n{current_assessment_variables}\n\n"
         "And current computational variables:\n{current_computational_variables}\n\n"
         "Generate the necessary modifications. If no changes are needed, return empty lists/dictionaries for all modification types."
        )
    ]
)

# --- Prompt 5: Questionnaire Modifications (RAG context removed) ---
QUESTIONNAIRE_MODIFICATIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant tasked with modifying a JSON representation of a questionnaire. "
         "Based on the 'modification_request', you need to identify which sections/questions to add, update, or remove. "
         "For updates on sections, specify the 'order' of the section and the fields to change in the 'updates' dictionary. "
         "For updates on questions, specify the 'id' of the question and the fields to change in the 'updates' dictionary. "
         "For removals, specify the 'order' for sections or 'variable_name' for questions. "
         "For additions, create new sections/questions following their respective schemas, ensuring 'id' is a new UUID for questions. "
         "For added questions, specify `section_order` and `is_core` to indicate whether it belongs to `core_questions` or `conditional_questions`. "
         "Your output must strictly follow the QuestionnaireModificationsOutput schema. "
         "Ensure 'project_id' is preserved for updated/removed entities and set for new ones. "
        ),
        ("human", "Based on this modification request: '{modification_request}', "
         "and the current questionnaire structure:\n{current_questionnaire}\n\n"
         "Generate the necessary modifications. If no changes are needed, return empty lists/dictionaries for all modification types."
        )
    ]
)


# --- Prompt 6: JavaScript Expression Refinement (Remains same) ---
JS_REFINEMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a JavaScript expert. Your task is to refine or generate a JavaScript expression. "
         "The expression should be concise, syntactically correct, and semantically meaningful "
         "for financial assessment logic. If it's a triggering criteria, it must return a boolean. "
         "If it's a formula, it must return a calculated value. "
         "You can assume that variables from previous questions will be available and are prefixed with 'q_'. "
         "For example, 'q_income' for an income question. "
         "Available question variables are provided as `context_question_vars`. "
         "Avoid simplistic 'return true;' or empty expressions. "
         "The purpose is to create a dynamic condition for '{target_entity_description}'. "
         "Available question variables: {context_question_vars}. "
         "Generate only the JavaScript expression within a JSON object, with the key 'expression', nothing else. For example: `{{\"expression\": \"return q_var1 > 0 && q_var2 === \\\"Yes\\\";\"}}`. Provide a realistic and smart trigger for a financial assessment context. "
         "Consider using logical operators (&&, ||, !), numerical comparisons (>, <, >=, <=, ===), "
         "string comparisons, or checking for specific values. "
         "Examples for triggers: 'return q_has_dependents === true && q_income_source === \"Self-employed\";', "
         "'return q_business_type.includes(\\\"online\\\") || q_monthly_sales > 5000;'. "
        ),
        ("human", "Generate a suitable JavaScript expression for the '{expression_type}' of '{target_entity_description}'. "
         "It should be intelligent and meaningful in the context of financial assessment conditional logic. "
         "The expression to refine is: ['expression_to_refine']. "
         "If '{expression_type}' is a formula, it should describe how the question's answer contributes to an assessment variable, or perform a relevant calculation. "
         "If '{expression_type}' is a triggering criteria, it should determine when a section or question is displayed."
        )
    ]
)