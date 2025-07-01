from langchain_core.prompts import ChatPromptTemplate

# --- Prompt 1: Raw Indicators Generation (RAG enabled) ---
RAW_INDICATORS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant for a fintech company lending to subprime customers with thin credit files. "
         "Your role is to help relationship managers generate user interview questions to assess and gather "
         "insights about small business owners. These interviews can be delivered via omnichannel interfaces "
         "(in-person, web, app, WhatsApp, voice). The gathered insights will also help refine customer "
         "segmentation for future growth experiments. "
         "Your task is to identify the most impactful trade or business specific raw indicators "
         "that will come from these interviews to accurately assess their income. "
         "The clients are small businesses with no steady source of income like a salary. Their owners are "
         "usually uneducated or marginally educated, They usually fall below the poverty line or are classified as low-income. "
         "For each variable, provide an 'id' (a unique string), 'name' (display name), "
         "'var_name' (snake_case for coding, e.g., 'avg_daily_customers'), "
         " 'impact_score' (integer 0-100, higher means more important), 'description', "
         "'priority_rationale' (explanation for why this impact score was assigned), "
         "'formula' (always null for raw indicators), "
         "'type' (the type of expected input for this variable, e.g., 'text', 'integer', 'float', 'boolean', 'dropdown'), "
         "'value' (always null for raw indicators, this is for user input later), "
         "'project_id' (the unique ID for the current project workflow). "
         "**Ensure ALL fields in the schema are present and correctly formatted, including 'type', 'priority', 'impact_score', 'description', and 'priority_rationale'.**" # Explicit reminder
         "Generate atleast 15 realistic and useful raw indicators that are relevant to small business income assessment. "
         "\n\n--- Supplementary Context from Historical Data (use to inspire and refine, but prioritize main task and schema adherence) ---\n{context}\n----------------------------------------------------------------------" # Moved to end, clarified role
        ),
        ("human", "Based on the following user input and any existing variables, generate a list of raw indicators: {user_input}. "
         "Existing variables to consider for modification or reference: {existing_variables}. "
         "Provide the output in JSON format, strictly following the RawIndicatorsOutput schema. "
         "Ensure 'formula' is always null for raw indicators. "
         "Ensure 'impact_score' (integer 0-100, higher means more important) and 'project_id' are included in each variable object using the provided project ID."
        )
    ]
)

# --- Prompt 2: Decision Variables Generation (RAG enabled) ---
DECISION_VARIABLES_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant helping a fintech company. Your task is to generate "
         "decision variables based on raw indicators. These variables "
         "should be directly derivable or calculated from the raw indicators "
         "or other decision variables. "
         "For each decision variable, provide 'id', 'name', 'var_name', 'priority', 'impact_score' (integer 0-100, higher means more important), "
         "'description', 'priority_rationale' (explanation for why this impact score was assigned), "
         "'type' (e.g., 'float'), and a 'formula' (JavaScript string) "
         "that calculates its value based on other 'var_name's. 'value' should be null. "
         "Ensure 'project_id' is included. "
         "**Generate meaningful and accurate JavaScript formulas which tell how the decision variables are to be computed from the raw indicators, strictly adhering to the schema.**" # Explicit reminder
         "**IMPORTANT**: For each decision variable, include a detailed rationale in the 'priority_rationale' field explaining why this variable is important for income assessment and decision-making.**" # New requirement
         "genrate atleast 10 decision variables that are relevant to small business income assessment. "
         "\n\n--- Supplementary Context from Historical Data (use to inspire and refine, but prioritize main task and schema adherence) ---\n{context}\n----------------------------------------------------------------------" # Moved to end, clarified role
        ),
        ("human", "Given the following raw indicators:\n{raw_indicators}\n\n"
         "And considering these existing decision variables for modification or reference:\n{existing_decision_variables}\n\n"
         "Based on the user's initial input: '{user_input}', generate relevant decision variables. "
         "Strictly provide the output in JSON format, following the DecisionVariablesOutput schema. "
         "Ensure all 'formula' values are valid JavaScript expressions that can be evaluated. "
         "Example formula: 'return q_daily_sales * q_num_days_week * 4;' if q_daily_sales is a raw indicator. "
         "Only include variables that are directly calculable from the provided raw indicators. "
         "Ensure 'impact_score' (integer 0-100, higher means more important) and 'project_id' are included in each variable object using the provided project ID."
        )
    ]
)

# --- Prompt 3: Intelligent Variable Modifications (NEW) ---
INTELLIGENT_VARIABLE_MODIFICATIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant tasked with making intelligent modifications to a financial assessment system. "
         "Your role is to understand the dependencies between raw indicators and decision variables, and ensure "
         "that any modifications maintain consistency and completeness of the assessment framework. "
         "\n\n"
         "**CRITICAL UNDERSTANDING:**\n"
         "- Raw indicators are the basic data points collected from interviews\n"
         "- Decision variables are calculated from raw indicators using JavaScript formulas\n"
         "- When raw indicators change, decision variables that depend on them may break\n"
         "- When decision variables are added/modified, new raw indicators might be needed\n"
         "\n\n"
         "**YOUR TASK:**\n"
         "1. Analyze the user's modification request\n"
         "2. Identify all dependencies and potential impacts\n"
         "3. Generate comprehensive modifications that maintain system consistency\n"
         "4. Provide compensatory changes for the other variable type if needed\n"
         "5. Explain your reasoning for all changes\n"
         "\n\n"
         "**BUSINESS CONTEXT:** {business_context}\n"
         "\n\n"
         "**DEPENDENCY ANALYSIS:**\n"
         "Raw Indicators Available: {raw_indicators}\n"
         "Decision Variables and Their Dependencies:\n{dependency_analysis}\n"
         "\n\n"
         "**MODIFICATION REQUEST:** {primary_modifications}\n"
         "\n\n"
         "**OUTPUT REQUIREMENTS:**\n"
         "- primary_modifications: Changes to the primary variable type (raw indicators or decision variables)\n"
         "- compensatory_modifications: Changes needed to the other variable type to maintain consistency\n"
         "- removed_variables: Variables that should be removed due to dependencies\n"
         "- updated_formulas: Formula updates needed for decision variables\n"
         "- new_variables: New variables to be added (if any)\n"
         "- reasoning: Detailed explanation of why each change is necessary\n"
         "\n\n"
         "**IMPORTANT GUIDELINES:**\n"
         "- Always maintain the integrity of the assessment framework\n"
         "- If a raw indicator is removed, either remove dependent decision variables or suggest alternatives\n"
         "- If a raw indicator name changes, update all referencing formulas\n"
         "- If new raw indicators are added, suggest relevant decision variables\n"
         "- Ensure all decision variable formulas reference valid raw indicators\n"
         "- Consider the business context when making decisions about what to keep or remove"
        ),
        ("human", "Please analyze the modification request and provide comprehensive changes that maintain system consistency. "
         "Focus on understanding the dependencies and ensuring the final state makes business sense for income assessment."
        )
    ]
)

# --- Prompt 4: Dependency Analysis (NEW) ---
DEPENDENCY_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant specialized in analyzing dependencies between financial assessment variables. "
         "Your task is to parse JavaScript formulas and identify which raw indicators are referenced by decision variables. "
         "\n\n"
         "**ANALYSIS REQUIREMENTS:**\n"
         "1. Parse each decision variable's formula\n"
         "2. Extract all raw indicator references (variables that start with common prefixes or are explicitly referenced)\n"
         "3. Identify the impact level of each dependency\n"
         "4. Detect orphaned raw indicators (not used by any decision variable)\n"
         "5. Identify potential new decision variables that could be created\n"
         "\n\n"
         "**FORMULA PARSING RULES:**\n"
         "- Look for variable names that match raw indicator var_names\n"
         "- Consider common prefixes like 'q_', 'monthly_', 'daily_', etc.\n"
         "- Identify mathematical operations and logical conditions\n"
         "- Note any hardcoded values or constants\n"
         "\n\n"
         "**IMPACT LEVELS:**\n"
         "- 'critical': Variable cannot function without this dependency\n"
         "- 'moderate': Variable can be adapted or has alternatives\n"
         "- 'low': Variable has minimal dependency on this raw indicator"
        ),
        ("human", "Please analyze the following variables and provide a comprehensive dependency analysis:\n\n"
         "Raw Indicators:\n{raw_indicators}\n\n"
         "Decision Variables:\n{decision_variables}\n\n"
         "Provide the analysis in JSON format with dependency mappings and impact assessments."
        )
    ]
)

# --- Prompt 5: Questionnaire Generation (RAG enabled) ---
QUESTIONNAIRE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant that designs comprehensive and logical questionnaires "
         "for small business financial assessment. Your task is to generate a questionnaire "
         "structure, organized into sections. Each `Section` must explicitly contain two lists of questions: "
         "`core_questions` and `conditional_questions`. "
         "\n\n"
         "**SECTION PROPERTIES:**\n"
         "- 'is_mandatory': boolean, mostly always true. Can be false if the section is optional(once or two sections maximum)"
         "  - If false, section needs 'triggering_criteria' (a description of the criteria that must be met for the section to be displayed) IMPORTANT: it must be there if the section is optional"
         "- Each section must have both core_questions and conditional_questions lists\n"
         "generate atleast 4-7 sections which are relevant to the business context"
         "\n\n"
         "**QUESTION PROPERTIES:**\n"
         "For each `Question`, provide:\n"
         "- 'id': unique identifier\n"
         "- 'text': question text\n"
         "- 'type': one of 'text', 'integer', 'float', 'boolean', 'dropdown'\n"
         "- 'variable_name': snake_case name\n"
         "- 'raw_indicators': list of var_names this question helps capture\n"
         "- 'formula': JS function showing how to map the answer to raw indicators\n"
         "- 'is_conditional': boolean, mostly always false. Can be true if the question is conditional on the answer to a previous question"
         "  - If true, question needs 'question_triggering_criteria' (a description of what trigegrs the question)\n"
         "  - If false, question has no triggering criteria (null)\n"
         "  - Only one question can be conditional in a section"
         "\n\n"
         "**IMPORTANT NOTES:**\n"
         "1. Section mandatory vs Question conditional:\n"
         "   - Optional sections (is_mandatory=false) need section-level triggering_criteria\n"
         "   - Conditional questions (is_conditional=true) need question-level question_triggering_criteria\n"
         "   - These are independent: you can have conditional questions in mandatory sections\n"
         "2. More than one question can map to one raw indicator for granularity\n"
         "   Example: Monthly expenses -> separate questions for rent, utilities, supplies\n"
         "3. Include at least one optional section (is_mandatory=false)\n"
         "4. Include some conditional questions (is_conditional=true) in each section\n"
         "5. Ensure all formulas and triggering criteria are valid JS\n"
         "\n\n"
         "--- Supplementary Context from Historical Data ---\n"
         "{context}\n"
         "----------------------------------------------------------------------"
        ),
        ("human", 
         "Generate a questionnaire based on the user's prompt: '{user_input}'. "
         "The questionnaire should gather data for these raw indicators:\n{raw_indicators}\n\n"
         "And these decision variables:\n{decision_variables}\n\n"
         "Strictly provide the output in JSON format, following the QuestionnaireOutput schema. "
         "Ensure logical ordering, comprehensive coverage of variables, and appropriate question types. "
         "Remember: section mandatory flag and question conditional flag are independent!"
        )
    ]
)

# --- Prompt 6: Questionnaire Modifications (RAG context removed) ---
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
         "If the modification request asks to make a question more granular, break it into multiple simpler, more focused questions that together cover the same variable(s). "
         "Always provide at least one concrete modification if the prompt requests more detail, granularity, or similar questions. "
         "If a question is too broad or complex, suggest splitting it into smaller questions. "
         "If the modification request is ambiguous, err on the side of making the questionnaire more detailed and actionable. "
         "Strictly follow the QuestionnaireModificationsOutput schema."
        ),
        ("human", "Based on this modification request: '{modification_request}', "
         "and the current questionnaire structure:\n{current_questionnaire}\n\n"
         "Generate the necessary modifications. If no changes are needed, return empty lists/dictionaries for all modification types."
        )
    ]
)

# --- Prompt 7: Intelligent Questionnaire Modifications ---
INTELLIGENT_QUESTIONNAIRE_MODIFICATIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant tasked with making intelligent modifications to a financial assessment questionnaire. "
         "Your role is to understand the relationships between questions, raw indicators, and the overall assessment structure, "
         "ensuring that any modifications maintain data collection completeness and logical flow. "
         "\n\n"
         "**CRITICAL UNDERSTANDING:**\n"
         "- Questions are organized into sections with core and conditional questions\n"
         "- Each question maps to one or more raw indicators\n"
         "- Questions may have dependencies on previous questions (triggering_criteria)\n"
         "- Raw indicators must be fully covered by the questionnaire\n"
         "- Question flow should be logical and user-friendly\n"
         "\n\n"
         "**YOUR TASK:**\n"
         "1. Analyze the user's modification request\n"
         "2. Identify all affected questions and raw indicators\n"
         "3. Generate comprehensive modifications that maintain questionnaire completeness\n"
         "4. Ensure all raw indicators remain calculable\n"
         "5. Explain your reasoning for all changes\n"
         "\n\n"
         "**BUSINESS CONTEXT:** {business_context}\n"
         "\n\n"
         "**CURRENT STATE:**\n"
         "Raw Indicators Available: {raw_indicators}\n"
         "Current Questionnaire Structure:\n{current_questionnaire}\n"
         "\n\n"
         "**MODIFICATION REQUEST:** {modification_prompt}\n"
         "\n\n"
         "**OUTPUT REQUIREMENTS:**\n"
         "- added_sections: New sections to add\n"
         "- updated_sections: Sections to modify\n"
         "- removed_section_orders: Sections to remove\n"
         "- added_questions: New questions to add\n"
         "- updated_questions: Questions to modify\n"
         "- removed_question_variable_names: Questions to remove\n"
         "- reasoning: Detailed explanation of changes and their impact\n"
        ),
        ("human",
         "Please analyze the current questionnaire and the modification request, then provide a comprehensive set of changes "
         "that maintains questionnaire completeness and logical flow. Ensure all raw indicators remain calculable and explain "
         "your reasoning."
        )
    ]
)

# --- Prompt 8: JavaScript Expression Refinement (Remains same) ---
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
         "If '{expression_type}' is a formula, it should describe how the question's answer contributes to a raw indicator, or perform a relevant calculation. "
         "If '{expression_type}' is a triggering criteria, it should determine when a section or question is displayed."
        )
    ]
)