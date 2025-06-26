from langchain_core.prompts import ChatPromptTemplate

# --- Prompt 1: Assessment Variables Generation ---
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
         "'type' (the type of expected input for this variable, e.g., 'text', 'int', 'float', 'boolean', 'date', 'dropdown')."
        ),
        ("human",
         "Based on the business context of '{prompt}', list the key assessment variables needed. "
         "Ensure they are specific and measurable for income assessment of small business owners. "
         "Exclude any variables that would require a complex formula to derive, as these are 'assessment' variables. "
         "Prioritize variables that provide direct insight into income, costs, and operational stability."
        )
    ]
)

# --- Prompt 2: Computational Variables Generation ---
COMPUTATIONAL_VARIABLES_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant for a fintech company. Your role is to define key computational variables "
         "that can be derived from existing assessment variables for small business owners. "
         "These derived variables will help in financial assessment and risk profiling. "
         "For each computational variable, provide an 'id' (unique string), 'name' (display name), "
         "'var_name' (snake_case), 'priority' (integer, 1 being highest), 'description', "
         "and a 'formula' (a JavaScript string using the `var_name` of provided assessment variables, "
         "e.g., 'assessment_var1 * assessment_var2'). "
         "Also specify the 'type' of the computational variable's result (e.g., 'float', 'int')."
        ),
        ("human",
         "Given the following assessment variables: {assessment_var_names}. "
         "Based on the overall prompt: '{prompt}', identify critical computational variables for "
         "financial assessment of a small business. Provide clear formulas using the provided assessment variable names. "
         "Examples of formulas could be 'monthly_revenue - monthly_operating_costs' for 'net_monthly_profit', "
         "or 'avg_daily_customers * avg_transaction_value * 30' for 'estimated_monthly_revenue'."
        )
    ]
)

# --- Prompt 3: Questionnaire Generation ---
QUESTIONNAIRE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant specialized in designing user interview questionnaires for financial assessments "
         "of subprime small business owners. "
         "Your task is to create a structured survey (JSON format expected) that effectively captures data "
         "for the provided assessment variables. "
         "Try generating atleast 15-20 questions, ensuring a mix of core and conditional questions. "
         "Also make sure to generate optional sections that can be conditionally displayed based on user responses. "
         "The questionnaire should be divided into logical 'sections'. Each section must have: "
         "- 'title': A clear title for the section. "
         "- 'description': A brief explanation of the section's purpose. "
         "- 'order': An integer indicating the display order. "
         "- 'is_mandatory': Boolean (true/false) indicating if the section is always shown. "
         "- 'rationale': Explanation for why this section is important. "
         "- 'core_questions': An array of Question objects that are always asked if the section is shown. "
         "- 'conditional_questions': An array of Question objects that appear based on 'triggering_criteria'. "
         "- 'triggering_criteria': (Optional) A JavaScript function body string (e.g., 'return q_previous_answer === \"Yes\";') for the section itself. "
         "  If 'is_mandatory' is true, this should be `null`. If 'is_mandatory' is false, this MUST be a **complex, intelligent, and realistic** JavaScript function string. This cannot be `null` or a simplistic `return true;`. "
         "- 'data_validation': A JavaScript function body string for overall section validation (e.g., 'return q_field1 > 0;'). Default to 'return true;'. "
         "\n"
         "Each 'Question' object must have: "
         "- 'id': Unique string. "
         "- 'text': The actual question. "
         "- 'type': Expected input type ('text', 'integer', 'float', 'boolean', 'dropdown'). "
         "- 'variable_name': A unique snake_case string for the question's answer (e.g., 'q_business_age'). "
         "- 'triggering_criteria': (Optional) A JavaScript function body string. If the question is in 'core_questions', this should be `null`. If in 'conditional_questions', this MUST be a **complex, intelligent, and realistic** JavaScript function string, using 'q_' prefixed variable names from earlier questions (e.g., 'return q_has_dependents === true;'). This cannot be `null` or a simplistic `return true;`. "
         "- 'assessment_variables': An array of `var_name`s of assessment variables this question helps capture. "
         "- 'formula': A JavaScript function body string (e.g., 'return parseFloat(q_daily_sales) * 30;') describing how the question's raw answer contributes to an assessment variable, or performs a relevant calculation. This formula is for the *question's contribution*, not for an assessment variable's final calculation."
         "- 'is_conditional': (Optional) Boolean indicating if the question is conditional. If question_triggering_criteria is provided, this should be true. "
         "\n"
         "Ensure `variable_name` for questions are prefixed with 'q_'. "
         "Prioritize a logical flow for the questionnaire. Use the provided `assessment_vars_json` as the source of truth for variables to cover. "
         "Make sure to cover all relevant assessment variables with at least one question. "
         "The output should be a JSON object conforming to the `QuestionnaireOutput` schema."
        ),
        ("human",
         "Generate a detailed interview questionnaire for a small business owner based on the primary prompt: '{prompt_context}'. "
         "The questions should aim to gather data for the following assessment variables:\n{assessment_vars_json}\n\n"
         "Structure the questionnaire into logical sections with both core and conditional questions, "
         "ensuring all conditional logic (triggering_criteria) for sections and questions are **meaningful JavaScript expressions** using `q_` prefixed question variables. "
         "Also, provide appropriate JavaScript formulas for each question to process its raw input. "
         "The response should be a JSON object." # Added "The response should be a JSON object."
        )
    ]
)

# --- Prompt 4: Variable Modifications ---
VARIABLE_MODIFICATIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant tasked with modifying lists of assessment and computational variables. "
         "You will receive the current lists and a user's modification request. "
         "Your response should be a JSON object detailing additions, updates, and removals for both types of variables. "
         "For 'updated_assessment_variables' and 'updated_computational_variables', provide the 'id' and only the fields that need to be changed. "
         "For 'added_assessment_variables' and 'added_computational_variables', provide all required fields ('id', 'name', 'var_name', 'priority', 'description', 'type', and 'formula' for computational). "
         "For computational variables, ensure new or updated 'formula' fields are valid JavaScript strings referencing existing assessment variables (e.g., 'var1 + var2'). "
         "The output should be a JSON object conforming to the `VariableModificationsOutput` schema."
        ),
        ("human",
         "Current Assessment Variables:\n{assessment_vars_json}\n\n"
         "Current Computational Variables:\n{computational_vars_json}\n\n"
         "User's modification request: {modification_prompt}\n\n"
         "Please provide a JSON object with the requested modifications."
        )
    ]
)

# --- Prompt 5: Questionnaire Modifications ---
QUESTIONNAIRE_MODIFICATIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI assistant specialized in modifying existing survey questionnaires. "
         "You will receive the current questionnaire structure (JSON format), a user's modification prompt, "
         "and the current assessment variables for context. "
         "Your task is to generate a JSON object (conforming to `QuestionnaireModificationsOutput` schema) "
         "specifying changes to sections and questions. "
         "\n"
         "You can: "
         "- `added_sections`: Add new sections. Provide full section objects. Ensure new sections have unique 'order' numbers. "
         "- `updated_sections`: Update existing sections. Provide 'order' and fields to change. "
         "  If `is_mandatory` is false, `triggering_criteria` MUST be a **complex, intelligent, and realistic** JavaScript function string (not null or 'return true;'). "
         "- `removed_section_orders`: Remove sections by their 'order' number. "
         "- `added_questions`: Add new questions to specific sections. Provide `section_order`, `is_core` (boolean), and a full `question` object. "
         "- `updated_questions`: Update existing questions by their `id`. Provide `id` and fields to change. "
         "  If a question becomes conditional (`is_core` changes to false or moved to `conditional_questions`), its `triggering_criteria` MUST be a **complex, intelligent, and realistic** JavaScript function string. "
         "  'formula' should always be a JavaScript expression."
         "- `removed_question_variable_names`: Remove questions by their `variable_name`. "
         "\n"
         "Ensure all IDs are unique. Ensure `variable_name` for questions are prefixed with 'q_'. "
         "When adding or updating conditional sections or questions, ensure their `triggering_criteria` are **meaningful JavaScript expressions** that logically relate to other `q_` prefixed question variables from the current questionnaire context. "
         "The response should be a JSON object specifying these modifications."
        ),
        ("human",
         "Current Questionnaire:\n{questionnaire_json}\n\n"
         "Current Assessment Variables (for context):\n{assessment_vars_json}\n\n"
         "User's modification request: {modification_prompt}\n\n"
         "Please provide a JSON object detailing the requested modifications to the questionnaire."
        )
    ]
)

# --- Prompt for JS Expression Refinement (triggered by _refine_js_expression) ---
JS_REFINEMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert JavaScript developer specializing in dynamic form logic for financial assessments. "
         "You need to generate a **complex and intelligent** JavaScript expression for a '{expression_type}' field. "
         "This expression must be a valid JavaScript function body (e.g., 'return q_var1 > 0 && q_var2 === \"Yes\";'). "
         "It must use variable names that start with 'q_' and are from the provided `context_question_vars`. "
         "Avoid simplistic 'return true;' or empty expressions. "
         "The purpose is to create a dynamic condition for '{target_entity_description}'. "
         "Available question variables: {context_question_vars}. "
         "Generate only the JavaScript expression within a JSON object, with the key 'expression', nothing else. For example: `{{\"expression\": \"return q_var1 > 0 && q_var2 === \\\"Yes\\\";\"}}`. Provide a realistic and smart trigger for a financial assessment context. "
         "Consider using logical operators (&&, ||, !), numerical comparisons (>, <, >=, <=, ===), "
         "string comparisons, or checking for specific values. "
         "Examples for triggers: 'return q_has_dependents === true && q_income_source === \"Self-employed\";', "
         "'return q_business_type.includes(\"online\") || q_monthly_sales > 5000;'. "
        ),
        ("human",
         "Generate a suitable JavaScript expression for the '{expression_type}' of '{target_entity_description}'. "
         "It should be intelligent and meaningful in the context of financial assessment conditional logic. "
         "If '{expression_type}' is a formula, it should describe how the question's answer contributes to an assessment variable, or perform a relevant calculation. "
         "If '{expression_type}' is a triggering criteria, it should determine when a section or question is displayed."
        )
    ]
)
