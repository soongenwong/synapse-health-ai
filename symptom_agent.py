import dspy
import os
import asyncio
import pandas as pd
from typing import List
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from rapidfuzz import fuzz

APP_NAME = "stock_app"
USER_ID = "1234"
SESSION_ID = "session1234"
# gemini_api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_api_key
llm = dspy.LM('gemini/gemini-2.5-flash-lite-preview-06-17', api_key=gemini_api_key)
dspy.configure(lm=llm)

def get_relevant_diseases(symptoms: List[str], threshold: int = 70) -> List[dict]:
    """
    Returns a list of dicts: [{'disease': ..., 'symptoms': [...]}, ...]
    """
    df = pd.read_csv('/Users/zhuoy1/Documents/machine_talents_hack/logo_1753104563084.csv')
    results = []
    for _, row in df.iterrows():
        disease = row['Disease']
        row_symptoms = [str(row[f'Symptom_{i}']).strip().lower()
                        for i in range(1, 18)
                        if pd.notna(row.get(f'Symptom_{i}')) and str(row[f'Symptom_{i}']).strip()]
        match_count = 0
        for user_sym in symptoms:
            for s in row_symptoms:
                if fuzz.partial_ratio(user_sym.lower(), s) >= threshold:
                    match_count += 1
                    break
        if match_count > 0:
            results.append({'disease': disease, 'symptoms': row_symptoms, 'match_count': match_count})
    # Sort by match_count descending
    results.sort(key=lambda x: x['match_count'], reverse=True)
    return results[:10]  # Return top 10 matches

class SymptomExtractor(dspy.Signature):
    """Extract symptoms from a string"""
    text: str = dspy.InputField(desc="Input text containing symptoms")
    symptoms: List[str] = dspy.OutputField(desc="List of extracted symptoms")

def extract_symptoms_from_text(text):
    symptom_extractor = dspy.Predict(SymptomExtractor)
    result = symptom_extractor(text=text)
    return result.symptoms

symptom_triage_agent = Agent(
    model='gemini-2.5-flash-lite-preview-06-17',
    name='symptom_triage_agent',
    instruction=(
        "You are a medical triage agent. Given a list of patient symptoms, "
        "use the provided tool to retrieve the most relevant diseases from the medical database. "
        "Rank the diseases by likelihood and explain your reasoning. "
        "If no relevant disease is found, say so."
    ),
    description=(
        "This agent specializes in mapping patient symptoms to likely diseases using a medical CSV database. "
        "It uses a fuzzy-matching tool to find and rank possible conditions."
    ),
    tools=[get_relevant_diseases],  # This function will be wrapped as a tool
)

async def setup_session_and_runner():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME, 
        user_id=USER_ID, 
        session_id=SESSION_ID
    )
    runner = Runner(
        agent=symptom_triage_agent, 
        app_name=APP_NAME, 
        session_service=session_service
    )
    return session, runner

# Agent Interaction
async def call_triage_agent_async(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    session, runner = await setup_session_and_runner()
    events = runner.run_async(
        user_id=USER_ID, 
        session_id=SESSION_ID, 
        new_message=content
    )

    async for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Triage Agent Response: ", final_response)
            return final_response

test_queries = [
    "I have a high fever, severe headache, and joint pain for 3 days",
    "I'm experiencing nausea, vomiting, and abdominal pain",
    "I have itching and skin rash on my arms that started yesterday"
]

# Run the tests
if __name__ == "__main__":
    asyncio.run(call_triage_agent_async("I have a high fever, severe headache, and joint pain for 3 days"))
# class DiseasePredictor(dspy.Signature):
#     """Predict most likely diseases based on symptoms and medical database"""
#     symptoms: List[str] = dspy.InputField(desc="List of symptoms")
#     disease_symptoms: str= dspy.InputField(desc="CSV string of disease symptoms")
#     predicted_diseases: List[str] = dspy.OutputField(desc="List of predicted diseases")
#     explanations: List[str] = dspy.OutputField(desc="Explanations for each predicted disease")
#     confidence_scores: List[float] = dspy.OutputField(desc="Confidence scores for each predicted disease")


# def predict_diseases_from_symptoms(symptoms, symptom_disease_df):
#     """Use LLM to predict diseases based on symptoms and CSV data"""
    
#     # Convert CSV to a readable format for the LLM
#     disease_data = []
#     for _, row in symptom_disease_df.iterrows():
#         disease = row['Disease']
#         symptoms_list = [str(row[f'Symptom_{i}']) for i in range(1, 18) 
#                         if pd.notna(row[f'Symptom_{i}']) and str(row[f'Symptom_{i}']).strip()]
#         if symptoms_list:
#             disease_data.append(f"{disease}: {', '.join(symptoms_list)}")
    
#     # Create a formatted string of the disease-symptom database
#     disease_symptom_data = "\n".join(disease_data[:50])  # Limit to avoid token limits
    
#     # Convert symptoms list to comma-separated string
#     symptoms_str = ", ".join(symptoms) if isinstance(symptoms, list) else str(symptoms)
    
#     # Use DSPy to predict diseases
#     disease_predictor = dspy.Predict(DiseasePredictor)
#     result = disease_predictor(
#         symptoms=symptoms_str,
#         disease_symptom_data=disease_symptom_data
#     )
    
#     return {
#         "predicted_diseases": result.predicted_diseases,
#         "explanations": result.explanations,
#         "confidence_scores": result.confidence_scores
#     }

# def symptom_extraction_disease_mapping_agent(user_text):
#     """Complete agent that extracts symptoms and predicts diseases"""
    
#     # Step 1: Extract symptoms
#     symptoms = extract_symptoms_from_text(user_text)
    
#     # Step 2: Predict diseases using LLM
#     disease_prediction = predict_diseases_from_symptoms(symptoms, symptom_disease_df)
    
#     return {
#         "symptoms": symptoms,
#         "predicted_diseases": disease_prediction["predicted_diseases"],
#         "explanations": disease_prediction["explanations"],
#         "confidence_scores": disease_prediction["confidence_scores"]
#     }

# if __name__ == "__main__":
#     user_input = "I have a high fever, headache, and joint pain for 3 days"
#     result = symptom_extraction_disease_mapping_agent(user_input)
    
#     print("User Input:", user_input)
#     print("Extracted Symptoms:", result["symptoms"])
#     print("Predicted Diseases:", result["predicted_diseases"])
#     print("Confidence Scores:", result["confidence_scores"])


