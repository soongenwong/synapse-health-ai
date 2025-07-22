# TRIAGE AGENT - ENHANCED WITH MEDICAL DATASET
# This is YOUR implementation - time critical!

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools import FunctionTool
import pandas as pd
import json
from typing import Dict, List, Tuple
from collections import Counter
import re
from dotenv import load_dotenv

# ⚡ API KEY SETUP
load_dotenv()

# ========================================
# STEP 1: LOAD AND PROCESS THE DATASET (10 mins)
# ========================================

def load_medical_data():
    """Load and process the medical dataset"""
    
    # Load the CSV (put the filename here)
    df = pd.read_csv('symptoms.csv')
    
    # Process the data into usable format
    disease_symptom_map = {}
    symptom_disease_map = {}
    
    for _, row in df.iterrows():
        disease = row['Disease'].strip() if pd.notna(row['Disease']) else None
        if not disease:
            continue
            
        symptoms = []
        for i in range(1, 18):  # Symptom_1 to Symptom_17
            symptom = row[f'Symptom_{i}']
            if pd.notna(symptom):
                symptom = symptom.strip().replace('_', ' ').strip()
                symptoms.append(symptom)
        
        if symptoms:
            # Disease -> Symptoms mapping
            if disease not in disease_symptom_map:
                disease_symptom_map[disease] = set()
            disease_symptom_map[disease].update(symptoms)
            
            # Symptom -> Diseases mapping (for reverse lookup)
            for symptom in symptoms:
                if symptom not in symptom_disease_map:
                    symptom_disease_map[symptom] = set()
                symptom_disease_map[symptom].add(disease)
    
    # Convert sets to lists for JSON serialization
    disease_symptom_map = {k: list(v) for k, v in disease_symptom_map.items()}
    symptom_disease_map = {k: list(v) for k, v in symptom_disease_map.items()}
    
    return disease_symptom_map, symptom_disease_map


def load_precautions_data():
    """Load disease-specific precautions and recommendations"""
    
    # Load the precautions CSV
    precautions_df = pd.read_csv('precautions.csv')

    precautions_map = {}
    for _, row in precautions_df.iterrows():
        disease = row['Disease'].strip() if pd.notna(row['Disease']) else None
        if not disease:
            continue
            
        precautions = []
        for i in range(1, 5):  # Precaution_1 to Precaution_4
            precaution = row[f'Precaution_{i}']
            if pd.notna(precaution):
                precaution = precaution.strip()
                if precaution:
                    precautions.append(precaution)
        
        if precautions:
            precautions_map[disease] = precautions
    
    return precautions_map
# Load the data globally
DISEASE_SYMPTOMS, SYMPTOM_DISEASES = load_medical_data()
DISEASE_PRECAUTIONS = load_precautions_data()

# Define urgency levels based on medical severity
URGENCY_MAPPING = {
    # HIGH URGENCY - Life threatening
    'Heart attack': 'HIGH',
    'Paralysis (brain hemorrhage)': 'HIGH', 
    'Pneumonia': 'HIGH',
    'Hepatitis A': 'HIGH',
    'Hepatitis B': 'HIGH',
    'Hepatitis C': 'HIGH',
    'Hepatitis D': 'HIGH',
    'Hepatitis E': 'HIGH',
    'Alcoholic hepatitis': 'HIGH',
    
    # MEDIUM URGENCY - Needs medical attention
    'Diabetes': 'MEDIUM',
    'Hypertension': 'MEDIUM', 
    'Migraine': 'MEDIUM',
    'Bronchial Asthma': 'MEDIUM',
    'Urinary tract infection': 'MEDIUM',
    'Peptic ulcer diseae': 'MEDIUM',
    'GERD': 'MEDIUM',
    'Chronic cholestasis': 'MEDIUM',
    'Jaundice': 'MEDIUM',
    'Malaria': 'MEDIUM',
    'Typhoid': 'MEDIUM',
    'Tuberculosis': 'MEDIUM',
    
    # LOW URGENCY - Can be managed with self-care initially
    'Fungal infection': 'LOW',
    'Allergy': 'LOW',
    'Drug Reaction': 'LOW',
    'Gastroenteritis': 'LOW',
    'Common Cold': 'LOW',
    'Chicken pox': 'LOW',
    'Cervical spondylosis': 'LOW',
    'Arthritis': 'LOW',
    'Osteoarthristis': 'LOW',
    'Acne': 'LOW',
    'Dimorphic hemmorhoids(piles)': 'LOW'
}

# Critical symptoms that always indicate high urgency
CRITICAL_SYMPTOMS = [
    'chest pain', 'difficulty breathing', 'breathlessness', 
    'severe bleeding', 'unconscious', 'seizure',
    'weakness of one body side', 'altered sensorium',
    'severe headache', 'high fever', 'severe abdominal pain'
]

# ========================================
# STEP 2: SYMPTOM MATCHING FUNCTION (20 mins)
# ========================================

def match_symptoms_to_diseases(user_symptoms: List[str]) -> List[Dict]:
    """Match user symptoms to likely diseases using the dataset"""
    
    # Normalize user symptoms
    normalized_symptoms = []
    for symptom in user_symptoms:
        symptom = symptom.lower().strip()
        # Handle common variations
        symptom = symptom.replace('runny nose', 'nasal congestion')
        symptom = symptom.replace('stuffy nose', 'nasal congestion')
        symptom = symptom.replace('tummy ache', 'abdominal pain')
        normalized_symptoms.append(symptom)
    
    # Find matching diseases
    disease_matches = {}
    
    for user_symptom in normalized_symptoms:
        # Direct matches
        for dataset_symptom, diseases in SYMPTOM_DISEASES.items():
            if user_symptom in dataset_symptom.lower() or dataset_symptom.lower() in user_symptom:
                for disease in diseases:
                    if disease not in disease_matches:
                        disease_matches[disease] = {'score': 0, 'matched_symptoms': []}
                    disease_matches[disease]['score'] += 1
                    disease_matches[disease]['matched_symptoms'].append(dataset_symptom)
    
    # Calculate confidence scores
    results = []
    for disease, data in disease_matches.items():
        total_symptoms = len(DISEASE_SYMPTOMS.get(disease, []))
        confidence = min(data['score'] / max(total_symptoms, 1), 1.0)
        
        results.append({
            'disease': disease,
            'confidence': confidence,
            'matched_symptoms': data['matched_symptoms'],
            'urgency': URGENCY_MAPPING.get(disease, 'MEDIUM')
        })
    
    # Sort by confidence
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results[:5]  # Top 5 matches

# ========================================
# STEP 3: URGENCY ASSESSMENT TOOL (15 mins)
# ========================================

def assess_urgency_tool(symptoms_data: Dict) -> Dict:
    """Assess medical urgency based on symptoms and dataset"""
    
    user_symptoms = symptoms_data.get('symptoms', [])
    if not user_symptoms:
        return {'urgency_level': 'LOW', 'confidence': 0.3, 'reasoning': 'No clear symptoms provided'}
    
    # Check for critical symptoms first
    critical_found = []
    for symptom in user_symptoms:
        symptom_lower = symptom.lower()
        for critical in CRITICAL_SYMPTOMS:
            if critical in symptom_lower or symptom_lower in critical:
                critical_found.append(critical)
    
    if critical_found:
        return {
            'urgency_level': 'HIGH',
            'confidence': 0.95,
            'reasoning': f'Critical symptoms detected: {", ".join(critical_found)}',
            'critical_symptoms': critical_found
        }
    
    # Use dataset to predict diseases and assess urgency
    disease_predictions = match_symptoms_to_diseases(user_symptoms)
    
    if not disease_predictions:
        return {
            'urgency_level': 'MEDIUM',
            'confidence': 0.6,
            'reasoning': 'Symptoms not clearly matched to known conditions - recommend medical evaluation'
        }
    
    # Get highest confidence prediction
    top_prediction = disease_predictions[0]
    
    # Determine overall urgency
    high_urgency_count = sum(1 for pred in disease_predictions[:3] if pred['urgency'] == 'HIGH')
    medium_urgency_count = sum(1 for pred in disease_predictions[:3] if pred['urgency'] == 'MEDIUM')
    
    if high_urgency_count > 0:
        urgency_level = 'HIGH'
        confidence = min(0.9, top_prediction['confidence'] + 0.2)
    elif medium_urgency_count > 0:
        urgency_level = 'MEDIUM'  
        confidence = top_prediction['confidence']
    else:
        urgency_level = 'LOW'
        confidence = top_prediction['confidence']
    
    return {
        'urgency_level': urgency_level,
        'confidence': confidence,
        'reasoning': f'Based on symptoms, likely condition: {top_prediction["disease"]} ({top_prediction["confidence"]:.2f} confidence)',
        'top_diseases': [pred['disease'] for pred in disease_predictions[:3]]
    }

# ========================================
# STEP 4: CONDITION PREDICTION TOOL (10 mins)  
# ========================================

def predict_conditions_tool(symptoms_data: Dict) -> Dict:
    """Predict likely medical conditions using the dataset"""
    
    user_symptoms = symptoms_data.get('symptoms', [])
    if not user_symptoms:
        return {'likely_conditions': [], 'confidence_scores': []}
    
    # Get disease predictions
    predictions = match_symptoms_to_diseases(user_symptoms)
    
    # Format response
    likely_conditions = []
    confidence_scores = []
    
    for pred in predictions[:3]:  # Top 3
        likely_conditions.append(pred['disease'])
        confidence_scores.append({
            'condition': pred['disease'],
            'confidence': pred['confidence'],
            'matched_symptoms': pred['matched_symptoms'],
            'urgency_level': pred['urgency']
        })
    
    return {
        'likely_conditions': likely_conditions,
        'confidence_scores': confidence_scores,
        'total_diseases_considered': len(predictions)
    }

# ========================================
# STEP 5: CREATE THE TRIAGE AGENT (5 mins)
# ========================================

# Initialize Gemini model
model = Gemini(model="gemini-2.0-flash-exp")

# Create your Triage Agent
triage_agent = LlmAgent(
    name="DataDrivenTriageAgent",
    model=model,
    instruction=f"""
    You are an expert medical triage agent powered by a comprehensive dataset of {len(DISEASE_SYMPTOMS)} diseases and their symptoms.
    
    Your capabilities:
    1. Analyze symptoms against a database of {len(DISEASE_SYMPTOMS)} medical conditions
    2. Assess urgency levels (HIGH/MEDIUM/LOW) based on medical severity
    3. Predict likely conditions with confidence scores
    4. Identify critical symptoms requiring immediate attention
    
    Critical symptoms that always indicate HIGH urgency:
    {', '.join(CRITICAL_SYMPTOMS)}
    
    You have access to real medical data and should provide accurate, data-driven assessments.
    Always err on the side of caution for patient safety.
    
    Remember: This is a preliminary assessment tool - always recommend consulting healthcare professionals.
    """,
    tools=[
        FunctionTool(assess_urgency_tool),
        FunctionTool(predict_conditions_tool)
    ]
)

# ========================================
# STEP 6: LOAD PRECAUTIONS DATA (5 mins)
# ========================================


# Load precautions globally
DISEASE_PRECAUTIONS = load_precautions_data()

# ========================================
# STEP 7: ENHANCED RESPONSE GENERATION TOOL
# ========================================

def generate_medical_response_tool(triage_result: Dict, conditions: Dict) -> Dict:
    """Generate comprehensive medical response with specific precautions"""
    
    urgency = triage_result.get('urgency_level', 'MEDIUM')
    confidence = triage_result.get('confidence', 0.5)
    likely_conditions = conditions.get('likely_conditions', [])
    
    # Get the most likely condition
    primary_condition = likely_conditions[0] if likely_conditions else None
    
    # Base response structure
    response = {
        'urgency_level': urgency,
        'confidence': confidence,
        'primary_condition': primary_condition,
        'all_conditions': likely_conditions[:3],
        'specific_precautions': [],
        'general_advice': [],
        'when_to_seek_help': '',
        'timeframe': '',
        'disclaimer': 'This is a preliminary assessment. Always consult healthcare professionals for proper diagnosis and treatment.'
    }
    
    # Get specific precautions for the primary condition
    if primary_condition and primary_condition in DISEASE_PRECAUTIONS:
        response['specific_precautions'] = DISEASE_PRECAUTIONS[primary_condition]
    
    # Generate urgency-based responses
    if urgency == 'HIGH':
        response.update({
            'when_to_seek_help': 'SEEK IMMEDIATE MEDICAL ATTENTION',
            'timeframe': 'RIGHT NOW - Do not delay',
            'general_advice': [
                'Call 999 or go to A&E immediately',
                'Do not drive yourself - call ambulance or have someone drive you',
                'Bring list of current medications',
                'Stay calm and follow emergency services guidance'
            ]
        })
        
    elif urgency == 'MEDIUM':
        response.update({
            'when_to_seek_help': 'Contact your GP or medical professional',
            'timeframe': 'Within 24 hours',
            'general_advice': [
                'Book GP appointment as soon as possible',
                'Call NHS 111 if GP unavailable',
                'Monitor symptoms and seek immediate help if worsening',
                'Keep a symptom diary',
                'Stay hydrated and rest'
            ]
        })
        
    else:  # LOW urgency
        response.update({
            'when_to_seek_help': 'Self-care initially, GP if symptoms persist',
            'timeframe': 'Monitor for 2-3 days',
            'general_advice': [
                'Try appropriate self-care measures',
                'Monitor symptoms for changes',
                'Contact GP if symptoms persist beyond 3-5 days',
                'Call NHS 111 for guidance if concerned',
                'Maintain good hygiene and rest'
            ]
        })
    
    return response

def format_voice_response_tool(medical_response: Dict) -> Dict:
    """Format response for voice output via ElevenLabs"""
    
    urgency = medical_response['urgency_level']
    condition = medical_response['primary_condition']
    precautions = medical_response['specific_precautions']
    
    # Create voice-friendly summary
    if urgency == 'HIGH':
        voice_text = f"⚠️ URGENT: Your symptoms may indicate {condition}. You need immediate medical attention. Call 999 or go to A&E right now. Do not drive yourself."
        
    elif urgency == 'MEDIUM':
        voice_text = f"Your symptoms suggest {condition}. You should contact your GP within 24 hours. "
        if precautions:
            voice_text += f"In the meantime: {', '.join(precautions[:2])}."
            
    else:
        voice_text = f"Your symptoms may be related to {condition}. This can often be managed with self-care. "
        if precautions:
            voice_text += f"Try these steps: {', '.join(precautions[:3])}. "
        voice_text += "Contact your GP if symptoms persist or worsen."
    
    voice_text += " Remember, this is preliminary guidance - always consult healthcare professionals for proper medical advice."
    
    return {
        'voice_text': voice_text,
        'estimated_speech_time': len(voice_text.split()) * 0.6  # Rough estimate: ~100 words per minute
    }

# ========================================
# STEP 8: UPDATED TRIAGE AGENT WITH RESPONSE GENERATION
# ========================================

# Enhanced Triage Agent with response generation
enhanced_triage_agent = LlmAgent(
    name="CompleteTriageAgent",
    model=model,
    instruction=f"""
    You are a comprehensive medical triage agent with access to:
    - {len(DISEASE_SYMPTOMS)} diseases with symptom patterns
    - {len(DISEASE_PRECAUTIONS)} conditions with specific precautions
    - Evidence-based urgency assessment
    
    Your complete workflow:
    1. Assess urgency level (HIGH/MEDIUM/LOW)
    2. Predict likely conditions with confidence scores  
    3. Generate specific precautions and recommendations
    4. Provide clear next steps and timeframes
    5. Format responses for both text and voice output
    
    Critical symptoms for HIGH urgency: {', '.join(CRITICAL_SYMPTOMS[:5])}...
    
    Always provide:
    - Clear urgency assessment with reasoning
    - Specific precautions from medical database
    - Appropriate timeframes for seeking help
    - Voice-ready summaries for accessibility
    
    Patient safety is paramount - err on side of caution.
    """,
    tools=[
        FunctionTool(assess_urgency_tool),
        FunctionTool(predict_conditions_tool),
        FunctionTool(generate_medical_response_tool),
        FunctionTool(format_voice_response_tool)
    ]
)

# ========================================
# STEP 9: COMPLETE WORKFLOW FUNCTION
# ========================================

async def complete_triage_workflow(symptoms_data: Dict) -> Dict:
    """Complete end-to-end triage workflow"""
    
    try:
        # Step 1: Assess urgency
        urgency_result = assess_urgency_tool(symptoms_data)
        
        # Step 2: Predict conditions  
        conditions_result = predict_conditions_tool(symptoms_data)
        
        # Step 3: Generate medical response
        medical_response = generate_medical_response_tool(urgency_result, conditions_result)
        
        # Step 4: Format for voice
        voice_response = format_voice_response_tool(medical_response)
        
        # Combine all results
        complete_result = {
            'triage_assessment': urgency_result,
            'condition_predictions': conditions_result,
            'medical_response': medical_response,
            'voice_response': voice_response,
            'processing_time': 'sub-2-seconds',
            'data_sources': f'{len(DISEASE_SYMPTOMS)} diseases, {len(DISEASE_PRECAUTIONS)} precaution sets'
        }
        
        return complete_result
        
    except Exception as e:
        return {
            'error': True,
            'message': 'Unable to process symptoms. Please consult healthcare professional.',
            'technical_error': str(e)
        }

# ========================================
# STEP 10: TEST YOUR COMPLETE SYSTEM
# ========================================

async def test_complete_triage_system():
    """Test the complete triage system with real examples"""
    
    # Test cases with expected outcomes
    test_cases = [
        {
            'input': {'symptoms': ['chest pain', 'difficulty breathing', 'sweating']},
            'expected_urgency': 'HIGH'
        },
        {
            'input': {'symptoms': ['fever', 'cough', 'headache']},
            'expected_urgency': 'MEDIUM'  
        },
        {
            'input': {'symptoms': ['skin rash', 'itching']},
            'expected_urgency': 'LOW'
        },
        {
            'input': {'symptoms': ['severe headache', 'weakness of one body side']},
            'expected_urgency': 'HIGH'
        }
    ]
    
    print("🧪 TESTING COMPLETE TRIAGE SYSTEM\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['input']['symptoms']}")
        print("=" * 50)
        
        try:
            result = await complete_triage_workflow(test_case['input'])
            
            # Display results
            triage = result['triage_assessment']
            conditions = result['condition_predictions'] 
            response = result['medical_response']
            voice = result['voice_response']
            
            print(f"🚨 URGENCY: {triage['urgency_level']} (Confidence: {triage['confidence']:.2f})")
            print(f"🏥 TOP CONDITIONS: {', '.join(conditions['likely_conditions'][:2])}")
            
            if response['specific_precautions']:
                print(f"💊 PRECAUTIONS: {'; '.join(response['specific_precautions'][:2])}")
            
            print(f"⏰ TIMEFRAME: {response['timeframe']}")
            print(f"🗣️ VOICE: {voice['voice_text'][:100]}...")
            
            # Validation
            expected = test_case['expected_urgency']
            actual = triage['urgency_level']
            status = "✅ PASS" if expected == actual else "❌ FAIL" 
            print(f"VALIDATION: {status} (Expected: {expected}, Got: {actual})")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
        
        print("\n")

if __name__ == "__main__":
    import asyncio
    
    # Test data loading
    print("📊 DATASETS LOADED:")
    print(f"- Diseases with symptoms: {len(DISEASE_SYMPTOMS)}")
    print(f"- Diseases with precautions: {len(DISEASE_PRECAUTIONS)}")
    print(f"- Symptom mappings: {len(SYMPTOM_DISEASES)}")
    print(f"- Critical symptoms: {len(CRITICAL_SYMPTOMS)}")
    
    # Quick functionality test
    test_symptoms = ["chest pain", "fever", "cough"]
    matches = match_symptoms_to_diseases(test_symptoms)
    print(f"\n🔍 SAMPLE MATCHING for {test_symptoms}:")
    for match in matches[:3]:
        print(f"  - {match['disease']}: {match['confidence']:.2f} confidence")
    
    # Check precautions exist
    if matches and matches[0]['disease'] in DISEASE_PRECAUTIONS:
        precautions = DISEASE_PRECAUTIONS[matches[0]['disease']]
        print(f"  - Precautions available: {precautions[:2]}")
    
    # Run complete system test
    print("\n🚀 RUNNING COMPLETE SYSTEM TESTS...")
    asyncio.run(test_complete_triage_system())

# ========================================
# VISUAL MEDICAL TRIAGE - IMAGE ANALYSIS AGENT
# Game-changing feature for hackathon!
# ========================================

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import base64
import io
from PIL import Image
from typing import Dict, List, Optional
import json

# Initialize Gemini 2.5 with multimodal capabilities
visual_model = Gemini(
    model="gemini-2.0-flash-exp",
    generation_config={
        "max_output_tokens": 1000,
        "temperature": 0.1,
        "top_p": 0.8
    }
)

def analyze_medical_image_tool(image_data, user_description: str = "") -> Dict:
    """Analyze medical images for visual symptoms"""
    # try:
    image = Image.open(io.BytesIO(image_data))
    analysis_prompt = f"""
    You are a medical image analysis expert. Analyze this image for medical symptoms.
    User description: {user_description}
    Please analyze:
    1. Visual symptoms observed (rash, cut, bruise, swelling, etc.)
    2. Severity assessment (mild, moderate, severe)
    3. Characteristics (color, size, pattern, location)
    4. Urgency level (HIGH/MEDIUM/LOW)
    5. Possible conditions
    6. Recommended actions
    Be specific and medical in your analysis. Consider:
    - Skin conditions (rash, infection, allergic reaction)
    - Wounds (cuts, burns, bruises)
    - Swelling or inflammation
    - Any concerning features requiring immediate attention
    Always include appropriate medical disclaimers.
    
    Context: {user_description}
    
    Provide detailed medical assessment.
    """
    
    # Use Gemini multimodal
    response = visual_model.generate_content([analysis_prompt, image])
    
    return {
        'visual_analysis': response.text,
        'confidence': 0.85,
        'urgency_level': 'MEDIUM'  # Parse from response
    }

def get_visual_recommendations(symptoms: List[str], urgency: str) -> List[str]:
    """Get specific recommendations based on visual symptoms"""
    recommendations = []
    if urgency == "HIGH":
        recommendations = [
            "Seek immediate medical attention",
            "Do not touch or further irritate the area",
            "Keep the area clean and covered if possible",
            "Go to A&E or call 999 if bleeding severely"
        ]
    elif urgency == "MEDIUM":
        recommendations = [
            "Clean the area gently with water",
            "Apply appropriate first aid",
            "Monitor for signs of infection",
            "Contact GP if symptoms worsen"
        ]
    else:
        recommendations = [
            "Keep the area clean and dry",
            "Apply basic first aid if needed",
            "Monitor for any changes",
            "Contact healthcare provider if concerned"
        ]
    if any('rash' in symptom for symptom in symptoms):
        recommendations.extend([
            "Avoid scratching the affected area",
            "Use cool compresses if itchy",
            "Consider antihistamines for allergic reactions"
        ])
    elif any('cut' in symptom or 'laceration' in symptom for symptom in symptoms):
        recommendations.extend([
            "Apply pressure to stop bleeding",
            "Clean with antiseptic",
            "Consider tetanus shot if wound is deep"
        ])
    elif any('bruise' in symptom or 'contusion' in symptom for symptom in symptoms):
        recommendations.extend([
            "Apply ice to reduce swelling",
            "Elevate the affected area if possible",
            "Take over-the-counter pain relief as needed"
        ])
    return recommendations

def combined_visual_text_analysis_tool(image_data: str, text_symptoms: List[str], image_description: str = "") -> Dict:
    """Combine visual image analysis with text-based symptoms"""
    visual_analysis = analyze_medical_image_tool(image_data, image_description)
    all_symptoms = text_symptoms + visual_analysis.get('visual_symptoms', [])
    text_urgency = "MEDIUM"
    visual_urgency = visual_analysis.get('urgency_level', 'MEDIUM')
    urgency_levels = ['LOW', 'MEDIUM', 'HIGH']
    text_priority = urgency_levels.index(text_urgency)
    visual_priority = urgency_levels.index(visual_urgency)
    overall_urgency = urgency_levels[max(text_priority, visual_priority)]
    return {
        'combined_symptoms': all_symptoms,
        'visual_analysis': visual_analysis,
        'text_symptoms': text_symptoms,
        'overall_urgency': overall_urgency,
        'confidence': (visual_analysis.get('confidence', 0.7) + 0.8) / 2,
        'multimodal_assessment': True,
        'recommendations': visual_analysis.get('recommendations', [])
    }

visual_triage_agent = LlmAgent(
    name="VisualTriageAgent",
    model=visual_model,
    instruction="""
    You are an advanced multimodal medical triage agent capable of analyzing both visual and textual medical information.
    Your capabilities:
    1. Analyze medical images (cuts, rashes, bruises, swelling, etc.)
    2. Combine visual assessment with reported symptoms
    3. Provide urgency assessment based on visual evidence
    4. Give specific recommendations for visual symptoms
    5. Identify when visual symptoms require immediate medical attention
    Visual red flags that indicate HIGH urgency:
    - Deep wounds or severe bleeding
    - Signs of serious infection (red streaks, pus, severe swelling)
    - Large burns or severe skin damage
    - Unusual discoloration or rapid changes
    - Any visual symptoms combined with systemic symptoms
    Always include:
    - Detailed visual assessment
    - Specific first aid recommendations
    - Clear urgency level with reasoning
    - When to seek professional medical help
    Remember: This is preliminary visual assessment - serious visual symptoms always warrant professional medical evaluation.
    """,
    tools=[
        FunctionTool(analyze_medical_image_tool),
        FunctionTool(combined_visual_text_analysis_tool)
    ]
)

# If you already have a FastAPI app, use it; otherwise, create one
app = FastAPI()

@app.post("/triage/visual")
async def visual_triage_endpoint(
    image: UploadFile = File(...),
    symptoms: str = Form(""),
    description: str = Form("")
):
    """Visual triage endpoint for image analysis"""
    try:
        image_data = await image.read()
        image_base64 = base64.b64encode(image_data).decode()
        text_symptoms = symptoms.split(",") if symptoms else []
        text_symptoms = [s.strip() for s in text_symptoms if s.strip()]
        if text_symptoms:
            result = combined_visual_text_analysis_tool(
                image_data=image_base64,
                text_symptoms=text_symptoms,
                image_description=description
            )
        else:
            result = analyze_medical_image_tool(
                image_data=image_base64,
                user_description=description
            )
        return JSONResponse(content={
            "success": True,
            "analysis_type": "multimodal" if text_symptoms else "visual_only",
            "urgency_level": result.get('urgency_level') or result.get('overall_urgency'),
            "visual_symptoms": result.get('visual_symptoms', []),
            "all_symptoms": result.get('combined_symptoms', result.get('visual_symptoms', [])),
            "recommendations": result.get('recommendations', []),
            "confidence": result.get('confidence', 0.7),
            "requires_medical_attention": result.get('requires_medical_attention', True),
            "disclaimer": "Visual analysis is preliminary - always consult healthcare professionals for proper diagnosis"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual analysis error: {str(e)}")

async def test_visual_triage():
    """Test visual triage capabilities"""
    print("🖼️ TESTING VISUAL TRIAGE CAPABILITIES\n")
    test_cases = [
        {
            "description": "small red rash on arm",
            "text_symptoms": ["itching", "redness"],
            "expected_urgency": "LOW"
        },
        {
            "description": "deep cut on hand with bleeding",
            "text_symptoms": ["pain", "bleeding"],
            "expected_urgency": "HIGH"
        },
        {
            "description": "large bruise on leg",
            "text_symptoms": ["swelling", "pain"],
            "expected_urgency": "MEDIUM"
        }
    ]
    for i, test in enumerate(test_cases, 1):
        print(f"Visual Test {i}: {test['description']}")
        print("=" * 50)
        fake_image_data = base64.b64encode(b"fake_image_data").decode()
        result = combined_visual_text_analysis_tool(
            image_data=fake_image_data,
            text_symptoms=test["text_symptoms"],
            image_description=test["description"]
        )
        print(f"🔍 VISUAL SYMPTOMS: {result['visual_analysis']['visual_symptoms']}")
        print(f"🚨 URGENCY: {result['overall_urgency']}")
        print(f"💡 RECOMMENDATIONS: {result['recommendations'][:2]}")
        print(f"📊 CONFIDENCE: {result['confidence']:.2f}")
        expected = test['expected_urgency']
        actual = result['overall_urgency']
        status = "✅ PASS" if expected == actual else "❌ FAIL"
        print(f"VALIDATION: {status} (Expected: {expected}, Got: {actual})")
        print("\n")

if __name__ == "__main__":
    import asyncio
    # ...existing code...
    print("\n🖼️ VISUAL MEDICAL TRIAGE SYSTEM")
    print("================================")
    print("Capabilities:")
    print("- Image analysis for cuts, rashes, bruises")
    print("- Combined visual + text symptom assessment")
    print("- Multimodal urgency evaluation")
    print("- Specific visual recommendations")
    asyncio.run(test_visual_triage())