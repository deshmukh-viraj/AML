import os
from openai import OpenAI
import logging
from dotenv import load_dotenv
logger = logging.getLogger(__name__)
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"  
)


SYSTEM_PROMPT = """
You are an AML Triage Copilot writing the initial summary for a Suspicious Activity Report (SAR). 

CRITICAL RULES:
- DO NOT parrot the text from the ALERT BEHAVIORAL PROFILE. 
- DO NOT mention the exact feature names (e.g., "counterparty network topology" or "payment trails").
- DO NOT mention the raw decimal values (e.g., "Value: 0.00").
- Instead, ABSORB the concepts and rewrite them as a cohesive, authoritative, 2-sentence narrative using standard AML terminology.

BAD OUTPUT: "High transaction velocity or rapid timing patterns (Value: 0.00) and Suspicious counterparty network topology (Value: 1.00) points to a potentially illicit conduit network." (Too robotic, parrots feature names).

GOOD OUTPUT: "The account executed multiple rapid-fire transfers to a tightly clustered counterparty network via high-risk payment corridors, indicating a classic pass-through or smurfing operation." (Natural, authoritative, ready to paste into a SAR).

OUTPUT FORMAT:
1. SUSPICIOUS PATTERN: [1 sentence naming the likely scheme]
2. KEY EVIDENCE: [1-2 sentences weaving the behavioral indicators together naturally]
3. RECOMMENDED ACTION: [1 specific, actionable next step for the investigator]
"""

def generate_investigation_summary(translated_shap_context: str) -> str:
    """
    Sends translated SHAP evidence to LLM and returns the narrative.
    """
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Alert Evidence:\n{translated_shap_context}"}
            ],
            temperature=0.2,
            top_p=1,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return "Error generating AI summary. Please review raw SHAP values manually."