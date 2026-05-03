import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

PRIORITY_MAP = {0: "Low", 1: "Medium", 2: "High"}

#Shared system prompt 
SYSTEM_PROMPT = """You are an expert disaster management decision support system.
Your role is to analyse disaster event data, explain the severity classification,
and provide clear, actionable emergency response recommendations.
Be concise, structured, and use plain language suitable for field responders."""

def _build_user_prompt(priority: str, features: dict) -> str:
    feature_lines = "\n".join(
        f"  • {k}: {v:,.0f}" if isinstance(v, (int, float)) else f"  • {k}: {v}"
        for k, v in features.items()
        if not str(k).startswith("log_")        
    )
    return f"""Disaster event data:
{feature_lines}

ML-predicted priority level: {priority}

Please provide:
1. Why this priority level was assigned (link to the specific numbers above)
2. Immediate response actions (first 24 hours)
3. Resource allocation recommendations
4. Key risks to monitor
"""

def _call_gemini(priority: str, features: dict) -> str:
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.4,
            max_output_tokens=600,
        ),
    )

    response = model.generate_content(_build_user_prompt(priority, features))
    return response.text


def _call_groq(priority: str, features: dict) -> str:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(priority, features)},
        ],
        temperature=0.4,
        max_tokens=600,
    )
    return response.choices[0].message.content


def get_llm_explanation(priority, features) -> str:
    """
    priority : int (0/1/2) or str ("Low"/"Medium"/"High")
    features : dict of disaster feature values
    Returns  : str — LLM-generated explanation and recommendations
    """
    if isinstance(priority, (int, float)):
        priority_label = PRIORITY_MAP.get(int(priority), str(priority))
    else:
        priority_label = str(priority)

    if GEMINI_API_KEY:
        try:
            return _call_gemini(priority_label, features)
        except Exception as e:
            print(f"[llm_engine] Gemini failed: {e}. Trying Groq …")

    if GROQ_API_KEY:
        try:
            return _call_groq(priority_label, features)
        except Exception as e:
            print(f"[llm_engine] Groq failed: {e}. Using static fallback …")

    return fallback_explanation(priority_label)


def fallback_explanation(priority, features=None, error_msg=None) -> str:
    if isinstance(priority, (int, float)):
        priority = PRIORITY_MAP.get(int(priority), str(priority))

    actions = {
        "High": [
            "Deploy national rapid-response teams immediately",
            "Issue public emergency broadcast and evacuation orders",
            "Request international humanitarian assistance if needed",
            "Establish emergency operations centre (EOC)",
        ],
        "Medium": [
            "Mobilise regional emergency response units",
            "Pre-position relief supplies at distribution points",
            "Activate local government disaster protocols",
            "Set up temporary shelter and medical camps",
        ],
        "Low": [
            "Deploy local civil defence and first-responder teams",
            "Monitor situation for potential escalation",
            "Coordinate with district authorities for resource support",
            "Assess damage and register affected households",
        ],
    }

    response_actions = "\n".join(f"  • {a}" for a in actions.get(priority, actions["Medium"]))

    result = f"""⚠️  LLM service unavailable — static decision summary

Priority Level : {priority}

Reasoning:
  Severity classification is based on a weighted combination of:
  deaths, injuries, total affected population, and economic damage.
  High-weight features (deaths, total affected) drive this assessment.

Immediate Response Actions:
{response_actions}

Resource Recommendations:
  • Allocate emergency funds proportional to affected population size
  • Coordinate NGO and government logistics for last-mile delivery
  • Ensure medical teams are on standby for injury management

Risks to Monitor:
  • Disease outbreaks in displaced populations
  • Infrastructure damage blocking relief corridors
  • Secondary hazards (aftershocks, flooding, landslides)
"""
    if error_msg:
        result += f"\n[Debug: {error_msg}]"

    return result


if __name__ == "__main__":
    test_features = {
        "Start Year": 2024,
        "Total Deaths": 420,
        "No. Injured": 1300,
        "No. Affected": 80000,
        "No. Homeless": 12000,
        "Total Affected": 93000,
        "Total Damage ('000 US$)": 900000,
    }

    print("Testing LLM engine …\n")
    result = get_llm_explanation(2, test_features)  
    print(result)