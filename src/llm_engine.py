from openai import OpenAI, RateLimitError, AuthenticationError

client = OpenAI()

def get_llm_explanation(priority, features):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a disaster management decision support system."
                },
                {
                    "role": "user",
                    "content": f"""
Disaster features:
{features}

Predicted priority level: {priority}

Explain why this priority was assigned and suggest response actions.
"""
                }
            ],
            temperature=0.4
        )

        return response.choices[0].message.content

    except (RateLimitError, AuthenticationError) as e:
        return fallback_explanation(priority, features, str(e))


def fallback_explanation(priority, features, error_msg=None):
    level_map = {0: "Low", 1: "Medium", 2: "High"}
    level = level_map[int(priority)]

    explanation = f"""
⚠️ LLM unavailable (API limitation).

Priority Level: **{level}**

Reasoning:
- Severity indicators (deaths, injuries, affected population)
- Economic damage estimates
- Historical disaster impact patterns

Recommended actions:
- Activate emergency response units
- Allocate relief resources immediately
- Coordinate with government and NGOs
- Continuous monitoring and reassessment
"""

    if error_msg:
        explanation += f"\n\n[Debug info: {error_msg}]"

    return explanation
