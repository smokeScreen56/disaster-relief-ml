def recommend_action(severity_level):
    """Maps predicted severity level to relief actions"""

    if severity_level == "Low":
        return {
            "priority": 3,
            "action": "Monitor situation, local authorities handle response"
        }

    elif severity_level == "Medium":
        return {
            "priority": 2,
            "action": "Deploy medical teams and emergency supplies"
        }

    else:  # High severity
        return {
            "priority": 1,
            "action": "National emergency response: NDRF, Army, large-scale evacuation"
        }
