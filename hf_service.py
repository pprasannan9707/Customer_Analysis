"""
hf_service.py
Generates natural-language explanations, retention recommendations,
and executive summaries via Hugging Face Inference API.

Environment variables:
  HUGGINGFACE_API_KEY  - required
  HF_MODEL            - optional, defaults to mistralai/Mistral-7B-Instruct-v0.2
"""

import os
import httpx
import asyncio
from typing import Optional

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_BASE = "https://api-inference.huggingface.co/models"

TIMEOUT = 30.0


async def _call_hf(prompt: str, max_new_tokens: int = 400) -> str:
    """
    Call the Hugging Face Inference API with the given prompt.
    Returns the generated text string.
    """
    if not HF_API_KEY:
        return _fallback_response(prompt)

    url = f"{HF_BASE}/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False,
        },
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "").strip()
            return str(data)
        except httpx.HTTPStatusError as e:
            return f"[HF API Error {e.response.status_code}]: {e.response.text[:200]}"
        except Exception as e:
            return f"[HF API Error]: {str(e)}"


def _fallback_response(prompt: str) -> str:
    """Return a canned analytical response when no API key is configured."""
    if "segment" in prompt.lower():
        return (
            "This customer segment shows above-average campaign engagement and spending patterns. "
            "High-value targeting through personalized email and catalog campaigns is recommended. "
            "Focus on wine and premium product offers to maximize conversion rates."
        )
    if "what-if" in prompt.lower() or "scenario" in prompt.lower():
        return (
            "Based on the adjusted parameters, this customer's response likelihood has shifted. "
            "The primary drivers are recency and total spend. Re-engagement via targeted promotions "
            "within the next 30 days is advised to capitalize on the elevated probability window."
        )
    return (
        "This customer profile indicates a moderate-to-high likelihood of campaign response. "
        "Key drivers include spending behavior, campaign history, and recency. "
        "Recommended action: personalized outreach with premium product focus within the next 2 weeks."
    )


async def explain_prediction(
    customer_profile: dict,
    probability: float,
    top_features: list,
    risk_label: str,
) -> str:
    """Generate a plain-English explanation for a single customer prediction."""
    features_str = "\n".join(
        f"  - {f['feature'].replace('_', ' ').title()}: {f['value']:.2f} "
        f"(contributes {f['contribution_pct']:.1f}%)"
        for f in top_features[:5]
    )

    prompt = f"""[INST] You are a senior marketing data scientist at a retail bank.
Analyze this customer's predicted response to a marketing campaign.

Customer Profile:
- Age: {customer_profile.get('Year_Birth', 'N/A')}
- Income: ${customer_profile.get('Income', 0):,.0f}
- Total Historical Spend: ${customer_profile.get('Total_Spent', customer_profile.get('MntWines', 0) + customer_profile.get('MntMeatProducts', 0)):.0f}
- Recency (days since last purchase): {customer_profile.get('Recency', 'N/A')}
- Campaign Response Probability: {probability*100:.1f}%
- Risk Classification: {risk_label}

Top Driving Factors:
{features_str}

Write a concise 3-sentence business explanation of WHY this customer is classified as "{risk_label}",
what the key behavioral signals mean, and ONE specific recommended marketing action.
Write for a non-technical marketing manager. [/INST]"""

    return await _call_hf(prompt, max_new_tokens=250)


async def recommend_action(
    customer_profile: dict,
    probability: float,
    top_features: list,
) -> str:
    """Generate a targeted retention/upsell recommendation."""
    prompt = f"""[INST] You are a CRM strategy expert.

Customer data:
- Response probability: {probability*100:.1f}%
- Top engagement signals: {', '.join(f['feature'] for f in top_features[:4])}
- Total spend: ${customer_profile.get('Total_Spent', 0):.0f}
- Children at home: {customer_profile.get('Children', customer_profile.get('Kidhome', 0) + customer_profile.get('Teenhome', 0))}
- Web visits/month: {customer_profile.get('NumWebVisitsMonth', 'N/A')}

Give 3 specific, actionable marketing recommendations to maximize this customer's campaign response.
Format as a numbered list. Be specific about channel, timing, and offer type. [/INST]"""

    return await _call_hf(prompt, max_new_tokens=300)


async def generate_segment_summary(
    segment_name: str,
    segment_stats: dict,
) -> str:
    """Generate an executive segment summary."""
    prompt = f"""[INST] You are a VP of Marketing Analytics.

Customer Segment: {segment_name}
Segment Statistics:
- Total customers: {segment_stats.get('count', 0)}
- Average response rate: {segment_stats.get('response_rate', 0)*100:.1f}%
- Average income: ${segment_stats.get('avg_income', 0):,.0f}
- Average total spend: ${segment_stats.get('avg_spend', 0):.0f}
- Average recency: {segment_stats.get('avg_recency', 0):.0f} days
- Campaign acceptance rate: {segment_stats.get('campaign_rate', 0)*100:.1f}%

Write a 4-sentence executive segment summary:
1. Who this segment is (demographics/behavior profile)
2. Why they are valuable (or not)
3. Key risk or opportunity
4. Strategic recommendation for this segment [/INST]"""

    return await _call_hf(prompt, max_new_tokens=300)


async def generate_campaign_strategy(
    overall_stats: dict,
    top_segments: list,
) -> str:
    """Generate a board-level campaign strategy summary."""
    seg_str = "\n".join(
        f"  - {s['name']}: {s['response_rate']*100:.1f}% response rate, "
        f"avg income ${s.get('avg_income', 0):,.0f}"
        for s in top_segments[:3]
    )

    prompt = f"""[INST] You are the Chief Marketing Officer of a global retail company.

Campaign Intelligence Report:
- Total customers analyzed: {overall_stats.get('total_customers', 0)}
- Overall response rate: {overall_stats.get('response_rate', 0)*100:.1f}%
- High-potential customers (>75% probability): {overall_stats.get('high_potential', 0)}
- Average income: ${overall_stats.get('avg_income', 0):,.0f}

Top Performing Segments:
{seg_str}

Write a strategic campaign brief (5-6 sentences) covering:
- Key opportunity for next campaign
- Recommended target segments and why
- Channel mix recommendation
- Budget prioritization guidance
- Expected outcome if executed well [/INST]"""

    return await _call_hf(prompt, max_new_tokens=400)


async def explain_whatif(
    original_prob: float,
    new_prob: float,
    changed_fields: dict,
    top_features: list,
) -> str:
    """Explain what changed in a what-if scenario."""
    changes_str = "\n".join(f"  - {k}: {v}" for k, v in changed_fields.items())
    delta = new_prob - original_prob

    prompt = f"""[INST] You are a marketing analytics consultant.

What-If Scenario Analysis:
- Original response probability: {original_prob*100:.1f}%
- New response probability: {new_prob*100:.1f}%
- Change: {'+' if delta >= 0 else ''}{delta*100:.1f} percentage points
- Parameters adjusted:
{changes_str}

Write a 3-sentence analytical interpretation:
1. What this probability change means in business terms
2. Which of the changed parameters had the most impact and why
3. Whether this customer should now be re-prioritized for the next campaign [/INST]"""

    return await _call_hf(prompt, max_new_tokens=200)
