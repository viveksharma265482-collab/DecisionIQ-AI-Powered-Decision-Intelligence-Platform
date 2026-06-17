import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("decision-iq-backend")

app = FastAPI(title="DecisionIQ AI Engine", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema for decision analysis
class DecisionAnalysisRequest(BaseModel):
    title: str = Field(..., max_length=200, description="The decision question")
    category: str = Field(..., description="Decision category (e.g., Shopping, Career)")
    mood: str = Field(..., description="Mood state")
    stress: str = Field(..., description="Stress level")
    sleep: str = Field(..., description="Sleep duration last night")
    urgency: str = Field(..., description="Urgency of decision")
    money: str = Field(..., description="Amount of money involved")
    reason: str = Field(..., description="Primary reason or driver")
    context: Optional[str] = Field("", max_length=1000, description="Additional context from user")

# Response schemas
class FactorInfo(BaseModel):
    name: str
    impact: str  # High, Medium, Low
    level: str   # High, Medium, Low

class Scenario(BaseModel):
    title: str
    description: str
    probability: int  # 0-100

class DevilsAdvocate(BaseModel):
    arguments_for: List[str]
    arguments_against: List[str]
    verdict: str

class DecisionAnalysisResponse(BaseModel):
    score: int
    confidence: int
    risk_score: int
    financial_risk_score: int
    emotional_influence_score: int
    risk_level: str  # Safe, Moderate, Risky, Impulsive
    regret_probability: str  # Low, Medium, High
    regret_percentage: int
    biases: List[str]
    risk_factors: List[FactorInfo]
    pros: List[str]
    cons: List[str]
    recommendation: str
    scenarios: List[Scenario]
    devils_advocate: DevilsAdvocate
    summary: str

def generate_deterministic_recommendation(score: int, mood: str, stress: str, urgency: str, sleep: str) -> str:
    recommendations_list = []
    if score < 50:
        recommendations_list.append(f"We strongly recommend waiting at least 48 hours. Your inputs show a high risk of an impulsive decision due to {mood.lower()} mood, high stress ({stress.lower()}), and extreme urgency ({urgency.lower()}).")
    elif score < 75:
        recommendations_list.append(f"We recommend waiting 24 hours. Your state is moderate, but the urgency is high and your sleep was sub-optimal ({sleep.lower()}). Settle your thoughts before proceeding.")
    else:
        recommendations_list.append("This decision has a solid score. You appear calm and well-rested, and the urgency is manageable. You are in a good state to proceed, but review the remaining risk factors first.")
    return " ".join(recommendations_list)

def compute_rule_based_analysis(req: DecisionAnalysisRequest) -> DecisionAnalysisResponse:
    """
    Highly advanced deterministic rule-based analysis that generates a rich, realistic response
    based on the input factors. Used when Gemini API key is missing or as fallback.
    """
    start = time.time()
    # 1. Map risk weights
    # Mood: Angry, Sad, Confused, Anxious = High. Happy = Medium (optimism bias). Calm = Low.
    mood_risks = {
        "Angry": 85, "Sad": 70, "Confused": 75, "Anxious": 80,
        "Happy": 40, "Calm": 10
    }
    mood_risk = mood_risks.get(req.mood, 40)

    # Stress: High = 90, Medium = 50, Low = 10
    stress_risks = {"High": 90, "Medium": 50, "Low": 10}
    stress_risk = stress_risks.get(req.stress, 30)

    # Sleep: Less than 4 hrs = 90, 4-6 hrs = 60, 6-8 hrs = 20, 8+ hrs = 10
    sleep_risks = {
        "Less than 4 hrs": 90,
        "4-6 hrs": 60,
        "6-8 hrs": 20,
        "8+ hrs": 10
    }
    sleep_risk = sleep_risks.get(req.sleep, 30)

    # Urgency: Immediate = 95, Today = 75, This Week = 40, Can Wait = 10
    urgency_risks = {
        "Immediate": 95,
        "Today": 75,
        "This Week": 40,
        "Can Wait": 10
    }
    urgency_risk = urgency_risks.get(req.urgency, 30)

    # Money involved: Above ₹5L = 90, ₹50K-₹5L = 70, ₹5K-₹50K = 45, Under ₹5K = 15, None = 0
    money_risks = {
        "Above ₹5L": 95,
        "₹50K-₹5L": 75,
        "₹5K-₹50K": 45,
        "Under ₹5K": 15,
        "None": 0
    }
    money_risk = money_risks.get(req.money, 10)

    # Reason: FOMO = 90, Fear = 85, Family Pressure = 75, Social Pressure = 75,
    # Curiosity = 30, Opportunity = 25, Career Growth = 15
    reason_risks = {
        "FOMO": 90,
        "Fear": 85,
        "Family Pressure": 75,
        "Social Pressure": 75,
        "Curiosity": 30,
        "Opportunity": 25,
        "Career Growth": 15
    }
    reason_risk = reason_risks.get(req.reason, 30)

    # Calculate weighted average risk score
    # Weights: Stress (0.20), Mood (0.20), Sleep (0.15), Urgency (0.20), Money (0.10), Reason (0.15)
    weighted_risk = (
        (stress_risk * 0.20) +
        (mood_risk * 0.20) +
        (sleep_risk * 0.15) +
        (urgency_risk * 0.20) +
        (money_risk * 0.10) +
        (reason_risk * 0.15)
    )

    # Calculate final score (0-100)
    # High risk -> Low score, Low risk -> High score
    score = int(100 - weighted_risk)
    score = max(10, min(98, score)) # clamp
    t_weights = time.time()
    logger.info(f"[RuleEngine] Risk mapping & scoring took {(t_weights - start)*1000:.2f}ms")

    # Regret percentage & probability mapping
    regret_percentage = int(weighted_risk)
    regret_percentage = max(5, min(95, regret_percentage))
    
    if regret_percentage > 70:
        regret_probability = "High"
        risk_level = "Impulsive" if req.urgency in ["Immediate", "Today"] else "Risky"
    elif regret_percentage > 40:
        regret_probability = "Medium"
        risk_level = "Moderate"
    else:
        regret_probability = "Low"
        risk_level = "Safe"

    # 2. Detect Cognitive Biases
    biases = []
    if req.reason == "FOMO":
        biases.append("FOMO (Fear Of Missing Out)")
    if req.reason == "Fear":
        biases.append("Loss Aversion Bias")
    if req.urgency in ["Immediate", "Today"]:
        biases.append("Hyperbolic Discounting (Urgency Bias)")
    if req.mood in ["Angry", "Anxious", "Sad"]:
        biases.append("Emotional Reasoning")
    if req.reason in ["Social Pressure", "Family Pressure"]:
        biases.append("Social Proof / Conformity Bias")
    if score > 80:
        biases.append("Overconfidence Bias")
    if len(biases) == 0:
        biases.append("Status Quo Bias")
    t_biases = time.time()
    logger.info(f"[RuleEngine] Bias detection took {(t_biases - t_weights)*1000:.2f}ms")

    # 3. Formulate key factor risk list
    risk_factors = []
    
    # Mood Factor
    mood_level = "High" if mood_risk >= 70 else ("Medium" if mood_risk >= 40 else "Low")
    risk_factors.append(FactorInfo(name=f"Emotional State ({req.mood})", impact=mood_level, level=mood_level))
    
    # Stress Factor
    stress_level = "High" if req.stress == "High" else ("Medium" if req.stress == "Medium" else "Low")
    risk_factors.append(FactorInfo(name=f"Stress Level ({req.stress})", impact=stress_level, level=stress_level))

    # Urgency Factor
    urgency_level = "High" if req.urgency in ["Immediate", "Today"] else ("Medium" if req.urgency == "This Week" else "Low")
    risk_factors.append(FactorInfo(name=f"Urgency Bias ({req.urgency})", impact=urgency_level, level=urgency_level))

    # Financial Factor
    financial_level = "High" if req.money in ["Above ₹5L", "₹50K-₹5L"] else ("Medium" if req.money == "₹5K-₹50K" else "Low")
    risk_factors.append(FactorInfo(name=f"Financial Impact ({req.money})", impact=financial_level, level=financial_level))

    # Sort risk factors so High are first
    level_order = {"High": 0, "Medium": 1, "Low": 2}
    risk_factors.sort(key=lambda x: level_order.get(x.level, 2))
    t_factors = time.time()
    logger.info(f"[RuleEngine] Risk factors formulation and sorting took {(t_factors - t_biases)*1000:.2f}ms")

    # 4. Pros and Cons mapping based on Category
    category_data = {
        "Shopping": {
            "pros": ["Immediate gratification/possession", "Solves a short-term need or desire", "Could be a useful tool if utilized properly"],
            "cons": ["Depreciating asset", "Opportunity cost of capital", "Potential buyer's regret due to impulse selection"],
            "summary": "This shopping decision appears to be driven by temporary impulses. The financial cost compared to utility needs balancing.",
            "verdict": "Wait 24 hours. Impulse shopping is heavily tied to dopamine peaks; letting the emotion settle will clarify if this is a need or a temporary want."
        },
        "Career": {
            "pros": ["Potential for professional growth", "New environment and challenge", "Escaping current negative work conditions"],
            "cons": ["Loss of job security during transition", "Uncertain team dynamic at the new role", "Adjustment stress"],
            "summary": "A significant career move that carries structural risks. The current stress and emotional state could be driving a desire to escape rather than step forward.",
            "verdict": "Avoid resigning in anger or high stress. Speak with a mentor outside your company first and list out core non-negotiable needs."
        },
        "Investment": {
            "pros": ["Potential for capital appreciation", "Wealth building in the long-term", "Diversifying your assets"],
            "cons": ["Direct exposure to market volatility", "Liquidity lock-up or loss of capital", "FOMO-driven valuation risk"],
            "summary": "Financial investments made under pressure or FOMO often result in buying at peaks. Critical evaluation of fundamentals is advised.",
            "verdict": "Do not invest today. If the opportunity is genuine, a 48-hour delay won't ruin it; if it's a bubble, a delay saves your capital."
        },
        "Business": {
            "pros": ["Exploits a market gap or client opportunity", "Scale business revenue", "Strategic partnership potential"],
            "cons": ["Commitment of core operational cash", "Operational overhead and execution risk", "Distraction from main product line"],
            "summary": "Business investments require cold, rational calculation. Doing this in an urgent state risks ignoring cash-flow limitations.",
            "verdict": "Consult your financial spreadsheet. Run a worst-case cash flow simulation before signing any commitments."
        },
        "Relationship": {
            "pros": ["Clears air and addresses lingering issues", "Potential to deepen bond or establish boundaries", "Protects emotional health"],
            "cons": ["Risk of temporary conflict or emotional pain", "Irreversible changes to relationship status", "Heightened emotional stress during confrontation"],
            "summary": "Relationship decisions are highly charged. Reacting when tired or angry will lead to communication breakdown.",
            "verdict": "Wait until you are rested. Do not have critical conversations after 9 PM or when sleep-deprived."
        },
        "Education": {
            "pros": ["Skill acquisition and career enhancement", "Structured path to credentials", "Networking with peers"],
            "cons": ["High tuition fees and debt risk", "Massive time commitment", "Potential mismatch with job market demands"],
            "summary": "Education is a long-term investment. Ensure the curriculum matches current market hiring demands.",
            "verdict": "Research three alumni of this program on LinkedIn. Verify their career path before enrolling."
        },
        "Health": {
            "pros": ["Improves long-term vitality", "Reduces chronic health risks", "Enhances mental clarity"],
            "cons": ["Financial cost of premium care/gym/diet", "Initial physical fatigue and adjustment", "Skeptical or unverified treatments"],
            "summary": "Health changes should be sustainable. Quick-fixes or extreme changes rarely stick and can cause injury.",
            "verdict": "Consult a medical professional. Avoid buying unverified supplements or signing up for non-refundable long-term programs."
        },
        "Other": {
            "pros": ["Addresses current bottleneck", "Provides clarity of action", "Moves past analysis paralysis"],
            "cons": ["Unintended secondary consequences", "Lacks strategic alignment", "Driven by temporary emotions"],
            "summary": "A miscellaneous decision that requires careful review. Ensure your emotional state is stable before committing.",
            "verdict": "De-escalate the urgency. If it can wait a week, review it then with fresh eyes."
        }
    }

    cat_info = category_data.get(req.category, category_data["Other"])
    t_proscons = time.time()
    logger.info(f"[RuleEngine] Category Pros/Cons mapping took {(t_proscons - t_factors)*1000:.2f}ms")
    
    # 5. Customize Pros and Cons based on context if user entered it
    pros = cat_info["pros"][:]
    cons = cat_info["cons"][:]
    
    # Heuristics updates based on inputs
    if req.sleep in ["Less than 4 hrs", "4-6 hrs"]:
        cons.append("Impaired cognitive processing due to sleep deprivation")
    if req.stress == "High":
        cons.append("High cortisol limits long-term thinking capacity")
    if req.money in ["Above ₹5L", "₹50K-₹5L"]:
        cons.append("Significant financial exposure of resources")
    if req.urgency in ["Immediate", "Today"]:
        cons.append("Artificially shortened deliberation window")

    # 6. Recommendation formulation
    recommendation = generate_deterministic_recommendation(
        score,
        req.mood,
        req.stress,
        req.urgency,
        req.sleep
    )
    t_rec = time.time()
    logger.info(f"[RuleEngine] Recommendation formulation took {(t_rec - t_proscons)*1000:.2f}ms")

    # 7. Scenarios simulation
    scenarios = [
        Scenario(
            title="Scenario A: Best Case",
            description=f"You move forward. Your motivations under {req.reason} succeed. You gain {pros[0].lower()} and achieve your goal with minimal friction.",
            probability=max(10, min(80, int(score * 0.8)))
        ),
        Scenario(
            title="Scenario B: Neutral Outcome",
            description=f"You move forward but meet moderate friction. The benefits of '{pros[1].lower()}' are realized, but you also encounter challenges related to '{cons[1].lower()}'. Results are mixed.",
            probability=max(20, min(60, int(100 - score)))
        ),
        Scenario(
            title="Scenario C: Worst Case",
            description=f"You move forward in an state of {req.stress.lower()} stress and {req.mood.lower()} mood. The risks materialize: {cons[0].lower()}, leading to a high probability of regret ({regret_percentage}%).",
            probability=max(10, min(80, regret_percentage))
        )
    ]
    
    # Normalize probabilities to sum up closer to 100
    total_prob = sum(s.probability for s in scenarios)
    for s in scenarios:
        s.probability = int((s.probability / total_prob) * 100)
    # Adjust last one to ensure exact sum of 100
    scenarios[-1].probability += (100 - sum(s.probability for s in scenarios))

    # 8. Devil's Advocate
    devils_advocate = DevilsAdvocate(
        arguments_for=[
            f"Fulfills the immediate driver of '{req.reason}'.",
            f"Capitalizes on the current urgency window ({req.urgency.lower()}).",
            f"Achieves: {pros[0]}."
        ],
        arguments_against=[
            f"Highly compromised state of mind: {req.mood} mood and {req.stress.lower()} stress.",
            f"Ignores key risk factor: {cons[0]}.",
            f"Under-deliberated due to immediate pressure."
        ],
        verdict=cat_info["verdict"]
    )

    # Context injections
    if req.context and len(req.context.strip()) > 0:
        summary = f"Based on your current state and the details you provided, there is a {regret_probability.lower()} risk of regret. You noted: '{req.context}'. This context suggests external pressures are playing a significant role in clouding your judgment."
    else:
        summary = f"Based on your current state and the details you provided, there is a {regret_probability.lower()} risk of regret. Your stress level is {req.stress.lower()} and sleep is {req.sleep.lower()}, which reduces cognitive performance."

    risk_score = 100 - score
    financial_risk_score = int(money_risk)
    emotional_influence_score = int((mood_risk * 0.5) + (stress_risk * 0.5))

    logger.info(f"[RuleEngine] Total rule-based analysis completed in {(time.time() - start)*1000:.2f}ms")
    return DecisionAnalysisResponse(
        score=score,
        confidence=int(80 - (stress_risk * 0.2)),
        risk_score=risk_score,
        financial_risk_score=financial_risk_score,
        emotional_influence_score=emotional_influence_score,
        risk_level=risk_level,
        regret_probability=regret_probability,
        regret_percentage=regret_percentage,
        biases=biases,
        risk_factors=risk_factors,
        pros=pros,
        cons=cons,
        recommendation=recommendation,
        scenarios=scenarios,
        devils_advocate=devils_advocate,
        summary=summary
    )

def compute_gemini_analysis(req: DecisionAnalysisRequest, api_key: str) -> Optional[DecisionAnalysisResponse]:
    """
    Attempts to call Google Gemini API for an intelligent decision critique.
    """
    start = time.time()
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # We will use gemini-2.5-flash or gemini-1.5-flash
        model = genai.GenerativeModel('gemini-1.5-flash')
        t_model = time.time()
        logger.info(f"[GeminiEngine] Configured model in {(t_model - start)*1000:.2f}ms")
        
        prompt = f"""
        You are a world-class Decision Consultant and Cognitive Psychologist.
        Analyze the following decision scenario and return a JSON object matches the required schema.

        DECISION: "{req.title}"
        CATEGORY: {req.category}
        MOOD: {req.mood}
        STRESS: {req.stress}
        SLEEP: {req.sleep}
        URGENCY: {req.urgency}
        MONEY INVOLVED: {req.money}
        PRIMARY REASON: {req.reason}
        ADDITIONAL CONTEXT: "{req.context}"

        Instructions:
        1. Evaluate if this is an impulsive, biased, or emotional decision.
        2. Detect cognitive biases present in this situation (e.g. FOMO, Sunk Cost Fallacy, Confirmation Bias, Loss Aversion, Present Bias).
        3. Rate the Decision Intelligence Score from 0 to 100 (where 100 is highly rational/sound and 0 is extremely impulsive/risky).
        4. Calculate:
           - risk_score: Overall risk score (0-100)
           - financial_risk_score: Financial risk score (0-100)
           - emotional_influence_score: Emotional risk/influence score (0-100)
        5. Calculate regret probability (Low, Medium, High) and percentage (0-100%).
        6. Return a detailed, professional evaluation in structured JSON format.

        JSON Schema:
        {{
            "score": <int 0-100>,
            "confidence": <int 0-100>,
            "risk_score": <int 0-100>,
            "financial_risk_score": <int 0-100>,
            "emotional_influence_score": <int 0-100>,
            "risk_level": "Safe" | "Moderate" | "Risky" | "Impulsive",
            "regret_probability": "Low" | "Medium" | "High",
            "regret_percentage": <int 0-100>,
            "biases": ["Bias 1", "Bias 2", ...],
            "risk_factors": [
                {{"name": "Stress Level", "impact": "High"|"Medium"|"Low", "level": "High"|"Medium"|"Low"}},
                {{"name": "Emotional State", "impact": "High"|"Medium"|"Low", "level": "High"|"Medium"|"Low"}},
                ...
            ],
            "pros": ["Pro 1", "Pro 2", "Pro 3"],
            "cons": ["Con 1", "Con 2", "Con 3"],
            "recommendation": "<Cooling-off advice based on time frame and psychological trigger>",
            "scenarios": [
                {{"title": "Scenario A: Best Case", "description": "...", "probability": <int>}},
                {{"title": "Scenario B: Neutral Outcome", "description": "...", "probability": <int>}},
                {{"title": "Scenario C: Worst Case", "description": "...", "probability": <int>}}
            ],
            "devils_advocate": {{
                "arguments_for": ["...", "..."],
                "arguments_against": ["...", "..."],
                "verdict": "<Final synthesising advisory advice>"
            }},
            "summary": "<2-3 sentence summary of the psychological assessment>"
        }}

        Return ONLY raw JSON. Do not include markdown tags like ```json.
        """
        
        t_prompt = time.time()
        # Set a 2.0 second timeout to ensure execution finishes under 3s limit
        response = model.generate_content(prompt, request_options={"timeout": 2.0})
        t_network = time.time()
        logger.info(f"[GeminiEngine] Network API call returned in {(t_network - t_prompt)*1000:.2f}ms")
        text = response.text.strip()
        
        # Parse JSON
        # Remove markdown helper blocks if LLM still outputs them
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                text = "\n".join(lines[1:-1])
        
        data = json.loads(text)
        t_network_parse = time.time()
        logger.info(f"[GeminiEngine] Response JSON parsed in {(t_network_parse - t_network)*1000:.2f}ms")
        
        score_val = int(data.get("score", 70))
        local_rec = generate_deterministic_recommendation(
            score_val,
            req.mood,
            req.stress,
            req.urgency,
            req.sleep
        )
        logger.info(f"[GeminiEngine] Overrode recommendation with deterministic local logic based on score={score_val}")
        
        logger.info(f"[GeminiEngine] Total Gemini analysis completed in {(time.time() - start)*1000:.2f}ms")
        # Map into response schema to ensure validation passes
        return DecisionAnalysisResponse(
            score=score_val,
            confidence=int(data.get("confidence", 80)),
            risk_score=int(data.get("risk_score", 100 - score_val)),
            financial_risk_score=int(data.get("financial_risk_score", 50)),
            emotional_influence_score=int(data.get("emotional_influence_score", 50)),
            risk_level=data.get("risk_level", "Moderate"),
            regret_probability=data.get("regret_probability", "Medium"),
            regret_percentage=int(data.get("regret_percentage", 50)),
            biases=data.get("biases", []),
            risk_factors=[FactorInfo(**rf) for rf in data.get("risk_factors", [])],
            pros=data.get("pros", []),
            cons=data.get("cons", []),
            recommendation=local_rec,
            scenarios=[Scenario(**s) for s in data.get("scenarios", [])],
            devils_advocate=DevilsAdvocate(**data.get("devils_advocate", {})),
            summary=data.get("summary", "")
        )
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}. Falling back to rule-based engine.")
        return None

@app.post("/api/analyze", response_model=DecisionAnalysisResponse)
async def analyze_decision(request: DecisionAnalysisRequest):
    start_time = time.time()
    logger.info(f"Analyzing decision: {request.title}")
    
    # Try calling Gemini API if key exists
    api_key = os.getenv("GEMINI_API_KEY")
    logger.info(f"GEMINI_API_KEY present: {bool(api_key)}")
    if api_key:
        gemini_start = time.time()
        logger.info("Starting Gemini analysis...")
        result = compute_gemini_analysis(request, api_key)
        logger.info(f"Gemini analysis finished in {(time.time() - gemini_start)*1000:.2f}ms")
        if result:
            logger.info(f"Total analysis endpoint processing time (Gemini): {(time.time() - start_time)*1000:.2f}ms")
            return result
            
    # Fallback to rule-based engine
    rule_start = time.time()
    logger.info("Starting rule-based analysis...")
    result = compute_rule_based_analysis(request)
    logger.info(f"Rule-based analysis finished in {(time.time() - rule_start)*1000:.2f}ms")
    logger.info(f"Total analysis endpoint processing time (Rule-based): {(time.time() - start_time)*1000:.2f}ms")
    return result

if __name__ == "__main__":
    import uvicorn
    # Load dotenv explicitly in script mode
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
