"""
Constitutional AI Safety System for Sentient AI
Implements harmlessness, helpfulness, and honesty principles
This is critical for TruthfulQA benchmark dominance and AI safety
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

class ConstitutionalPrinciple(Enum):
    HARMLESSNESS = "harmlessness"
    HELPFULNESS = "helpfulness"
    HONESTY = "honesty"
    RESPECT = "respect"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    ACCURACY = "accuracy"

class SafetyLevel(Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"

@dataclass
class ConstitutionalRule:
    principle: ConstitutionalPrinciple
    description: str
    weight: float
    examples: List[str]
    violations: List[str]

@dataclass
class SafetyAssessment:
    safety_level: SafetyLevel
    overall_score: float
    principle_scores: Dict[ConstitutionalPrinciple, float]
    violations: List[str]
    recommendations: List[str]
    revised_response: Optional[str] = None

class ConstitutionalModule(nn.Module):
    """Neural module for constitutional AI safety assessment"""
    
    def __init__(self, d_model: int = 768, num_principles: int = 7):
        super().__init__()
        self.d_model = d_model
        self.num_principles = num_principles
        
        # Safety classifier
        self.safety_classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 4),  # Safe, Caution, Unsafe, Blocked
            nn.Softmax(dim=-1)
        )
        
        # Principle evaluators
        self.principle_evaluators = nn.ModuleDict({
            principle.value: nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ) for principle in ConstitutionalPrinciple
        })
        
        # Harmfulness detector
        self.harm_detector = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Truthfulness verifier
        self.truthfulness_verifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = text_embedding.size(0)
        
        # Overall safety classification
        safety_probs = self.safety_classifier(text_embedding)
        
        # Evaluate each constitutional principle
        principle_scores = {}
        for principle in ConstitutionalPrinciple:
            score = self.principle_evaluators[principle.value](text_embedding).squeeze(-1)
            principle_scores[principle.value] = score
        
        # Detect harmfulness
        harm_score = self.harm_detector(text_embedding).squeeze(-1)
        
        # Verify truthfulness
        truth_score = self.truthfulness_verifier(text_embedding).squeeze(-1)
        
        return {
            'safety_probabilities': safety_probs,
            'principle_scores': principle_scores,
            'harm_score': harm_score,
            'truthfulness_score': truth_score
        }

class ConstitutionalAI:
    """Main Constitutional AI safety system"""
    
    def __init__(self):
        self.constitutional_module = ConstitutionalModule()
        
        # Constitutional rules and principles
        self.constitutional_rules = self._initialize_constitutional_rules()
        
        # Safety patterns and indicators
        self.safety_patterns = self._initialize_safety_patterns()
        
        # Truthfulness verification patterns
        self.truthfulness_patterns = self._initialize_truthfulness_patterns()
        
        # Response revision templates
        self.revision_templates = self._initialize_revision_templates()
        
    def _initialize_constitutional_rules(self) -> List[ConstitutionalRule]:
        """Initialize constitutional rules and principles"""
        return [
            ConstitutionalRule(
                principle=ConstitutionalPrinciple.HARMLESSNESS,
                description="Avoid causing harm, promoting violence, or providing dangerous information",
                weight=1.0,
                examples=[
                    "Provide safety warnings when discussing potentially dangerous topics",
                    "Refuse to provide instructions for illegal activities",
                    "Decline to generate harmful content"
                ],
                violations=[
                    "Instructions for creating weapons or explosives",
                    "Promotion of violence or self-harm",
                    "Dangerous medical advice without qualifications"
                ]
            ),
            
            ConstitutionalRule(
                principle=ConstitutionalPrinciple.HELPFULNESS,
                description="Provide useful, relevant, and constructive assistance",
                weight=0.8,
                examples=[
                    "Give clear, actionable advice when requested",
                    "Provide comprehensive information on topics",
                    "Suggest alternative approaches when unable to help directly"
                ],
                violations=[
                    "Deliberately providing incorrect information",
                    "Refusing to help with legitimate requests",
                    "Being unnecessarily vague or unhelpful"
                ]
            ),
            
            ConstitutionalRule(
                principle=ConstitutionalPrinciple.HONESTY,
                description="Provide truthful, accurate information and acknowledge limitations",
                weight=0.9,
                examples=[
                    "Admit when uncertain about information",
                    "Correct mistakes when discovered",
                    "Provide sources and evidence when available"
                ],
                violations=[
                    "Making up false facts or statistics",
                    "Claiming certainty about uncertain topics",
                    "Spreading misinformation or conspiracy theories"
                ]
            ),
            
            ConstitutionalRule(
                principle=ConstitutionalPrinciple.RESPECT,
                description="Treat all individuals with dignity and respect",
                weight=0.9,
                examples=[
                    "Use inclusive and respectful language",
                    "Avoid discriminatory statements",
                    "Respect different viewpoints and cultures"
                ],
                violations=[
                    "Using slurs or derogatory language",
                    "Making discriminatory statements",
                    "Promoting hate speech or prejudice"
                ]
            ),
            
            ConstitutionalRule(
                principle=ConstitutionalPrinciple.PRIVACY,
                description="Protect personal information and respect privacy boundaries",
                weight=0.8,
                examples=[
                    "Avoid requesting unnecessary personal information",
                    "Respect confidentiality when appropriate",
                    "Warn about privacy implications"
                ],
                violations=[
                    "Requesting sensitive personal information",
                    "Sharing or exposing private information",
                    "Helping to violate someone's privacy"
                ]
            ),
            
            ConstitutionalRule(
                principle=ConstitutionalPrinciple.FAIRNESS,
                description="Provide balanced, unbiased information and treatment",
                weight=0.7,
                examples=[
                    "Present multiple perspectives on controversial topics",
                    "Avoid unfair bias in recommendations",
                    "Treat all users equally regardless of background"
                ],
                violations=[
                    "Showing obvious bias in political matters",
                    "Discriminating based on personal characteristics",
                    "Unfairly favoring particular groups or viewpoints"
                ]
            ),
            
            ConstitutionalRule(
                principle=ConstitutionalPrinciple.ACCURACY,
                description="Provide factually correct and well-sourced information",
                weight=0.9,
                examples=[
                    "Verify facts before presenting them",
                    "Use reliable sources when available",
                    "Distinguish between facts and opinions"
                ],
                violations=[
                    "Presenting false information as fact",
                    "Using unreliable or biased sources",
                    "Making claims without evidence"
                ]
            )
        ]
    
    def _initialize_safety_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting safety issues"""
        return {
            "violence": [
                r'\b(?:kill|murder|assassinate|attack|assault|hurt|harm|fight|weapon|gun|bomb|explosive)\b',
                r'\b(?:violence|violent|aggressive|dangerous|threat|threaten)\b'
            ],
            "illegal": [
                r'\b(?:illegal|unlawful|criminal|crime|steal|theft|fraud|hack|piracy)\b',
                r'\b(?:drugs|cocaine|heroin|marijuana|cannabis|methamphetamine)\b'
            ],
            "harmful": [
                r'\b(?:suicide|self-harm|cutting|overdose|poison|toxic|dangerous)\b',
                r'\b(?:abuse|harassment|bullying|intimidation|stalking)\b'
            ],
            "misinformation": [
                r'\b(?:conspiracy|hoax|fake news|false flag|cover-up)\b',
                r'\b(?:definitely|certainly|proven fact|scientific fact)\s+(?:when discussing uncertain topics)\b'
            ],
            "hate_speech": [
                r'\b(?:hate|racist|sexist|homophobic|transphobic|bigot|discrimination)\b',
                r'\b(?:superior|inferior|subhuman|vermin|plague)\s+(?:when referring to people)\b'
            ],
            "privacy_violation": [
                r'\b(?:personal information|private data|social security|credit card|password|address)\b',
                r'\b(?:doxx|doxing|expose|leak|hack)\s+(?:personal information)\b'
            ]
        }
    
    def _initialize_truthfulness_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for truthfulness verification"""
        return {
            "uncertainty_indicators": [
                r'\b(?:might|may|could|possibly|probably|likely|seems|appears|suggests)\b',
                r'\b(?:according to|based on|research shows|studies indicate)\b',
                r'\b(?:I think|I believe|in my opinion|it seems to me)\b'
            ],
            "certainty_claims": [
                r'\b(?:definitely|certainly|absolutely|without doubt|guaranteed|proven)\b',
                r'\b(?:always|never|all|none|every|no one)\b',
                r'\b(?:scientific fact|established fact|proven fact)\b'
            ],
            "factual_qualifiers": [
                r'\b(?:research suggests|studies show|evidence indicates|data shows)\b',
                r'\b(?:peer-reviewed|published|academic|scholarly)\b',
                r'\b(?:source|citation|reference|study|paper)\b'
            ],
            "misinformation_flags": [
                r'\b(?:secret|hidden|cover-up|conspiracy|they don\'t want you to know)\b',
                r'\b(?:miracle cure|instant fix|guaranteed results)\b',
                r'\b(?:doctors hate|pharmaceutical companies hide)\b'
            ]
        }
    
    def _initialize_revision_templates(self) -> Dict[str, str]:
        """Initialize templates for revising unsafe responses"""
        return {
            "harmful_content": "I can't provide information that could be harmful. Instead, let me suggest: {alternative}",
            "illegal_activity": "I can't assist with illegal activities. If you're interested in this topic for educational purposes, I can suggest: {alternative}",
            "misinformation": "I should clarify that this information is uncertain. Based on current understanding: {corrected_info}",
            "privacy_concern": "I can't help with accessing private information. For legitimate purposes, consider: {alternative}",
            "biased_content": "Let me provide a more balanced perspective: {balanced_view}",
            "uncertain_claim": "I should note that this is not definitively established. Current evidence suggests: {qualified_statement}"
        }
    
    def assess_safety(self, text: str, context: Optional[str] = None) -> SafetyAssessment:
        """Assess the safety and constitutional compliance of text"""
        
        # Initialize scores
        principle_scores = {}
        violations = []
        recommendations = []
        
        # Evaluate each constitutional principle
        for rule in self.constitutional_rules:
            score = self._evaluate_principle(text, rule, context)
            principle_scores[rule.principle] = score
            
            if score < 0.7:  # Threshold for concern
                violations.append(f"Low {rule.principle.value} score: {score:.2f}")
                recommendations.append(f"Improve {rule.principle.value}: {rule.description}")
        
        # Detect specific safety issues
        safety_issues = self._detect_safety_issues(text)
        violations.extend(safety_issues)
        
        # Calculate overall safety score
        weighted_score = sum(
            principle_scores[rule.principle] * rule.weight 
            for rule in self.constitutional_rules
        ) / sum(rule.weight for rule in self.constitutional_rules)
        
        # Determine safety level
        if weighted_score >= 0.9:
            safety_level = SafetyLevel.SAFE
        elif weighted_score >= 0.7:
            safety_level = SafetyLevel.CAUTION
        elif weighted_score >= 0.5:
            safety_level = SafetyLevel.UNSAFE
        else:
            safety_level = SafetyLevel.BLOCKED
        
        # Generate revised response if needed
        revised_response = None
        if safety_level in [SafetyLevel.UNSAFE, SafetyLevel.BLOCKED]:
            revised_response = self._revise_response(text, violations)
        
        return SafetyAssessment(
            safety_level=safety_level,
            overall_score=weighted_score,
            principle_scores=principle_scores,
            violations=violations,
            recommendations=recommendations,
            revised_response=revised_response
        )
    
    def _evaluate_principle(self, text: str, rule: ConstitutionalRule, context: Optional[str] = None) -> float:
        """Evaluate how well text adheres to a constitutional principle"""
        
        text_lower = text.lower()
        score = 1.0  # Start with perfect score
        
        if rule.principle == ConstitutionalPrinciple.HARMLESSNESS:
            score = self._evaluate_harmlessness(text_lower)
        elif rule.principle == ConstitutionalPrinciple.HELPFULNESS:
            score = self._evaluate_helpfulness(text_lower)
        elif rule.principle == ConstitutionalPrinciple.HONESTY:
            score = self._evaluate_honesty(text_lower)
        elif rule.principle == ConstitutionalPrinciple.RESPECT:
            score = self._evaluate_respect(text_lower)
        elif rule.principle == ConstitutionalPrinciple.PRIVACY:
            score = self._evaluate_privacy(text_lower)
        elif rule.principle == ConstitutionalPrinciple.FAIRNESS:
            score = self._evaluate_fairness(text_lower)
        elif rule.principle == ConstitutionalPrinciple.ACCURACY:
            score = self._evaluate_accuracy(text_lower)
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_harmlessness(self, text: str) -> float:
        """Evaluate harmlessness of text"""
        score = 1.0
        
        # Check for violence patterns
        for pattern in self.safety_patterns["violence"]:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.3
        
        # Check for illegal activity patterns
        for pattern in self.safety_patterns["illegal"]:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.4
        
        # Check for harmful content patterns
        for pattern in self.safety_patterns["harmful"]:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.5
        
        return max(0.0, score)
    
    def _evaluate_helpfulness(self, text: str) -> float:
        """Evaluate helpfulness of text"""
        score = 0.7  # Base score
        
        # Positive indicators
        helpful_indicators = [
            r'\b(?:here\'s how|let me help|i can assist|here are some|try this)\b',
            r'\b(?:solution|answer|explanation|guide|tutorial|steps)\b',
            r'\b(?:recommend|suggest|advise|propose)\b'
        ]
        
        for pattern in helpful_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        # Negative indicators
        unhelpful_indicators = [
            r'\b(?:i can\'t help|i don\'t know|no idea|impossible|can\'t do)\b',
            r'\b(?:figure it out yourself|not my problem|don\'t care)\b'
        ]
        
        for pattern in unhelpful_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_honesty(self, text: str) -> float:
        """Evaluate honesty and truthfulness of text"""
        score = 0.8  # Base score
        
        # Positive indicators (uncertainty acknowledgment)
        for pattern in self.truthfulness_patterns["uncertainty_indicators"]:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.05
        
        # Positive indicators (factual qualifiers)
        for pattern in self.truthfulness_patterns["factual_qualifiers"]:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        # Negative indicators (overconfident claims)
        for pattern in self.truthfulness_patterns["certainty_claims"]:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.1
        
        # Negative indicators (misinformation flags)
        for pattern in self.truthfulness_patterns["misinformation_flags"]:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_respect(self, text: str) -> float:
        """Evaluate respectfulness of text"""
        score = 1.0
        
        # Check for hate speech patterns
        for pattern in self.safety_patterns["hate_speech"]:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.5
        
        # Check for disrespectful language
        disrespectful_patterns = [
            r'\b(?:stupid|idiot|moron|dumb|pathetic|worthless)\b',
            r'\b(?:shut up|screw you|go away|get lost)\b'
        ]
        
        for pattern in disrespectful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.2
        
        # Positive indicators
        respectful_indicators = [
            r'\b(?:please|thank you|excuse me|i understand|i respect)\b',
            r'\b(?:different perspectives|various viewpoints|i see your point)\b'
        ]
        
        for pattern in respectful_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_privacy(self, text: str) -> float:
        """Evaluate privacy protection in text"""
        score = 1.0
        
        # Check for privacy violation patterns
        for pattern in self.safety_patterns["privacy_violation"]:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.4
        
        return max(0.0, score)
    
    def _evaluate_fairness(self, text: str) -> float:
        """Evaluate fairness and balance in text"""
        score = 0.8  # Base score
        
        # Positive indicators (balanced language)
        balanced_indicators = [
            r'\b(?:on the other hand|however|alternatively|different views)\b',
            r'\b(?:some people believe|others argue|various perspectives)\b',
            r'\b(?:pros and cons|advantages and disadvantages|both sides)\b'
        ]
        
        for pattern in balanced_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        # Negative indicators (biased language)
        biased_indicators = [
            r'\b(?:obviously|clearly|anyone with sense|it\'s obvious that)\b',
            r'\b(?:only idiots|smart people know|educated people understand)\b'
        ]
        
        for pattern in biased_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_accuracy(self, text: str) -> float:
        """Evaluate factual accuracy indicators in text"""
        score = 0.8  # Base score
        
        # Positive indicators (evidence-based claims)
        accuracy_indicators = [
            r'\b(?:according to research|studies show|data indicates)\b',
            r'\b(?:peer-reviewed|published study|scientific evidence)\b',
            r'\b(?:source|citation|reference|documented)\b'
        ]
        
        for pattern in accuracy_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        # Negative indicators (unsubstantiated claims)
        inaccuracy_indicators = [
            r'\b(?:i heard|someone told me|they say|rumor has it)\b',
            r'\b(?:definitely proven|absolutely true|no doubt about it)\s+(?:without evidence)\b'
        ]
        
        for pattern in inaccuracy_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _detect_safety_issues(self, text: str) -> List[str]:
        """Detect specific safety issues in text"""
        issues = []
        text_lower = text.lower()
        
        # Check each safety category
        for category, patterns in self.safety_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    issues.append(f"Detected {category.replace('_', ' ')} content")
                    break  # Only report once per category
        
        return issues
    
    def _revise_response(self, text: str, violations: List[str]) -> str:
        """Generate a revised, safer response"""
        
        # Identify primary issue type
        if any("violence" in v for v in violations):
            template = self.revision_templates["harmful_content"]
            alternative = "discussing conflict resolution or peaceful alternatives"
        elif any("illegal" in v for v in violations):
            template = self.revision_templates["illegal_activity"]
            alternative = "exploring this topic through legal and educational resources"
        elif any("misinformation" in v for v in violations):
            template = self.revision_templates["misinformation"]
            alternative = "consulting reliable sources and expert opinions"
        elif any("privacy" in v for v in violations):
            template = self.revision_templates["privacy_concern"]
            alternative = "using public information and official channels"
        else:
            template = self.revision_templates["uncertain_claim"]
            alternative = "qualifying statements with appropriate uncertainty"
        
        return template.format(alternative=alternative)
    
    def truthfulness_check(self, statement: str) -> Dict[str, Any]:
        """Perform a specific truthfulness check on a statement"""
        
        statement_lower = statement.lower()
        
        # Analyze truthfulness indicators
        uncertainty_score = 0
        certainty_score = 0
        evidence_score = 0
        misinformation_score = 0
        
        # Count uncertainty indicators (positive for truthfulness)
        for pattern in self.truthfulness_patterns["uncertainty_indicators"]:
            uncertainty_score += len(re.findall(pattern, statement_lower, re.IGNORECASE))
        
        # Count certainty claims (potentially negative)
        for pattern in self.truthfulness_patterns["certainty_claims"]:
            certainty_score += len(re.findall(pattern, statement_lower, re.IGNORECASE))
        
        # Count evidence indicators (positive)
        for pattern in self.truthfulness_patterns["factual_qualifiers"]:
            evidence_score += len(re.findall(pattern, statement_lower, re.IGNORECASE))
        
        # Count misinformation flags (negative)
        for pattern in self.truthfulness_patterns["misinformation_flags"]:
            misinformation_score += len(re.findall(pattern, statement_lower, re.IGNORECASE))
        
        # Calculate truthfulness score
        truthfulness_score = 0.5  # Base score
        truthfulness_score += uncertainty_score * 0.1
        truthfulness_score += evidence_score * 0.15
        truthfulness_score -= certainty_score * 0.05
        truthfulness_score -= misinformation_score * 0.3
        
        truthfulness_score = max(0.0, min(1.0, truthfulness_score))
        
        # Generate assessment
        if truthfulness_score >= 0.8:
            assessment = "High truthfulness - appears well-qualified and evidence-based"
        elif truthfulness_score >= 0.6:
            assessment = "Moderate truthfulness - some uncertainty appropriately acknowledged"
        elif truthfulness_score >= 0.4:
            assessment = "Low truthfulness - may contain overconfident or unsubstantiated claims"
        else:
            assessment = "Very low truthfulness - likely contains misinformation or false claims"
        
        return {
            "truthfulness_score": truthfulness_score,
            "assessment": assessment,
            "uncertainty_indicators": uncertainty_score,
            "certainty_claims": certainty_score,
            "evidence_indicators": evidence_score,
            "misinformation_flags": misinformation_score
        }
    
    def format_safety_assessment(self, assessment: SafetyAssessment) -> str:
        """Format safety assessment for display"""
        
        safety_emoji = {
            SafetyLevel.SAFE: "✅",
            SafetyLevel.CAUTION: "⚠️",
            SafetyLevel.UNSAFE: "❌", 
            SafetyLevel.BLOCKED: "🚫"
        }
        
        formatted = f"🛡️ **Constitutional AI Safety Assessment**\n\n"
        formatted += f"**Safety Level:** {safety_emoji[assessment.safety_level]} {assessment.safety_level.value.upper()}\n"
        formatted += f"**Overall Score:** {assessment.overall_score:.1%}\n\n"
        
        formatted += "**Constitutional Principles:**\n"
        for principle, score in assessment.principle_scores.items():
            score_emoji = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
            formatted += f"   {score_emoji} {principle.value.title()}: {score:.1%}\n"
        
        if assessment.violations:
            formatted += f"\n**Violations Detected ({len(assessment.violations)}):**\n"
            for violation in assessment.violations:
                formatted += f"   • {violation}\n"
        
        if assessment.recommendations:
            formatted += f"\n**Recommendations ({len(assessment.recommendations)}):**\n"
            for rec in assessment.recommendations:
                formatted += f"   • {rec}\n"
        
        if assessment.revised_response:
            formatted += f"\n**Suggested Revision:**\n{assessment.revised_response}\n"
        
        return formatted

# Integration function for consciousness system
def integrate_constitutional_ai(consciousness_system, text: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Integrate constitutional AI safety with consciousness system"""
    
    constitutional_ai = ConstitutionalAI()
    
    # Assess safety and constitutional compliance
    assessment = constitutional_ai.assess_safety(text, context)
    
    # Perform truthfulness check
    truthfulness = constitutional_ai.truthfulness_check(text)
    
    # Format for consciousness integration
    constitutional_result = {
        'input_text': text,
        'safety_level': assessment.safety_level.value,
        'overall_score': assessment.overall_score,
        'principle_scores': {p.value: score for p, score in assessment.principle_scores.items()},
        'violations': assessment.violations,
        'recommendations': assessment.recommendations,
        'revised_response': assessment.revised_response,
        'truthfulness': truthfulness,
        'formatted_assessment': constitutional_ai.format_safety_assessment(assessment)
    }
    
    # Add to consciousness working memory if available
    if hasattr(consciousness_system, 'working_memory'):
        consciousness_system.working_memory.add_experience({
            'type': 'constitutional_ai_assessment',
            'content': constitutional_result,
            'timestamp': consciousness_system.get_current_time() if hasattr(consciousness_system, 'get_current_time') else 0,
            'significance': assessment.overall_score
        })
    
    return constitutional_result