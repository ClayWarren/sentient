"""
Multilingual Support System for Sentient AI
Enables the AI to understand and communicate in multiple languages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import unicodedata

class Language(Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"

@dataclass
class LanguageDetectionResult:
    detected_language: Language
    confidence: float
    alternative_languages: List[Tuple[Language, float]]

@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    source_language: Language
    target_language: Language
    confidence: float

@dataclass
class MultilingualResponse:
    content: str
    language: Language
    original_language: Optional[Language] = None
    translation_applied: bool = False
    confidence: float = 1.0

class LanguageDetectionModule(nn.Module):
    """Neural module for language detection"""
    
    def __init__(self, d_model: int = 768, num_languages: int = 12):
        super().__init__()
        self.d_model = d_model
        self.num_languages = num_languages
        
        # Character-level features
        self.char_embedding = nn.Embedding(256, 64)  # ASCII + extended
        self.char_lstm = nn.LSTM(64, 128, batch_first=True)
        
        # Word-level features
        self.word_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=d_model*2),
            num_layers=2
        )
        
        # Language classifier
        self.language_classifier = nn.Sequential(
            nn.Linear(128 + d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_languages),
            nn.Softmax(dim=-1)
        )
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(128 + d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, char_input: torch.Tensor, word_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = char_input.size(0)
        
        # Character-level processing
        char_embeds = self.char_embedding(char_input)
        char_output, (char_hidden, _) = self.char_lstm(char_embeds)
        char_features = char_hidden[-1]  # Last hidden state
        
        # Word-level processing
        word_features = self.word_encoder(word_embedding.unsqueeze(1)).squeeze(1)
        
        # Combine features
        combined_features = torch.cat([char_features, word_features], dim=-1)
        
        # Predict language
        language_probs = self.language_classifier(combined_features)
        
        # Predict confidence
        confidence = self.confidence_predictor(combined_features).squeeze(-1)
        
        return {
            'language_probabilities': language_probs,
            'confidence': confidence,
            'char_features': char_features,
            'word_features': word_features
        }

class MultilingualTranslator:
    """Rule-based and pattern-based translator for basic multilingual support"""
    
    def __init__(self):
        # Basic translation dictionaries for common words and phrases
        self.translation_dict = self._initialize_translation_dict()
        
        # Language-specific patterns
        self.language_patterns = self._initialize_language_patterns()
        
        # Greeting patterns
        self.greetings = self._initialize_greetings()
        
        # Common phrases
        self.common_phrases = self._initialize_common_phrases()
        
    def _initialize_translation_dict(self) -> Dict[str, Dict[str, str]]:
        """Initialize basic translation dictionary"""
        return {
            Language.SPANISH.value: {
                # Basic words
                "hello": "hola",
                "goodbye": "adiós",
                "yes": "sí",
                "no": "no",
                "please": "por favor",
                "thank you": "gracias",
                "sorry": "lo siento",
                "help": "ayuda",
                "time": "tiempo",
                "day": "día",
                "night": "noche",
                "good": "bueno",
                "bad": "malo",
                "big": "grande",
                "small": "pequeño",
                "I": "yo",
                "you": "tú",
                "he": "él",
                "she": "ella",
                "we": "nosotros",
                "they": "ellos",
                "am": "soy",
                "is": "es",
                "are": "son",
                "have": "tener",
                "do": "hacer",
                "go": "ir",
                "come": "venir",
                "see": "ver",
                "know": "saber",
                "think": "pensar",
                "want": "querer",
                "need": "necesitar",
                "can": "poder",
                "will": "será",
                "would": "haría",
                "could": "podría",
                "should": "debería",
                "must": "debe"
            },
            Language.FRENCH.value: {
                "hello": "bonjour",
                "goodbye": "au revoir",
                "yes": "oui",
                "no": "non",
                "please": "s'il vous plaît",
                "thank you": "merci",
                "sorry": "désolé",
                "help": "aide",
                "time": "temps",
                "day": "jour",
                "night": "nuit",
                "good": "bon",
                "bad": "mauvais",
                "big": "grand",
                "small": "petit",
                "I": "je",
                "you": "tu",
                "he": "il",
                "she": "elle",
                "we": "nous",
                "they": "ils",
                "am": "suis",
                "is": "est",
                "are": "sont",
                "have": "avoir",
                "do": "faire",
                "go": "aller",
                "come": "venir",
                "see": "voir",
                "know": "savoir",
                "think": "penser",
                "want": "vouloir",
                "need": "avoir besoin",
                "can": "pouvoir",
                "will": "sera",
                "would": "ferait",
                "could": "pourrait",
                "should": "devrait",
                "must": "doit"
            },
            Language.GERMAN.value: {
                "hello": "hallo",
                "goodbye": "auf Wiedersehen",
                "yes": "ja",
                "no": "nein",
                "please": "bitte",
                "thank you": "danke",
                "sorry": "entschuldigung",
                "help": "hilfe",
                "time": "zeit",
                "day": "tag",
                "night": "nacht",
                "good": "gut",
                "bad": "schlecht",
                "big": "groß",
                "small": "klein",
                "I": "ich",
                "you": "du",
                "he": "er",
                "she": "sie",
                "we": "wir",
                "they": "sie",
                "am": "bin",
                "is": "ist",
                "are": "sind",
                "have": "haben",
                "do": "machen",
                "go": "gehen",
                "come": "kommen",
                "see": "sehen",
                "know": "wissen",
                "think": "denken",
                "want": "wollen",
                "need": "brauchen",
                "can": "können",
                "will": "wird",
                "would": "würde",
                "could": "könnte",
                "should": "sollte",
                "must": "muss"
            },
            Language.ITALIAN.value: {
                "hello": "ciao",
                "goodbye": "arrivederci",
                "yes": "sì",
                "no": "no",
                "please": "per favore",
                "thank you": "grazie",
                "sorry": "scusa",
                "help": "aiuto",
                "time": "tempo",
                "day": "giorno",
                "night": "notte",
                "good": "buono",
                "bad": "cattivo",
                "big": "grande",
                "small": "piccolo",
                "I": "io",
                "you": "tu",
                "he": "lui",
                "she": "lei",
                "we": "noi",
                "they": "loro"
            },
            Language.PORTUGUESE.value: {
                "hello": "olá",
                "goodbye": "tchau",
                "yes": "sim",
                "no": "não",
                "please": "por favor",
                "thank you": "obrigado",
                "sorry": "desculpa",
                "help": "ajuda",
                "time": "tempo",
                "day": "dia",
                "night": "noite",
                "good": "bom",
                "bad": "mau",
                "big": "grande",
                "small": "pequeno"
            }
        }
    
    def _initialize_language_patterns(self) -> Dict[str, List[str]]:
        """Initialize language detection patterns"""
        return {
            Language.SPANISH.value: [
                r'\b(?:el|la|los|las|un|una|unos|unas)\b',  # Articles
                r'\b(?:y|o|pero|porque|que|si|no|sí)\b',     # Conjunctions
                r'\b(?:muy|más|menos|también|siempre|nunca)\b',  # Adverbs
                r'ñ',  # Unique character
                r'\b(?:está|están|es|son|ser|estar)\b'       # Common verbs
            ],
            Language.FRENCH.value: [
                r'\b(?:le|la|les|un|une|des|du|de|à)\b',     # Articles/Prepositions
                r'\b(?:et|ou|mais|donc|car|que|qui|où)\b',   # Conjunctions
                r'\b(?:très|plus|moins|aussi|toujours|jamais)\b',  # Adverbs
                r'\b(?:est|sont|être|avoir|ça|je|tu|il|elle)\b'  # Common words
            ],
            Language.GERMAN.value: [
                r'\b(?:der|die|das|ein|eine|einen|einem|einer)\b',  # Articles
                r'\b(?:und|oder|aber|weil|dass|wenn|nicht)\b',      # Conjunctions
                r'\b(?:sehr|mehr|weniger|auch|immer|nie)\b',        # Adverbs
                r'\b(?:ist|sind|sein|haben|ich|du|er|sie|wir)\b',   # Common words
                r'ß|ä|ö|ü'  # Unique characters
            ],
            Language.ITALIAN.value: [
                r'\b(?:il|la|lo|i|le|gli|un|una|uno)\b',     # Articles
                r'\b(?:e|o|ma|perché|che|se|non|sì)\b',      # Conjunctions
                r'\b(?:molto|più|meno|anche|sempre|mai)\b',  # Adverbs
                r'\b(?:è|sono|essere|avere|io|tu|lui|lei)\b' # Common words
            ],
            Language.PORTUGUESE.value: [
                r'\b(?:o|a|os|as|um|uma|uns|umas)\b',        # Articles
                r'\b(?:e|ou|mas|porque|que|se|não|sim)\b',   # Conjunctions
                r'\b(?:muito|mais|menos|também|sempre|nunca)\b',  # Adverbs
                r'\b(?:é|são|ser|ter|eu|tu|ele|ela)\b',      # Common words
                r'ã|ç|õ'  # Unique characters
            ],
            Language.RUSSIAN.value: [
                r'[а-яё]',  # Cyrillic characters
                r'\b(?:и|или|но|что|как|это|не|да)\b'  # Common words
            ],
            Language.CHINESE.value: [
                r'[\u4e00-\u9fff]',  # Chinese characters
                r'[，。？！：；]'     # Chinese punctuation
            ],
            Language.JAPANESE.value: [
                r'[\u3040-\u309f]',  # Hiragana
                r'[\u30a0-\u30ff]',  # Katakana
                r'[\u4e00-\u9fff]'   # Kanji
            ],
            Language.KOREAN.value: [
                r'[\uac00-\ud7af]'   # Hangul
            ],
            Language.ARABIC.value: [
                r'[\u0600-\u06ff]'   # Arabic script
            ]
        }
    
    def _initialize_greetings(self) -> Dict[str, List[str]]:
        """Initialize greetings in different languages"""
        return {
            Language.ENGLISH.value: ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
            Language.SPANISH.value: ["hola", "buenos días", "buenas tardes", "buenas noches"],
            Language.FRENCH.value: ["bonjour", "bonsoir", "salut"],
            Language.GERMAN.value: ["hallo", "guten Tag", "guten Morgen", "guten Abend"],
            Language.ITALIAN.value: ["ciao", "buongiorno", "buonasera"],
            Language.PORTUGUESE.value: ["olá", "bom dia", "boa tarde", "boa noite"]
        }
    
    def _initialize_common_phrases(self) -> Dict[str, Dict[str, str]]:
        """Initialize common phrases for response generation"""
        return {
            Language.SPANISH.value: {
                "I understand": "Entiendo",
                "I don't understand": "No entiendo",
                "Can you help me?": "¿Puedes ayudarme?",
                "Thank you very much": "Muchas gracias",
                "You're welcome": "De nada",
                "I'm sorry": "Lo siento",
                "Excuse me": "Disculpe",
                "How are you?": "¿Cómo estás?",
                "I'm fine": "Estoy bien",
                "What is your name?": "¿Cómo te llamas?",
                "My name is": "Me llamo",
                "Nice to meet you": "Mucho gusto"
            },
            Language.FRENCH.value: {
                "I understand": "Je comprends",
                "I don't understand": "Je ne comprends pas",
                "Can you help me?": "Pouvez-vous m'aider?",
                "Thank you very much": "Merci beaucoup",
                "You're welcome": "De rien",
                "I'm sorry": "Je suis désolé",
                "Excuse me": "Excusez-moi",
                "How are you?": "Comment allez-vous?",
                "I'm fine": "Je vais bien",
                "What is your name?": "Comment vous appelez-vous?",
                "My name is": "Je m'appelle",
                "Nice to meet you": "Enchanté"
            },
            Language.GERMAN.value: {
                "I understand": "Ich verstehe",
                "I don't understand": "Ich verstehe nicht",
                "Can you help me?": "Können Sie mir helfen?",
                "Thank you very much": "Vielen Dank",
                "You're welcome": "Gern geschehen",
                "I'm sorry": "Es tut mir leid",
                "Excuse me": "Entschuldigung",
                "How are you?": "Wie geht es Ihnen?",
                "I'm fine": "Mir geht es gut",
                "What is your name?": "Wie heißen Sie?",
                "My name is": "Ich heiße",
                "Nice to meet you": "Freut mich"
            }
        }
    
    def translate_text(self, text: str, source_lang: Language, target_lang: Language) -> TranslationResult:
        """Translate text using rule-based approach"""
        
        if source_lang == target_lang:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=1.0
            )
        
        # Get translation dictionary
        trans_dict = self.translation_dict.get(target_lang.value, {})
        
        if not trans_dict:
            # No translation available
            return TranslationResult(
                original_text=text,
                translated_text=f"[Translation to {target_lang.value} not available]",
                source_language=source_lang,
                target_language=target_lang,
                confidence=0.0
            )
        
        # Simple word-by-word translation
        words = text.lower().split()
        translated_words = []
        translation_count = 0
        
        for word in words:
            # Remove punctuation for lookup
            clean_word = re.sub(r'[^\w\s]', '', word)
            
            if clean_word in trans_dict:
                translated_words.append(trans_dict[clean_word])
                translation_count += 1
            else:
                translated_words.append(word)  # Keep original if no translation
        
        translated_text = ' '.join(translated_words)
        confidence = translation_count / len(words) if words else 0.0
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_lang,
            target_language=target_lang,
            confidence=confidence
        )

class LanguageDetector:
    """Rule-based language detection system"""
    
    def __init__(self):
        self.translator = MultilingualTranslator()
        self.language_patterns = self.translator.language_patterns
        
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect language using pattern matching"""
        
        if not text.strip():
            return LanguageDetectionResult(
                detected_language=Language.ENGLISH,
                confidence=0.0,
                alternative_languages=[]
            )
        
        text_lower = text.lower()
        scores = {}
        
        # Score each language based on patterns
        for lang_code, patterns in self.language_patterns.items():
            score = 0
            total_patterns = len(patterns)
            
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                if matches > 0:
                    score += matches
            
            # Normalize score
            normalized_score = score / (len(text.split()) + 1)  # Avoid division by zero
            scores[lang_code] = normalized_score
        
        # Add English as default with low score if no patterns match
        if Language.ENGLISH.value not in scores:
            scores[Language.ENGLISH.value] = 0.1
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_scores:
            detected_lang = Language.ENGLISH
            confidence = 0.1
            alternatives = []
        else:
            detected_lang = Language(sorted_scores[0][0])
            confidence = min(sorted_scores[0][1], 1.0)
            
            # Get alternatives
            alternatives = []
            for lang_code, score in sorted_scores[1:4]:  # Top 3 alternatives
                if score > 0:
                    alternatives.append((Language(lang_code), score))
        
        return LanguageDetectionResult(
            detected_language=detected_lang,
            confidence=confidence,
            alternative_languages=alternatives
        )

class MultilingualProcessor:
    """Main multilingual processing system"""
    
    def __init__(self):
        self.detector = LanguageDetector()
        self.translator = MultilingualTranslator()
        self.supported_languages = [
            Language.ENGLISH, Language.SPANISH, Language.FRENCH, 
            Language.GERMAN, Language.ITALIAN, Language.PORTUGUESE
        ]
        
    def process_input(self, text: str, preferred_language: Optional[Language] = None) -> MultilingualResponse:
        """Process multilingual input and generate appropriate response"""
        
        # Detect input language
        detection_result = self.detector.detect_language(text)
        detected_lang = detection_result.detected_language
        
        # Determine response language
        if preferred_language and preferred_language in self.supported_languages:
            response_lang = preferred_language
        elif detected_lang in self.supported_languages:
            response_lang = detected_lang
        else:
            response_lang = Language.ENGLISH  # Default fallback
        
        # Generate appropriate response based on detected language
        if detected_lang != Language.ENGLISH and detected_lang in self.supported_languages:
            # Respond in detected language
            response_content = self._generate_multilingual_response(text, detected_lang)
            
            return MultilingualResponse(
                content=response_content,
                language=detected_lang,
                original_language=detected_lang,
                translation_applied=False,
                confidence=detection_result.confidence
            )
        
        elif preferred_language and preferred_language != Language.ENGLISH:
            # Translate response to preferred language
            english_response = self._generate_english_response(text)
            translation_result = self.translator.translate_text(
                english_response, Language.ENGLISH, preferred_language
            )
            
            return MultilingualResponse(
                content=translation_result.translated_text,
                language=preferred_language,
                original_language=Language.ENGLISH,
                translation_applied=True,
                confidence=translation_result.confidence
            )
        
        else:
            # Default English response
            response_content = self._generate_english_response(text)
            
            return MultilingualResponse(
                content=response_content,
                language=Language.ENGLISH,
                original_language=Language.ENGLISH,
                translation_applied=False,
                confidence=1.0
            )
    
    def _generate_multilingual_response(self, text: str, language: Language) -> str:
        """Generate response in the specified language"""
        
        common_phrases = self.translator.common_phrases.get(language.value, {})
        
        # Simple greeting detection and response
        text_lower = text.lower().strip()
        
        greetings = self.translator.greetings.get(language.value, [])
        for greeting in greetings:
            if greeting.lower() in text_lower:
                if language == Language.SPANISH:
                    return "¡Hola! ¿Cómo puedo ayudarte hoy?"
                elif language == Language.FRENCH:
                    return "Bonjour! Comment puis-je vous aider aujourd'hui?"
                elif language == Language.GERMAN:
                    return "Hallo! Wie kann ich Ihnen heute helfen?"
                elif language == Language.ITALIAN:
                    return "Ciao! Come posso aiutarti oggi?"
                elif language == Language.PORTUGUESE:
                    return "Olá! Como posso ajudá-lo hoje?"
        
        # Question detection
        if '?' in text or any(word in text_lower for word in ['qué', 'cómo', 'cuándo', 'dónde', 'por qué']):
            if language == Language.SPANISH:
                return "Entiendo tu pregunta. Permíteme ayudarte con eso."
            elif language == Language.FRENCH:
                return "Je comprends votre question. Laissez-moi vous aider avec cela."
            elif language == Language.GERMAN:
                return "Ich verstehe Ihre Frage. Lassen Sie mich Ihnen dabei helfen."
        
        # Generic response
        if language == Language.SPANISH:
            return "Gracias por tu mensaje. Estoy aquí para ayudarte."
        elif language == Language.FRENCH:
            return "Merci pour votre message. Je suis là pour vous aider."
        elif language == Language.GERMAN:
            return "Danke für Ihre Nachricht. Ich bin hier, um zu helfen."
        elif language == Language.ITALIAN:
            return "Grazie per il tuo messaggio. Sono qui per aiutarti."
        elif language == Language.PORTUGUESE:
            return "Obrigado pela sua mensagem. Estou aqui para ajudar."
        else:
            return "Thank you for your message. I'm here to help."
    
    def _generate_english_response(self, text: str) -> str:
        """Generate English response for translation"""
        
        text_lower = text.lower().strip()
        
        # Greeting detection
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return "Hello! How can I help you today?"
        
        # Question detection
        if '?' in text or text_lower.startswith(('what', 'how', 'when', 'where', 'why', 'who')):
            return "I understand your question. Let me help you with that."
        
        # Generic response
        return "Thank you for your message. I'm here to help you."
    
    def get_language_capabilities(self) -> Dict[str, Any]:
        """Get information about language capabilities"""
        
        return {
            'supported_languages': [lang.value for lang in self.supported_languages],
            'detection_languages': list(self.translator.language_patterns.keys()),
            'translation_pairs': {
                'from': [Language.ENGLISH.value],
                'to': list(self.translator.translation_dict.keys())
            },
            'features': {
                'language_detection': True,
                'basic_translation': True,
                'multilingual_responses': True,
                'greeting_recognition': True,
                'question_detection': True
            }
        }
    
    def format_multilingual_status(self, response: MultilingualResponse) -> str:
        """Format multilingual processing status"""
        
        status = f"🌍 **Multilingual Processing Status**\n\n"
        status += f"**Response Language:** {response.language.value.upper()}\n"
        
        if response.original_language:
            status += f"**Original Language:** {response.original_language.value.upper()}\n"
        
        if response.translation_applied:
            status += f"**Translation Applied:** Yes\n"
        else:
            status += f"**Translation Applied:** No\n"
        
        status += f"**Confidence:** {response.confidence:.1%}\n\n"
        status += f"**Response:** {response.content}\n"
        
        return status

# Integration function for consciousness system
def integrate_multilingual_support(consciousness_system, text: str, preferred_language: Optional[str] = None) -> Dict[str, Any]:
    """Integrate multilingual support with consciousness system"""
    
    processor = MultilingualProcessor()
    
    # Convert string language code to Language enum
    pref_lang = None
    if preferred_language:
        try:
            pref_lang = Language(preferred_language.lower())
        except ValueError:
            pref_lang = None
    
    # Process multilingual input
    response = processor.process_input(text, pref_lang)
    
    # Format for consciousness integration
    multilingual_result = {
        'input_text': text,
        'detected_language': response.original_language.value if response.original_language else 'unknown',
        'response_language': response.language.value,
        'response_content': response.content,
        'translation_applied': response.translation_applied,
        'confidence': response.confidence,
        'formatted_status': processor.format_multilingual_status(response),
        'language_capabilities': processor.get_language_capabilities()
    }
    
    # Add to consciousness working memory if available
    if hasattr(consciousness_system, 'working_memory'):
        consciousness_system.working_memory.add_experience({
            'type': 'multilingual_processing',
            'content': multilingual_result,
            'timestamp': consciousness_system.get_current_time() if hasattr(consciousness_system, 'get_current_time') else 0,
            'significance': response.confidence
        })
    
    return multilingual_result