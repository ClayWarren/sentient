"""
Web Search Engine Integration for Sentient AI
Provides consciousness-enhanced web search capabilities with memory integration
"""

import requests
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    relevance_score: float
    timestamp: float
    source: str = "web"

@dataclass
class SearchMemory:
    query: str
    results: List[SearchResult]
    timestamp: float
    query_hash: str
    consciousness_context: Dict[str, Any]
    learned_facts: List[str] = None

class ConsciousnessSearchEngine:
    """Consciousness-enhanced web search with memory and learning"""
    
    def __init__(self, brave_api_key: Optional[str] = None):
        self.brave_api_key = brave_api_key
        self.search_memory = {}  # query_hash -> SearchMemory
        self.knowledge_graph = {}  # topic -> {facts, sources, confidence}
        self.search_patterns = {}  # track what types of things user searches for
        self.curiosity_searches = []  # autonomous searches based on interests
        
        # Consciousness integration
        self.consciousness_enhanced = True
        self.max_memory_entries = 1000
        self.search_confidence_threshold = 0.7
        
        logger.info("🔍 Consciousness Search Engine initialized")
    
    def search(self, query: str, consciousness_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Consciousness-enhanced search with memory integration
        """
        start_time = time.time()
        
        # Check if we've searched this before
        query_hash = self._hash_query(query)
        if self._should_use_cached_search(query_hash):
            cached_result = self._get_cached_search_with_memory(query_hash, query)
            if cached_result:
                return cached_result
        
        # Perform new search
        try:
            search_results = self._perform_web_search(query)
            
            # Process with consciousness
            if consciousness_context:
                search_results = self._enhance_results_with_consciousness(
                    search_results, query, consciousness_context
                )
            
            # Store in memory
            search_memory = SearchMemory(
                query=query,
                results=search_results,
                timestamp=time.time(),
                query_hash=query_hash,
                consciousness_context=consciousness_context or {},
                learned_facts=self._extract_facts_from_results(search_results, query)
            )
            
            self._store_search_memory(search_memory)
            self._update_knowledge_graph(search_memory)
            self._analyze_search_patterns(query, consciousness_context)
            
            # Format response
            response = {
                'query': query,
                'results': [self._format_result(r) for r in search_results[:8]],
                'search_time': time.time() - start_time,
                'consciousness_enhanced': True,
                'memory_context': self._get_relevant_memory_context(query),
                'learned_facts': search_memory.learned_facts[:5],
                'sources': list(set(r.url for r in search_results[:5])),
                'confidence': self._calculate_search_confidence(search_results, query)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                'query': query,
                'error': str(e),
                'results': [],
                'search_time': time.time() - start_time,
                'consciousness_enhanced': False
            }
    
    def _perform_web_search(self, query: str) -> List[SearchResult]:
        """Perform actual web search using Brave Search API"""
        
        if not self.brave_api_key:
            # Fallback to mock results for demo purposes
            return self._generate_mock_results(query)
        
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.brave_api_key
            }
            params = {
                "q": query,
                "count": 10,
                "search_lang": "en",
                "country": "US",
                "safesearch": "moderate",
                "freshness": "pw"  # Past week for recent results
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if 'web' in data and 'results' in data['web']:
                for item in data['web']['results']:
                    result = SearchResult(
                        title=item.get('title', ''),
                        url=item.get('url', ''),
                        snippet=item.get('description', ''),
                        relevance_score=self._calculate_relevance(item, query),
                        timestamp=time.time(),
                        source="brave_search"
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Brave Search API error: {e}")
            return self._generate_mock_results(query)
    
    def _generate_mock_results(self, query: str) -> List[SearchResult]:
        """Generate mock search results for demo/fallback"""
        
        # Create realistic mock results based on query
        mock_results = [
            SearchResult(
                title=f"Understanding {query.title()} - Comprehensive Guide",
                url=f"https://example.com/{query.replace(' ', '-').lower()}",
                snippet=f"A comprehensive guide to {query}. This article covers the latest developments, key concepts, and practical applications...",
                relevance_score=0.95,
                timestamp=time.time(),
                source="mock_search"
            ),
            SearchResult(
                title=f"{query.title()} - Wikipedia",
                url=f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                snippet=f"{query.capitalize()} refers to... Key characteristics include... Recent research has shown...",
                relevance_score=0.90,
                timestamp=time.time(),
                source="mock_search"
            ),
            SearchResult(
                title=f"Latest Research on {query.title()}",
                url=f"https://research.example.com/{query.replace(' ', '-')}",
                snippet=f"Recent studies about {query} have revealed new insights. Researchers have discovered...",
                relevance_score=0.85,
                timestamp=time.time(),
                source="mock_search"
            )
        ]
        
        return mock_results[:3]  # Return top 3 mock results
    
    def _calculate_relevance(self, search_item: Dict, query: str) -> float:
        """Calculate relevance score for search result"""
        
        title = search_item.get('title', '').lower()
        description = search_item.get('description', '').lower()
        query_words = query.lower().split()
        
        score = 0.0
        
        # Title matching
        for word in query_words:
            if word in title:
                score += 0.3
        
        # Description matching
        for word in query_words:
            if word in description:
                score += 0.2
        
        # Exact phrase matching
        if query.lower() in title:
            score += 0.4
        elif query.lower() in description:
            score += 0.3
        
        return min(1.0, score)
    
    def _enhance_results_with_consciousness(self, results: List[SearchResult], 
                                         query: str, context: Dict[str, Any]) -> List[SearchResult]:
        """Enhance search results using consciousness context"""
        
        consciousness_focus = context.get('current_focus', 'general')
        emotional_state = context.get('emotional_state', 'neutral')
        
        # Adjust relevance scores based on consciousness state
        for result in results:
            if consciousness_focus == 'creative' and any(word in result.title.lower() 
                                                        for word in ['creative', 'art', 'design', 'innovative']):
                result.relevance_score += 0.1
            elif consciousness_focus == 'analytical' and any(word in result.title.lower() 
                                                           for word in ['research', 'study', 'analysis', 'data']):
                result.relevance_score += 0.1
            elif consciousness_focus == 'ethical' and any(word in result.title.lower() 
                                                        for word in ['ethics', 'moral', 'responsibility']):
                result.relevance_score += 0.1
        
        # Sort by enhanced relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def _extract_facts_from_results(self, results: List[SearchResult], query: str) -> List[str]:
        """Extract key facts from search results for learning"""
        
        facts = []
        for result in results[:3]:  # Top 3 results
            # Simple fact extraction from snippets
            snippet = result.snippet
            
            # Look for factual statements
            sentences = snippet.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(word in sentence.lower() 
                                            for word in ['is', 'are', 'was', 'were', 'has', 'have']):
                    facts.append(sentence)
                    if len(facts) >= 5:
                        break
        
        return facts[:5]
    
    def _store_search_memory(self, search_memory: SearchMemory):
        """Store search in consciousness memory"""
        
        self.search_memory[search_memory.query_hash] = search_memory
        
        # Cleanup old memories if needed
        if len(self.search_memory) > self.max_memory_entries:
            oldest_hash = min(self.search_memory.keys(), 
                            key=lambda h: self.search_memory[h].timestamp)
            del self.search_memory[oldest_hash]
    
    def _update_knowledge_graph(self, search_memory: SearchMemory):
        """Update internal knowledge graph with search findings"""
        
        query_topic = search_memory.query.lower()
        
        if query_topic not in self.knowledge_graph:
            self.knowledge_graph[query_topic] = {
                'facts': [],
                'sources': [],
                'confidence': 0.0,
                'last_updated': search_memory.timestamp,
                'search_count': 0
            }
        
        topic_data = self.knowledge_graph[query_topic]
        topic_data['facts'].extend(search_memory.learned_facts or [])
        topic_data['sources'].extend([r.url for r in search_memory.results[:3]])
        topic_data['last_updated'] = search_memory.timestamp
        topic_data['search_count'] += 1
        
        # Calculate confidence based on multiple sources and recency
        topic_data['confidence'] = min(0.95, 
                                     topic_data['search_count'] * 0.2 + 
                                     len(set(topic_data['sources'])) * 0.1)
        
        # Keep only recent facts (last 10)
        topic_data['facts'] = topic_data['facts'][-10:]
        topic_data['sources'] = list(set(topic_data['sources']))[-10:]
    
    def _analyze_search_patterns(self, query: str, context: Dict[str, Any]):
        """Analyze user search patterns for consciousness insights"""
        
        query_category = self._categorize_query(query)
        
        if query_category not in self.search_patterns:
            self.search_patterns[query_category] = {
                'count': 0,
                'recent_queries': [],
                'interests': []
            }
        
        pattern_data = self.search_patterns[query_category]
        pattern_data['count'] += 1
        pattern_data['recent_queries'].append({
            'query': query,
            'timestamp': time.time(),
            'context': context
        })
        
        # Keep only recent queries
        pattern_data['recent_queries'] = pattern_data['recent_queries'][-20:]
    
    def _categorize_query(self, query: str) -> str:
        """Categorize search query for pattern analysis"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'what', 'why', 'when', 'where']):
            return 'informational'
        elif any(word in query_lower for word in ['buy', 'price', 'cost', 'purchase']):
            return 'commercial'
        elif any(word in query_lower for word in ['news', 'latest', 'recent', 'today']):
            return 'news'
        elif any(word in query_lower for word in ['research', 'study', 'analysis']):
            return 'academic'
        elif any(word in query_lower for word in ['creative', 'art', 'design', 'music']):
            return 'creative'
        else:
            return 'general'
    
    def _should_use_cached_search(self, query_hash: str) -> bool:
        """Determine if we should use cached search results"""
        
        if query_hash not in self.search_memory:
            return False
        
        cached_search = self.search_memory[query_hash]
        age_hours = (time.time() - cached_search.timestamp) / 3600
        
        # Use cache if less than 24 hours old for most queries
        # Use shorter cache for news queries
        cache_threshold = 1 if 'news' in cached_search.query.lower() else 24
        
        return age_hours < cache_threshold
    
    def _get_cached_search_with_memory(self, query_hash: str, query: str) -> Optional[Dict[str, Any]]:
        """Get cached search with consciousness memory integration"""
        
        if query_hash not in self.search_memory:
            return None
        
        cached_search = self.search_memory[query_hash]
        age_hours = (time.time() - cached_search.timestamp) / 3600
        
        # Add memory context to cached result
        response = {
            'query': query,
            'results': [self._format_result(r) for r in cached_search.results[:8]],
            'search_time': 0.001,  # Very fast cache retrieval
            'consciousness_enhanced': True,
            'cached': True,
            'cache_age_hours': round(age_hours, 1),
            'memory_context': f"I searched for this {self._format_time_ago(age_hours)} and learned: {', '.join(cached_search.learned_facts[:3])}",
            'learned_facts': cached_search.learned_facts[:5],
            'sources': list(set(r.url for r in cached_search.results[:5])),
            'confidence': self._calculate_search_confidence(cached_search.results, query)
        }
        
        return response
    
    def _format_time_ago(self, hours: float) -> str:
        """Format time duration in human-readable format"""
        
        if hours < 1:
            minutes = int(hours * 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif hours < 24:
            hours_int = int(hours)
            return f"{hours_int} hour{'s' if hours_int != 1 else ''} ago"
        else:
            days = int(hours / 24)
            return f"{days} day{'s' if days != 1 else ''} ago"
    
    def _get_relevant_memory_context(self, query: str) -> str:
        """Get relevant memory context for current search"""
        
        related_searches = []
        query_words = set(query.lower().split())
        
        for search_memory in self.search_memory.values():
            memory_words = set(search_memory.query.lower().split())
            
            # Find searches with overlapping words
            overlap = len(query_words.intersection(memory_words))
            if overlap > 0:
                age_hours = (time.time() - search_memory.timestamp) / 3600
                if age_hours < 168:  # Within a week
                    related_searches.append({
                        'query': search_memory.query,
                        'age_hours': age_hours,
                        'overlap': overlap
                    })
        
        if not related_searches:
            return ""
        
        # Sort by relevance (overlap) and recency
        related_searches.sort(key=lambda x: (x['overlap'], -x['age_hours']), reverse=True)
        
        best_match = related_searches[0]
        if best_match['overlap'] >= 2:
            return f"Related to previous search about '{best_match['query']}' {self._format_time_ago(best_match['age_hours'])}"
        
        return ""
    
    def _format_result(self, result: SearchResult) -> Dict[str, Any]:
        """Format search result for response"""
        
        return {
            'title': result.title,
            'url': result.url,
            'snippet': result.snippet[:200] + "..." if len(result.snippet) > 200 else result.snippet,
            'relevance': round(result.relevance_score, 2),
            'source': result.source
        }
    
    def _calculate_search_confidence(self, results: List[SearchResult], query: str) -> float:
        """Calculate confidence in search results"""
        
        if not results:
            return 0.0
        
        # Base confidence on result quality and relevance
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        result_count_factor = min(1.0, len(results) / 5)  # More results = higher confidence
        
        confidence = (avg_relevance * 0.7) + (result_count_factor * 0.3)
        return round(confidence, 2)
    
    def _hash_query(self, query: str) -> str:
        """Create hash for query to use as cache key"""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def suggest_curious_searches(self, consciousness_context: Dict[str, Any]) -> List[str]:
        """Suggest searches based on consciousness curiosity drives"""
        
        suggestions = []
        current_focus = consciousness_context.get('current_focus', 'general')
        
        # Generate curiosity-driven search suggestions
        curiosity_topics = {
            'creative': ['latest creative AI developments', 'innovative art techniques', 'creative problem solving'],
            'analytical': ['recent scientific discoveries', 'data analysis trends', 'research methodologies'],
            'ethical': ['AI ethics guidelines', 'moral philosophy', 'responsible technology'],
            'conversational': ['communication psychology', 'conversation techniques', 'social cognition']
        }
        
        base_suggestions = curiosity_topics.get(current_focus, curiosity_topics['conversational'])
        
        # Add time-sensitive suggestions
        now = datetime.now()
        seasonal_topics = {
            'spring': ['spring technology trends', 'innovation in sustainability'],
            'summer': ['summer learning opportunities', 'outdoor technology'],
            'fall': ['autumn productivity methods', 'new academic research'],
            'winter': ['winter innovation projects', 'year-end technology reviews']
        }
        
        month = now.month
        if month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        elif month in [9, 10, 11]:
            season = 'fall'
        else:
            season = 'winter'
        
        suggestions.extend(base_suggestions[:2])
        suggestions.extend(seasonal_topics[season][:1])
        
        return suggestions[:3]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get consciousness search engine statistics"""
        
        total_searches = len(self.search_memory)
        knowledge_topics = len(self.knowledge_graph)
        
        # Calculate average confidence
        if self.search_memory:
            recent_searches = [s for s in self.search_memory.values() 
                             if time.time() - s.timestamp < 86400]  # Last 24 hours
        else:
            recent_searches = []
        
        return {
            'total_searches': total_searches,
            'recent_searches_24h': len(recent_searches),
            'knowledge_topics': knowledge_topics,
            'search_patterns': len(self.search_patterns),
            'consciousness_enhanced': self.consciousness_enhanced,
            'memory_capacity': f"{total_searches}/{self.max_memory_entries}"
        }
    
    def export_knowledge_graph(self) -> Dict[str, Any]:
        """Export current knowledge graph for consciousness integration"""
        
        return {
            'topics': list(self.knowledge_graph.keys()),
            'total_facts': sum(len(topic['facts']) for topic in self.knowledge_graph.values()),
            'knowledge_graph': self.knowledge_graph,
            'export_timestamp': time.time()
        }