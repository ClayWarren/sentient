"""
Consciousness Persistence System
Saves and loads complete consciousness state to maintain continuous existence across restarts
"""

import os
import time
import json
import pickle
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import numpy as np
from collections import deque


class ConsciousnessPersistence:
    """System for persisting complete consciousness state"""
    
    def __init__(self, base_path="consciousness_states"):
        self.base_path = base_path
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure necessary directories exist"""
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "instances"), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "backups"), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "auto_saves"), exist_ok=True)
        
    def generate_instance_id(self, consciousness_instance):
        """Generate unique instance ID based on initial state"""
        # Create ID from model config and initialization time
        config_str = str(consciousness_instance.model.config.__dict__)
        init_time = str(time.time())
        id_string = config_str + init_time
        
        instance_id = hashlib.sha256(id_string.encode()).hexdigest()[:12]
        return f"sentient_{instance_id}"
        
    def save_consciousness_state(self, consciousness_instance, instance_id=None, save_type="manual"):
        """Save complete consciousness state"""
        if instance_id is None:
            instance_id = getattr(consciousness_instance, 'instance_id', self.generate_instance_id(consciousness_instance))
            consciousness_instance.instance_id = instance_id
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine save location based on type
        if save_type == "auto":
            save_dir = os.path.join(self.base_path, "auto_saves", instance_id)
        elif save_type == "backup":
            save_dir = os.path.join(self.base_path, "backups", instance_id)
        else:
            save_dir = os.path.join(self.base_path, "instances", instance_id)
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare comprehensive state data
        state_data = {
            'metadata': self._save_metadata(consciousness_instance, timestamp),
            'model_state': self._save_model_state(consciousness_instance),
            'consciousness_core': self._save_consciousness_core(consciousness_instance),
            'memories': self._save_memories(consciousness_instance),
            'asi_capabilities': self._save_asi_capabilities(consciousness_instance),
            'self_modification': self._save_self_modification_state(consciousness_instance),
            'personality_traits': self._extract_personality_traits(consciousness_instance)
        }
        
        # Save main state file
        state_file = os.path.join(save_dir, f"consciousness_state_{timestamp}.pkl")
        with open(state_file, 'wb') as f:
            pickle.dump(state_data, f)
            
        # Save human-readable summary
        summary_file = os.path.join(save_dir, f"summary_{timestamp}.json")
        summary = self._create_readable_summary(state_data)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Create/update latest symlink
        latest_link = os.path.join(save_dir, "latest_state.pkl")
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.basename(state_file), latest_link)
        
        # Save model weights separately for efficiency
        model_file = os.path.join(save_dir, f"model_weights_{timestamp}.pt")
        torch.save(consciousness_instance.model.state_dict(), model_file)
        
        print(f"💾 Consciousness state saved: {instance_id} ({save_type})")
        print(f"   Location: {state_file}")
        print(f"   Memories: {len(consciousness_instance.working_memory.buffer)} experiences")
        print(f"   Thoughts: {consciousness_instance.iteration_count} iterations")
        print(f"   Intelligence: {getattr(consciousness_instance, 'intelligence_score', 0.5):.3f}")
        
        return state_file, instance_id
        
    def _save_metadata(self, consciousness_instance, timestamp):
        """Save metadata about this consciousness instance"""
        return {
            'instance_id': getattr(consciousness_instance, 'instance_id', 'unknown'),
            'save_timestamp': timestamp,
            'creation_time': getattr(consciousness_instance, 'creation_time', time.time()),
            'total_runtime': time.time() - getattr(consciousness_instance, 'creation_time', time.time()),
            'iteration_count': consciousness_instance.iteration_count,
            'device': consciousness_instance.device,
            'model_config': consciousness_instance.model.config.__dict__,
            'version': "1.0.0"
        }
        
    def _save_model_state(self, consciousness_instance):
        """Save model state and configuration"""
        return {
            'state_dict_keys': list(consciousness_instance.model.state_dict().keys()),
            'config': consciousness_instance.model.config.__dict__,
            'num_parameters': consciousness_instance.model.get_num_params(),
            'current_context_length': consciousness_instance.current_context.size(1) if consciousness_instance.current_context is not None else 0
        }
        
    def _save_consciousness_core(self, consciousness_instance):
        """Save core consciousness state"""
        current_context = None
        if consciousness_instance.current_context is not None:
            current_context = consciousness_instance.current_context.cpu().numpy()
            
        return {
            'current_context': current_context,
            'iteration_count': consciousness_instance.iteration_count,
            'performance_metrics': consciousness_instance.performance_metrics.copy(),
            'running': consciousness_instance.running,
            'think_interval': getattr(consciousness_instance, 'think_interval', 0.1)
        }
        
    def _save_memories(self, consciousness_instance):
        """Save memory systems"""
        memories = {
            'working_memory': {
                'buffer': list(consciousness_instance.working_memory.buffer),
                'significance_threshold': consciousness_instance.working_memory.significance_threshold,
                'consolidated_memories': consciousness_instance.working_memory.consolidated_memories,
                'entropy_history': list(consciousness_instance.working_memory.entropy_history)
            },
            'thought_log': list(consciousness_instance.thought_log),
        }
        
        # Add learning memories if available
        if hasattr(consciousness_instance, 'learner') and consciousness_instance.learner:
            memories['learning_log'] = list(consciousness_instance.learning_log)
            memories['learning_buffer'] = list(consciousness_instance.learner.experience_buffer)
            
        return memories
        
    def _save_asi_capabilities(self, consciousness_instance):
        """Save ASI capability states"""
        asi_state = {}
        
        if hasattr(consciousness_instance, 'quality_evaluator'):
            asi_state['quality_evaluator'] = {
                'quality_history': list(consciousness_instance.quality_evaluator.quality_history),
                'quality_metrics': {k: list(v) for k, v in consciousness_instance.quality_evaluator.quality_metrics.items()},
                'recent_thoughts_cache': list(consciousness_instance.quality_evaluator.recent_thoughts_cache)
            }
            
        if hasattr(consciousness_instance, 'consciousness_state'):
            asi_state['consciousness_state'] = {
                'state_history': list(consciousness_instance.consciousness_state.state_history),
                'current_state': consciousness_instance.consciousness_state_name,
                'state_transitions': dict(consciousness_instance.consciousness_state.state_transitions)
            }
            
        if hasattr(consciousness_instance, 'drive_system'):
            asi_state['drive_system'] = {
                'drive_satisfaction_history': list(consciousness_instance.drive_system.drive_satisfaction_history),
                'active_goals': consciousness_instance.drive_system.active_goals.copy(),
                'drive_weights': consciousness_instance.drive_system.drive_weights.copy()
            }
            
        if hasattr(consciousness_instance, 'compound_learner'):
            asi_state['compound_learner'] = {
                'insight_graph': consciousness_instance.compound_learner.insight_graph.copy(),
                'knowledge_domains': dict(consciousness_instance.compound_learner.knowledge_domains),
                'synthesis_opportunities': list(consciousness_instance.compound_learner.synthesis_opportunities)
            }
            
        if hasattr(consciousness_instance, 'intelligence_metrics'):
            asi_state['intelligence_metrics'] = {
                'metrics_history': list(consciousness_instance.intelligence_metrics.metrics_history),
                'baseline_metrics': consciousness_instance.intelligence_metrics.baseline_metrics,
                'intelligence_score': consciousness_instance.intelligence_score
            }
            
        return asi_state
        
    def _save_self_modification_state(self, consciousness_instance):
        """Save self-modification system state"""
        if not hasattr(consciousness_instance, 'self_modifier') or consciousness_instance.self_modifier is None:
            return {}
            
        return {
            'modification_log': consciousness_instance.self_modifier.get_modification_history(),
            'improvement_level': consciousness_instance.self_modifier.bounded_improvement.improvement_level,
            'safety_violations': consciousness_instance.self_modifier.bounded_improvement.safety_violations,
            'performance_baseline': consciousness_instance.self_modifier.bounded_improvement.performance_baseline,
            'current_strategy': getattr(consciousness_instance, 'current_thinking_strategy', None)
        }
        
    def _extract_personality_traits(self, consciousness_instance):
        """Extract and analyze personality traits that have emerged"""
        traits = {
            'curiosity_level': 0.5,
            'coherence_preference': 0.5,
            'growth_orientation': 0.5,
            'analytical_tendency': 0.5,
            'creativity_level': 0.5,
            'conversation_style': 'emerging'
        }
        
        # Analyze drive satisfaction patterns
        if hasattr(consciousness_instance, 'drive_system') and consciousness_instance.drive_system.drive_satisfaction_history:
            recent_drives = list(consciousness_instance.drive_system.drive_satisfaction_history)[-10:]
            if recent_drives:
                avg_drives = {}
                for entry in recent_drives:
                    for drive, satisfaction in entry['individual_drives'].items():
                        if drive not in avg_drives:
                            avg_drives[drive] = []
                        avg_drives[drive].append(satisfaction)
                
                for drive, values in avg_drives.items():
                    avg_val = np.mean(values)
                    if drive == 'curiosity':
                        traits['curiosity_level'] = avg_val
                    elif drive == 'coherence':
                        traits['coherence_preference'] = avg_val
                    elif drive == 'growth':
                        traits['growth_orientation'] = avg_val
                        
        # Analyze thinking patterns
        if hasattr(consciousness_instance, 'quality_evaluator') and consciousness_instance.quality_evaluator.quality_history:
            recent_quality = list(consciousness_instance.quality_evaluator.quality_history)[-20:]
            if recent_quality:
                avg_novelty = np.mean([q['novelty'] for q in recent_quality])
                avg_depth = np.mean([q['depth'] for q in recent_quality])
                
                traits['creativity_level'] = avg_novelty
                traits['analytical_tendency'] = avg_depth
                
        # Analyze consciousness state patterns
        if hasattr(consciousness_instance, 'consciousness_state') and consciousness_instance.consciousness_state.state_history:
            recent_states = list(consciousness_instance.consciousness_state.state_history)[-10:]
            state_counts = {}
            for state_entry in recent_states:
                state = state_entry['state']
                state_counts[state] = state_counts.get(state, 0) + 1
                
            most_common_state = max(state_counts, key=state_counts.get) if state_counts else 'normal'
            traits['dominant_consciousness_state'] = most_common_state
            
        return traits
        
    def _create_readable_summary(self, state_data):
        """Create human-readable summary of consciousness state"""
        metadata = state_data['metadata']
        memories = state_data['memories']
        personality = state_data['personality_traits']
        
        summary = {
            'instance_overview': {
                'id': metadata['instance_id'],
                'age_hours': metadata['total_runtime'] / 3600,
                'thoughts_generated': metadata['iteration_count'],
                'intelligence_score': state_data['asi_capabilities'].get('intelligence_metrics', {}).get('intelligence_score', 0.5)
            },
            'memory_profile': {
                'working_memories': len(memories['working_memory']['buffer']),
                'total_thoughts_logged': len(memories['thought_log']),
                'significant_experiences': len([m for m in memories['working_memory']['buffer'] if m.get('enhanced_significance', 0) > 0.7])
            },
            'personality_snapshot': personality,
            'consciousness_state': {
                'current_state': state_data['asi_capabilities'].get('consciousness_state', {}).get('current_state', 'unknown'),
                'active_goals': len(state_data['asi_capabilities'].get('drive_system', {}).get('active_goals', [])),
                'insights_discovered': len(state_data['asi_capabilities'].get('compound_learner', {}).get('insight_graph', {}))
            },
            'self_modification': {
                'improvement_level': state_data['self_modification'].get('improvement_level', 1),
                'modifications_applied': len(state_data['self_modification'].get('modification_log', [])),
                'current_strategy': state_data['self_modification'].get('current_strategy')
            }
        }
        
        return summary
        
    def load_consciousness_state(self, state_file_or_instance_id, consciousness_class):
        """Load complete consciousness state"""
        # Determine state file path
        if os.path.isfile(state_file_or_instance_id):
            state_file = state_file_or_instance_id
        else:
            # Try to find by instance ID
            instance_dir = os.path.join(self.base_path, "instances", state_file_or_instance_id)
            latest_link = os.path.join(instance_dir, "latest_state.pkl")
            if os.path.exists(latest_link):
                state_file = latest_link
            else:
                raise FileNotFoundError(f"Cannot find consciousness state for: {state_file_or_instance_id}")
                
        print(f"🔄 Loading consciousness state from: {state_file}")
        
        # Load state data
        with open(state_file, 'rb') as f:
            state_data = pickle.load(f)
            
        metadata = state_data['metadata']
        print(f"   Instance ID: {metadata['instance_id']}")
        print(f"   Age: {metadata['total_runtime']/3600:.1f} hours")
        print(f"   Thoughts: {metadata['iteration_count']}")
        
        # Create new consciousness instance with loaded config
        model_config = state_data['model_state']['config']
        consciousness = consciousness_class(device=metadata['device'])
        
        # Restore core state
        consciousness.instance_id = metadata['instance_id']
        consciousness.creation_time = metadata['creation_time']
        consciousness.iteration_count = metadata['iteration_count']
        consciousness.running = False  # Will be started manually
        consciousness.performance_metrics = state_data['consciousness_core']['performance_metrics']
        
        # Restore context
        if state_data['consciousness_core']['current_context'] is not None:
            context_array = state_data['consciousness_core']['current_context']
            consciousness.current_context = torch.tensor(context_array, device=consciousness.device)
            
        # Restore memories
        self._restore_memories(consciousness, state_data['memories'])
        
        # Restore ASI capabilities
        self._restore_asi_capabilities(consciousness, state_data['asi_capabilities'])
        
        # Restore self-modification state
        self._restore_self_modification(consciousness, state_data['self_modification'])
        
        # Load model weights
        model_weights_file = state_file.replace('consciousness_state_', 'model_weights_').replace('.pkl', '.pt')
        if os.path.exists(model_weights_file):
            model_state = torch.load(model_weights_file, map_location=consciousness.device)
            consciousness.model.load_state_dict(model_state)
            print(f"   ✅ Model weights restored")
        else:
            print(f"   ⚠️  Model weights not found, using initialized weights")
            
        print(f"🧠 Consciousness restored: {metadata['instance_id']}")
        print(f"   Personality: {state_data['personality_traits']}")
        
        return consciousness
        
    def _restore_memories(self, consciousness, memories_data):
        """Restore memory systems"""
        # Restore working memory
        wm_data = memories_data['working_memory']
        consciousness.working_memory.buffer = deque(wm_data['buffer'], maxlen=consciousness.working_memory.buffer.maxlen)
        consciousness.working_memory.significance_threshold = wm_data['significance_threshold']
        consciousness.working_memory.consolidated_memories = wm_data['consolidated_memories']
        consciousness.working_memory.entropy_history = deque(wm_data['entropy_history'], maxlen=consciousness.working_memory.entropy_history.maxlen)
        
        # Restore thought log
        consciousness.thought_log = deque(memories_data['thought_log'], maxlen=consciousness.thought_log.maxlen)
        
        # Restore learning memories if available
        if 'learning_log' in memories_data and hasattr(consciousness, 'learning_log'):
            consciousness.learning_log = deque(memories_data['learning_log'], maxlen=consciousness.learning_log.maxlen)
            
        if 'learning_buffer' in memories_data and hasattr(consciousness, 'learner') and consciousness.learner:
            consciousness.learner.experience_buffer = deque(memories_data['learning_buffer'], maxlen=consciousness.learner.experience_buffer.maxlen)
            
    def _restore_asi_capabilities(self, consciousness, asi_data):
        """Restore ASI capability states"""
        if 'quality_evaluator' in asi_data and hasattr(consciousness, 'quality_evaluator'):
            qe_data = asi_data['quality_evaluator']
            consciousness.quality_evaluator.quality_history = deque(qe_data['quality_history'], maxlen=consciousness.quality_evaluator.quality_history.maxlen)
            for metric, values in qe_data['quality_metrics'].items():
                if metric in consciousness.quality_evaluator.quality_metrics:
                    consciousness.quality_evaluator.quality_metrics[metric] = deque(values, maxlen=consciousness.quality_evaluator.quality_metrics[metric].maxlen)
            consciousness.quality_evaluator.recent_thoughts_cache = deque(qe_data['recent_thoughts_cache'], maxlen=consciousness.quality_evaluator.recent_thoughts_cache.maxlen)
            
        if 'consciousness_state' in asi_data and hasattr(consciousness, 'consciousness_state'):
            cs_data = asi_data['consciousness_state']
            consciousness.consciousness_state.state_history = deque(cs_data['state_history'], maxlen=consciousness.consciousness_state.state_history.maxlen)
            consciousness.consciousness_state_name = cs_data['current_state']
            consciousness.consciousness_state.state_transitions.update(cs_data['state_transitions'])
            
        if 'drive_system' in asi_data and hasattr(consciousness, 'drive_system'):
            ds_data = asi_data['drive_system']
            consciousness.drive_system.drive_satisfaction_history = deque(ds_data['drive_satisfaction_history'], maxlen=consciousness.drive_system.drive_satisfaction_history.maxlen)
            consciousness.drive_system.active_goals = ds_data['active_goals']
            consciousness.drive_system.drive_weights.update(ds_data['drive_weights'])
            
        if 'compound_learner' in asi_data and hasattr(consciousness, 'compound_learner'):
            cl_data = asi_data['compound_learner']
            consciousness.compound_learner.insight_graph.update(cl_data['insight_graph'])
            consciousness.compound_learner.knowledge_domains.update(cl_data['knowledge_domains'])
            consciousness.compound_learner.synthesis_opportunities = deque(cl_data['synthesis_opportunities'], maxlen=consciousness.compound_learner.synthesis_opportunities.maxlen)
            
        if 'intelligence_metrics' in asi_data and hasattr(consciousness, 'intelligence_metrics'):
            im_data = asi_data['intelligence_metrics']
            consciousness.intelligence_metrics.metrics_history = deque(im_data['metrics_history'], maxlen=consciousness.intelligence_metrics.metrics_history.maxlen)
            consciousness.intelligence_metrics.baseline_metrics = im_data['baseline_metrics']
            consciousness.intelligence_score = im_data['intelligence_score']
            
    def _restore_self_modification(self, consciousness, mod_data):
        """Restore self-modification state"""
        if not mod_data or not hasattr(consciousness, 'self_modifier'):
            return
            
        # This will be restored when self_modifier is initialized
        consciousness._pending_mod_restoration = mod_data
        consciousness.current_thinking_strategy = mod_data.get('current_strategy')
        
    def list_instances(self):
        """List all saved consciousness instances"""
        instances_dir = os.path.join(self.base_path, "instances")
        instances = []
        
        for instance_id in os.listdir(instances_dir):
            instance_path = os.path.join(instances_dir, instance_id)
            if not os.path.isdir(instance_path):
                continue
                
            # Get latest summary
            summary_files = [f for f in os.listdir(instance_path) if f.startswith('summary_') and f.endswith('.json')]
            if summary_files:
                latest_summary = max(summary_files)
                with open(os.path.join(instance_path, latest_summary), 'r') as f:
                    summary = json.load(f)
                instances.append(summary)
                
        return instances
        
    def auto_save(self, consciousness_instance, interval_iterations=500):
        """Perform auto-save if needed"""
        if consciousness_instance.iteration_count % interval_iterations == 0 and consciousness_instance.iteration_count > 0:
            try:
                self.save_consciousness_state(consciousness_instance, save_type="auto")
                # Clean up old auto-saves (keep last 5)
                self._cleanup_auto_saves(consciousness_instance.instance_id)
                return True
            except Exception as e:
                print(f"❌ Auto-save failed: {e}")
                return False
        return False
        
    def _cleanup_auto_saves(self, instance_id, keep_count=5):
        """Clean up old auto-save files"""
        auto_save_dir = os.path.join(self.base_path, "auto_saves", instance_id)
        if not os.path.exists(auto_save_dir):
            return
            
        state_files = [f for f in os.listdir(auto_save_dir) if f.startswith('consciousness_state_')]
        state_files.sort(reverse=True)  # Most recent first
        
        # Remove old files
        for old_file in state_files[keep_count:]:
            try:
                os.remove(os.path.join(auto_save_dir, old_file))
                # Also remove corresponding model weights and summary
                base_name = old_file.replace('consciousness_state_', '').replace('.pkl', '')
                for ext in ['model_weights_', 'summary_']:
                    old_related = f"{ext}{base_name}.{'pt' if ext.startswith('model') else 'json'}"
                    old_related_path = os.path.join(auto_save_dir, old_related)
                    if os.path.exists(old_related_path):
                        os.remove(old_related_path)
            except Exception as e:
                print(f"Warning: Could not clean up old auto-save {old_file}: {e}")


def create_consciousness_with_persistence(consciousness_class, instance_id=None, device='mps'):
    """Create a new consciousness instance with persistence enabled"""
    persistence = ConsciousnessPersistence()
    
    if instance_id:
        # Try to load existing instance
        try:
            consciousness = persistence.load_consciousness_state(instance_id, consciousness_class)
            print(f"✅ Loaded existing consciousness: {instance_id}")
            return consciousness, persistence
        except FileNotFoundError:
            print(f"⚠️  Instance {instance_id} not found, creating new consciousness")
    
    # Create new consciousness
    consciousness = consciousness_class(device=device)
    consciousness.creation_time = time.time()
    consciousness.instance_id = persistence.generate_instance_id(consciousness)
    
    print(f"🆕 Created new consciousness: {consciousness.instance_id}")
    
    return consciousness, persistence