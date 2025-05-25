"""
ARC-AGI-2 Abstract Reasoning System for Sentient AI
The ultimate test of abstract intelligence and pattern recognition
Implements visual reasoning, rule induction, and systematic generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import itertools
from collections import defaultdict, Counter
import copy

class PatternType(Enum):
    SPATIAL = "spatial"
    COLOR = "color" 
    SHAPE = "shape"
    SIZE = "size"
    TOPOLOGY = "topology"
    TRANSFORMATION = "transformation"
    COUNTING = "counting"
    SYMMETRY = "symmetry"

class TransformationType(Enum):
    TRANSLATION = "translation"
    ROTATION = "rotation"
    REFLECTION = "reflection"
    SCALING = "scaling"
    COLOR_CHANGE = "color_change"
    SHAPE_COMPLETION = "shape_completion"
    PATTERN_EXTENSION = "pattern_extension"
    OBJECT_REMOVAL = "object_removal"
    OBJECT_ADDITION = "object_addition"
    CONDITIONAL_RULE = "conditional_rule"

@dataclass
class ARCGrid:
    grid: np.ndarray
    width: int
    height: int
    colors: set
    
    def __post_init__(self):
        self.width = self.grid.shape[1]
        self.height = self.grid.shape[0]
        self.colors = set(self.grid.flatten())

@dataclass
class ARCExample:
    input_grid: ARCGrid
    output_grid: ARCGrid
    
@dataclass
class ARCTask:
    task_id: str
    train_examples: List[ARCExample]
    test_input: ARCGrid
    test_output: Optional[ARCGrid] = None

@dataclass
class Pattern:
    pattern_type: PatternType
    description: str
    confidence: float
    parameters: Dict[str, Any]
    applicable_examples: List[int]

@dataclass
class Transformation:
    transformation_type: TransformationType
    description: str
    confidence: float
    rule: callable
    parameters: Dict[str, Any]

@dataclass
class ARCSolution:
    task: ARCTask
    predicted_output: ARCGrid
    confidence: float
    discovered_patterns: List[Pattern]
    applied_transformations: List[Transformation]
    reasoning_steps: List[str]
    alternative_solutions: List[ARCGrid]

class AbstractReasoningModule(nn.Module):
    """Neural module for abstract reasoning and pattern recognition"""
    
    def __init__(self, d_model: int = 768, grid_size: int = 30):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.max_colors = 10
        
        # Grid encoder - processes visual patterns
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(self.max_colors, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, d_model)
        )
        
        # Pattern recognition layers
        self.pattern_detector = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 12, dim_feedforward=d_model*4),
            num_layers=6
        )
        
        # Transformation predictor
        self.transformation_predictor = nn.Sequential(
            nn.Linear(d_model * 2, 512),  # Input and output embeddings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, len(TransformationType)),
            nn.Softmax(dim=-1)
        )
        
        # Rule abstraction network
        self.rule_abstractor = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def encode_grid(self, grid: np.ndarray) -> torch.Tensor:
        """Encode grid as one-hot tensor and process through CNN"""
        # Pad grid to standard size
        padded_grid = np.zeros((self.grid_size, self.grid_size))
        h, w = grid.shape
        padded_grid[:h, :w] = grid
        
        # One-hot encode colors
        one_hot = np.zeros((self.max_colors, self.grid_size, self.grid_size))
        for color in range(self.max_colors):
            one_hot[color] = (padded_grid == color).astype(float)
        
        # Convert to tensor and encode
        grid_tensor = torch.FloatTensor(one_hot).unsqueeze(0)
        return self.grid_encoder(grid_tensor)
    
    def forward(self, input_grids: List[np.ndarray], output_grids: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        # Encode all grids
        input_embeddings = torch.stack([self.encode_grid(grid) for grid in input_grids])
        output_embeddings = torch.stack([self.encode_grid(grid) for grid in output_grids])
        
        # Detect patterns across examples
        all_embeddings = torch.cat([input_embeddings, output_embeddings], dim=0)
        pattern_features = self.pattern_detector(all_embeddings.unsqueeze(1)).squeeze(1)
        
        # Predict transformations
        io_pairs = torch.cat([input_embeddings, output_embeddings], dim=1)
        transformation_probs = self.transformation_predictor(io_pairs)
        
        # Abstract rules
        rule_features = self.rule_abstractor(pattern_features.mean(dim=0, keepdim=True))
        
        # Predict confidence
        confidence = self.confidence_predictor(rule_features).squeeze(-1)
        
        return {
            'pattern_features': pattern_features,
            'transformation_probs': transformation_probs,
            'rule_features': rule_features,
            'confidence': confidence
        }

class VisualPatternAnalyzer:
    """Analyzes visual patterns in ARC grids"""
    
    def __init__(self):
        self.color_patterns = {}
        self.shape_patterns = {}
        self.spatial_patterns = {}
        
    def analyze_grid(self, grid: ARCGrid) -> List[Pattern]:
        """Comprehensive analysis of a single grid"""
        patterns = []
        
        # Color analysis
        patterns.extend(self._analyze_colors(grid))
        
        # Shape analysis
        patterns.extend(self._analyze_shapes(grid))
        
        # Spatial analysis
        patterns.extend(self._analyze_spatial_relations(grid))
        
        # Symmetry analysis
        patterns.extend(self._analyze_symmetries(grid))
        
        # Topology analysis
        patterns.extend(self._analyze_topology(grid))
        
        return patterns
    
    def _analyze_colors(self, grid: ARCGrid) -> List[Pattern]:
        """Analyze color patterns"""
        patterns = []
        
        # Color frequency analysis
        color_counts = Counter(grid.grid.flatten())
        total_pixels = grid.width * grid.height
        
        # Dominant color pattern
        if len(color_counts) > 1:
            most_common_color, count = color_counts.most_common(1)[0]
            if count / total_pixels > 0.5:
                patterns.append(Pattern(
                    pattern_type=PatternType.COLOR,
                    description=f"Dominant color {most_common_color} ({count}/{total_pixels} pixels)",
                    confidence=count / total_pixels,
                    parameters={'dominant_color': most_common_color, 'ratio': count / total_pixels},
                    applicable_examples=[]
                ))
        
        # Color segregation pattern
        unique_colors = len(color_counts) - (1 if 0 in color_counts else 0)  # Exclude background
        if unique_colors <= 3:
            patterns.append(Pattern(
                pattern_type=PatternType.COLOR,
                description=f"Limited color palette: {unique_colors} colors",
                confidence=0.8,
                parameters={'num_colors': unique_colors, 'colors': list(color_counts.keys())},
                applicable_examples=[]
            ))
        
        return patterns
    
    def _analyze_shapes(self, grid: ARCGrid) -> List[Pattern]:
        """Analyze geometric shapes and objects"""
        patterns = []
        
        # Connected component analysis
        objects = self._find_connected_components(grid.grid)
        
        if objects:
            # Object count pattern
            patterns.append(Pattern(
                pattern_type=PatternType.SHAPE,
                description=f"Contains {len(objects)} distinct objects",
                confidence=0.9,
                parameters={'object_count': len(objects), 'objects': objects},
                applicable_examples=[]
            ))
            
            # Object size analysis
            sizes = [len(obj) for obj in objects]
            if len(set(sizes)) == 1:  # All same size
                patterns.append(Pattern(
                    pattern_type=PatternType.SIZE,
                    description=f"All objects same size ({sizes[0]} pixels)",
                    confidence=0.95,
                    parameters={'uniform_size': sizes[0]},
                    applicable_examples=[]
                ))
            elif len(sizes) > 1:
                patterns.append(Pattern(
                    pattern_type=PatternType.SIZE,
                    description=f"Object sizes: {sorted(sizes)}",
                    confidence=0.7,
                    parameters={'sizes': sizes},
                    applicable_examples=[]
                ))
        
        return patterns
    
    def _analyze_spatial_relations(self, grid: ARCGrid) -> List[Pattern]:
        """Analyze spatial relationships between objects"""
        patterns = []
        
        objects = self._find_connected_components(grid.grid)
        if len(objects) >= 2:
            # Alignment patterns
            x_coords = []
            y_coords = []
            
            for obj in objects:
                obj_x = [pos[1] for pos in obj]
                obj_y = [pos[0] for pos in obj]
                x_coords.append(sum(obj_x) / len(obj_x))  # Center X
                y_coords.append(sum(obj_y) / len(obj_y))  # Center Y
            
            # Check for horizontal alignment
            if len(set(round(y, 1) for y in y_coords)) == 1:
                patterns.append(Pattern(
                    pattern_type=PatternType.SPATIAL,
                    description="Objects horizontally aligned",
                    confidence=0.9,
                    parameters={'alignment': 'horizontal'},
                    applicable_examples=[]
                ))
            
            # Check for vertical alignment
            if len(set(round(x, 1) for x in x_coords)) == 1:
                patterns.append(Pattern(
                    pattern_type=PatternType.SPATIAL,
                    description="Objects vertically aligned",
                    confidence=0.9,
                    parameters={'alignment': 'vertical'},
                    applicable_examples=[]
                ))
            
            # Check for grid pattern
            if len(objects) >= 4:
                x_unique = sorted(set(round(x, 1) for x in x_coords))
                y_unique = sorted(set(round(y, 1) for y in y_coords))
                if len(x_unique) * len(y_unique) == len(objects):
                    patterns.append(Pattern(
                        pattern_type=PatternType.SPATIAL,
                        description=f"Grid arrangement: {len(x_unique)}×{len(y_unique)}",
                        confidence=0.95,
                        parameters={'grid_dims': (len(x_unique), len(y_unique))},
                        applicable_examples=[]
                    ))
        
        return patterns
    
    def _analyze_symmetries(self, grid: ARCGrid) -> List[Pattern]:
        """Analyze symmetrical patterns"""
        patterns = []
        
        # Horizontal symmetry
        if np.array_equal(grid.grid, np.flipud(grid.grid)):
            patterns.append(Pattern(
                pattern_type=PatternType.SYMMETRY,
                description="Horizontal symmetry (top-bottom mirror)",
                confidence=1.0,
                parameters={'symmetry_type': 'horizontal'},
                applicable_examples=[]
            ))
        
        # Vertical symmetry
        if np.array_equal(grid.grid, np.fliplr(grid.grid)):
            patterns.append(Pattern(
                pattern_type=PatternType.SYMMETRY,
                description="Vertical symmetry (left-right mirror)",
                confidence=1.0,
                parameters={'symmetry_type': 'vertical'},
                applicable_examples=[]
            ))
        
        # Rotational symmetry (90 degrees)
        if np.array_equal(grid.grid, np.rot90(grid.grid)):
            patterns.append(Pattern(
                pattern_type=PatternType.SYMMETRY,
                description="90-degree rotational symmetry",
                confidence=1.0,
                parameters={'symmetry_type': 'rotation_90'},
                applicable_examples=[]
            ))
        
        return patterns
    
    def _analyze_topology(self, grid: ARCGrid) -> List[Pattern]:
        """Analyze topological properties"""
        patterns = []
        
        # Connectivity analysis
        objects = self._find_connected_components(grid.grid)
        
        # Hole detection
        for i, obj in enumerate(objects):
            if self._has_hole(obj, grid.grid):
                patterns.append(Pattern(
                    pattern_type=PatternType.TOPOLOGY,
                    description=f"Object {i} contains hole(s)",
                    confidence=0.9,
                    parameters={'object_id': i, 'has_hole': True},
                    applicable_examples=[]
                ))
        
        # Border contact analysis
        border_objects = 0
        for obj in objects:
            if self._touches_border(obj, grid.width, grid.height):
                border_objects += 1
        
        if border_objects > 0:
            patterns.append(Pattern(
                pattern_type=PatternType.TOPOLOGY,
                description=f"{border_objects}/{len(objects)} objects touch border",
                confidence=0.8,
                parameters={'border_contact_ratio': border_objects / len(objects)},
                applicable_examples=[]
            ))
        
        return patterns
    
    def _find_connected_components(self, grid: np.ndarray, background=0) -> List[List[Tuple[int, int]]]:
        """Find connected components (8-connectivity)"""
        visited = np.zeros_like(grid, dtype=bool)
        components = []
        
        def dfs(r, c, color, component):
            if (r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1] or 
                visited[r, c] or grid[r, c] != color):
                return
            
            visited[r, c] = True
            component.append((r, c))
            
            # 8-connectivity
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr != 0 or dc != 0:
                        dfs(r + dr, c + dc, color, component)
        
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if not visited[r, c] and grid[r, c] != background:
                    component = []
                    dfs(r, c, grid[r, c], component)
                    if component:
                        components.append(component)
        
        return components
    
    def _has_hole(self, obj: List[Tuple[int, int]], grid: np.ndarray) -> bool:
        """Check if object has holes using flood fill"""
        # Create bounding box
        min_r = min(pos[0] for pos in obj)
        max_r = max(pos[0] for pos in obj)
        min_c = min(pos[1] for pos in obj)
        max_c = max(pos[1] for pos in obj)
        
        # Extract subgrid
        subgrid = grid[min_r:max_r+1, min_c:max_c+1]
        obj_mask = np.zeros_like(subgrid, dtype=bool)
        
        for r, c in obj:
            obj_mask[r-min_r, c-min_c] = True
        
        # Flood fill from border to find external background
        visited = np.zeros_like(subgrid, dtype=bool)
        
        def flood_fill_external(r, c):
            if (r < 0 or r >= subgrid.shape[0] or c < 0 or c >= subgrid.shape[1] or 
                visited[r, c] or obj_mask[r, c]):
                return
            visited[r, c] = True
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                flood_fill_external(r + dr, c + dc)
        
        # Start flood fill from all border pixels
        for r in range(subgrid.shape[0]):
            if not obj_mask[r, 0] and not visited[r, 0]:
                flood_fill_external(r, 0)
            if not obj_mask[r, -1] and not visited[r, -1]:
                flood_fill_external(r, -1)
        
        for c in range(subgrid.shape[1]):
            if not obj_mask[0, c] and not visited[0, c]:
                flood_fill_external(0, c)
            if not obj_mask[-1, c] and not visited[-1, c]:
                flood_fill_external(-1, c)
        
        # Check if there are unvisited non-object pixels (holes)
        for r in range(subgrid.shape[0]):
            for c in range(subgrid.shape[1]):
                if not obj_mask[r, c] and not visited[r, c]:
                    return True
        
        return False
    
    def _touches_border(self, obj: List[Tuple[int, int]], width: int, height: int) -> bool:
        """Check if object touches grid border"""
        for r, c in obj:
            if r == 0 or r == height - 1 or c == 0 or c == width - 1:
                return True
        return False

class TransformationEngine:
    """Infers and applies transformations between input-output pairs"""
    
    def __init__(self):
        self.transformation_catalog = self._initialize_transformations()
        
    def _initialize_transformations(self) -> Dict[str, callable]:
        """Initialize catalog of transformation functions"""
        return {
            'translate': self._translate,
            'rotate_90': self._rotate_90,
            'rotate_180': self._rotate_180,
            'rotate_270': self._rotate_270,
            'flip_horizontal': self._flip_horizontal,
            'flip_vertical': self._flip_vertical,
            'scale_2x': self._scale_2x,
            'scale_half': self._scale_half,
            'color_replace': self._color_replace,
            'extract_objects': self._extract_objects,
            'fill_holes': self._fill_holes,
            'remove_noise': self._remove_noise,
            'connect_objects': self._connect_objects,
            'crop_to_content': self._crop_to_content,
            'extend_pattern': self._extend_pattern,
            'apply_mask': self._apply_mask
        }
    
    def infer_transformation(self, input_grid: ARCGrid, output_grid: ARCGrid) -> List[Transformation]:
        """Infer what transformation(s) were applied"""
        transformations = []
        
        # Size change detection
        if input_grid.width != output_grid.width or input_grid.height != output_grid.height:
            size_change = self._analyze_size_change(input_grid, output_grid)
            transformations.extend(size_change)
        
        # Geometric transformations
        geometric_transforms = self._test_geometric_transformations(input_grid, output_grid)
        transformations.extend(geometric_transforms)
        
        # Color transformations
        color_transforms = self._test_color_transformations(input_grid, output_grid)
        transformations.extend(color_transforms)
        
        # Object-level transformations
        object_transforms = self._test_object_transformations(input_grid, output_grid)
        transformations.extend(object_transforms)
        
        # Pattern completion
        pattern_transforms = self._test_pattern_completion(input_grid, output_grid)
        transformations.extend(pattern_transforms)
        
        return transformations
    
    def _analyze_size_change(self, input_grid: ARCGrid, output_grid: ARCGrid) -> List[Transformation]:
        """Analyze changes in grid dimensions"""
        transformations = []
        
        input_area = input_grid.width * input_grid.height
        output_area = output_grid.width * output_grid.height
        
        # Scaling
        if output_area == input_area * 4:
            transformations.append(Transformation(
                transformation_type=TransformationType.SCALING,
                description="Grid scaled 2x",
                confidence=0.9,
                rule=self._scale_2x,
                parameters={'scale_factor': 2}
            ))
        elif output_area == input_area // 4:
            transformations.append(Transformation(
                transformation_type=TransformationType.SCALING,
                description="Grid scaled 0.5x",
                confidence=0.9,
                rule=self._scale_half,
                parameters={'scale_factor': 0.5}
            ))
        
        # Cropping
        elif output_area < input_area:
            transformations.append(Transformation(
                transformation_type=TransformationType.SHAPE_COMPLETION,
                description="Grid cropped to content",
                confidence=0.8,
                rule=self._crop_to_content,
                parameters={}
            ))
        
        return transformations
    
    def _test_geometric_transformations(self, input_grid: ARCGrid, output_grid: ARCGrid) -> List[Transformation]:
        """Test various geometric transformations"""
        transformations = []
        
        if input_grid.width == output_grid.width and input_grid.height == output_grid.height:
            # Rotation tests
            if np.array_equal(output_grid.grid, np.rot90(input_grid.grid, 1)):
                transformations.append(Transformation(
                    transformation_type=TransformationType.ROTATION,
                    description="90-degree clockwise rotation",
                    confidence=1.0,
                    rule=self._rotate_90,
                    parameters={'angle': 90}
                ))
            elif np.array_equal(output_grid.grid, np.rot90(input_grid.grid, 2)):
                transformations.append(Transformation(
                    transformation_type=TransformationType.ROTATION,
                    description="180-degree rotation",
                    confidence=1.0,
                    rule=self._rotate_180,
                    parameters={'angle': 180}
                ))
            elif np.array_equal(output_grid.grid, np.rot90(input_grid.grid, 3)):
                transformations.append(Transformation(
                    transformation_type=TransformationType.ROTATION,
                    description="270-degree clockwise rotation",
                    confidence=1.0,
                    rule=self._rotate_270,
                    parameters={'angle': 270}
                ))
            
            # Reflection tests
            elif np.array_equal(output_grid.grid, np.fliplr(input_grid.grid)):
                transformations.append(Transformation(
                    transformation_type=TransformationType.REFLECTION,
                    description="Horizontal reflection",
                    confidence=1.0,
                    rule=self._flip_horizontal,
                    parameters={'axis': 'horizontal'}
                ))
            elif np.array_equal(output_grid.grid, np.flipud(input_grid.grid)):
                transformations.append(Transformation(
                    transformation_type=TransformationType.REFLECTION,
                    description="Vertical reflection",
                    confidence=1.0,
                    rule=self._flip_vertical,
                    parameters={'axis': 'vertical'}
                ))
        
        return transformations
    
    def _test_color_transformations(self, input_grid: ARCGrid, output_grid: ARCGrid) -> List[Transformation]:
        """Test color-based transformations"""
        transformations = []
        
        input_colors = Counter(input_grid.grid.flatten())
        output_colors = Counter(output_grid.grid.flatten())
        
        # Color replacement
        if len(input_colors) == len(output_colors) and input_grid.grid.shape == output_grid.grid.shape:
            # Check if it's a systematic color replacement
            color_mapping = {}
            consistent_mapping = True
            
            for r in range(input_grid.height):
                for c in range(input_grid.width):
                    in_color = input_grid.grid[r, c]
                    out_color = output_grid.grid[r, c]
                    
                    if in_color in color_mapping:
                        if color_mapping[in_color] != out_color:
                            consistent_mapping = False
                            break
                    else:
                        color_mapping[in_color] = out_color
                if not consistent_mapping:
                    break
            
            if consistent_mapping and len(color_mapping) > 1:
                transformations.append(Transformation(
                    transformation_type=TransformationType.COLOR_CHANGE,
                    description=f"Systematic color replacement: {color_mapping}",
                    confidence=0.95,
                    rule=self._color_replace,
                    parameters={'color_mapping': color_mapping}
                ))
        
        return transformations
    
    def _test_object_transformations(self, input_grid: ARCGrid, output_grid: ARCGrid) -> List[Transformation]:
        """Test object-level transformations"""
        transformations = []
        
        # Simplified object extraction test
        input_nonzero = np.count_nonzero(input_grid.grid)
        output_nonzero = np.count_nonzero(output_grid.grid)
        
        # Object removal
        if output_nonzero < input_nonzero * 0.8:
            transformations.append(Transformation(
                transformation_type=TransformationType.OBJECT_REMOVAL,
                description="Some objects removed",
                confidence=0.7,
                rule=self._remove_noise,
                parameters={'threshold': 'small_objects'}
            ))
        
        # Object addition
        elif output_nonzero > input_nonzero * 1.2:
            transformations.append(Transformation(
                transformation_type=TransformationType.OBJECT_ADDITION,
                description="Objects or patterns added",
                confidence=0.7,
                rule=self._extend_pattern,
                parameters={'method': 'addition'}
            ))
        
        return transformations
    
    def _test_pattern_completion(self, input_grid: ARCGrid, output_grid: ARCGrid) -> List[Transformation]:
        """Test pattern completion transformations"""
        transformations = []
        
        # Simple pattern completion heuristic
        if output_grid.width >= input_grid.width and output_grid.height >= input_grid.height:
            # Check if input is contained within output
            input_contained = False
            for r_offset in range(output_grid.height - input_grid.height + 1):
                for c_offset in range(output_grid.width - input_grid.width + 1):
                    subgrid = output_grid.grid[r_offset:r_offset+input_grid.height, 
                                            c_offset:c_offset+input_grid.width]
                    if np.array_equal(subgrid, input_grid.grid):
                        input_contained = True
                        break
                if input_contained:
                    break
            
            if input_contained:
                transformations.append(Transformation(
                    transformation_type=TransformationType.PATTERN_EXTENSION,
                    description="Pattern extended or completed",
                    confidence=0.8,
                    rule=self._extend_pattern,
                    parameters={'method': 'completion'}
                ))
        
        return transformations
    
    # Transformation implementation functions
    def _translate(self, grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate grid contents"""
        result = np.zeros_like(grid)
        h, w = grid.shape
        
        for r in range(h):
            for c in range(w):
                new_r, new_c = r + dy, c + dx
                if 0 <= new_r < h and 0 <= new_c < w:
                    result[new_r, new_c] = grid[r, c]
        
        return result
    
    def _rotate_90(self, grid: np.ndarray) -> np.ndarray:
        """Rotate 90 degrees clockwise"""
        return np.rot90(grid, -1)
    
    def _rotate_180(self, grid: np.ndarray) -> np.ndarray:
        """Rotate 180 degrees"""
        return np.rot90(grid, 2)
    
    def _rotate_270(self, grid: np.ndarray) -> np.ndarray:
        """Rotate 270 degrees clockwise"""
        return np.rot90(grid, 1)
    
    def _flip_horizontal(self, grid: np.ndarray) -> np.ndarray:
        """Flip horizontally"""
        return np.fliplr(grid)
    
    def _flip_vertical(self, grid: np.ndarray) -> np.ndarray:
        """Flip vertically"""
        return np.flipud(grid)
    
    def _scale_2x(self, grid: np.ndarray) -> np.ndarray:
        """Scale up by 2x"""
        return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
    
    def _scale_half(self, grid: np.ndarray) -> np.ndarray:
        """Scale down by 0.5x"""
        return grid[::2, ::2]
    
    def _color_replace(self, grid: np.ndarray, color_mapping: Dict[int, int]) -> np.ndarray:
        """Replace colors according to mapping"""
        result = grid.copy()
        for old_color, new_color in color_mapping.items():
            result[grid == old_color] = new_color
        return result
    
    def _extract_objects(self, grid: np.ndarray) -> np.ndarray:
        """Extract main objects (placeholder)"""
        return grid  # Simplified implementation
    
    def _fill_holes(self, grid: np.ndarray) -> np.ndarray:
        """Fill holes in objects (placeholder)"""
        return grid  # Simplified implementation
    
    def _remove_noise(self, grid: np.ndarray) -> np.ndarray:
        """Remove small isolated pixels"""
        result = grid.copy()
        h, w = grid.shape
        
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:  # Non-background pixel
                    # Check if isolated
                    neighbors = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == grid[r, c]:
                                neighbors += 1
                    
                    if neighbors == 0:  # Isolated pixel
                        result[r, c] = 0
        
        return result
    
    def _connect_objects(self, grid: np.ndarray) -> np.ndarray:
        """Connect nearby objects (placeholder)"""
        return grid  # Simplified implementation
    
    def _crop_to_content(self, grid: np.ndarray) -> np.ndarray:
        """Crop grid to non-zero content"""
        nonzero_r, nonzero_c = np.nonzero(grid)
        if len(nonzero_r) == 0:
            return grid
        
        min_r, max_r = nonzero_r.min(), nonzero_r.max()
        min_c, max_c = nonzero_c.min(), nonzero_c.max()
        
        return grid[min_r:max_r+1, min_c:max_c+1]
    
    def _extend_pattern(self, grid: np.ndarray) -> np.ndarray:
        """Extend or complete patterns (placeholder)"""
        return grid  # Simplified implementation
    
    def _apply_mask(self, grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to grid"""
        return grid * mask

class ARCAGI2Solver:
    """Main solver for ARC-AGI-2 tasks"""
    
    def __init__(self):
        self.reasoning_module = AbstractReasoningModule()
        self.pattern_analyzer = VisualPatternAnalyzer()
        self.transformation_engine = TransformationEngine()
        
        # Meta-learning components
        self.pattern_memory = []
        self.transformation_memory = []
        
    def parse_arc_task(self, task_data: Dict[str, Any]) -> ARCTask:
        """Parse ARC task from JSON format"""
        
        # Parse training examples
        train_examples = []
        for example in task_data.get('train', []):
            input_grid = ARCGrid(np.array(example['input']))
            output_grid = ARCGrid(np.array(example['output']))
            train_examples.append(ARCExample(input_grid, output_grid))
        
        # Parse test input
        test_data = task_data.get('test', [{}])[0]
        test_input = ARCGrid(np.array(test_data['input']))
        
        # Test output if available (for evaluation)
        test_output = None
        if 'output' in test_data:
            test_output = ARCGrid(np.array(test_data['output']))
        
        return ARCTask(
            task_id=task_data.get('id', 'unknown'),
            train_examples=train_examples,
            test_input=test_input,
            test_output=test_output
        )
    
    def solve_arc_task(self, task: ARCTask) -> ARCSolution:
        """Solve ARC-AGI-2 task using abstract reasoning"""
        
        reasoning_steps = []
        discovered_patterns = []
        applied_transformations = []
        alternative_solutions = []
        
        # Step 1: Pattern Discovery Phase
        reasoning_steps.append("🔍 Analyzing patterns across training examples")
        
        # Analyze each training example
        example_patterns = []
        for i, example in enumerate(task.train_examples):
            input_patterns = self.pattern_analyzer.analyze_grid(example.input_grid)
            output_patterns = self.pattern_analyzer.analyze_grid(example.output_grid)
            
            example_patterns.append({
                'input_patterns': input_patterns,
                'output_patterns': output_patterns,
                'example_id': i
            })
            
            reasoning_steps.append(f"   Example {i+1}: Found {len(input_patterns)} input patterns, {len(output_patterns)} output patterns")
        
        # Find consistent patterns across examples
        consistent_patterns = self._find_consistent_patterns(example_patterns)
        discovered_patterns.extend(consistent_patterns)
        
        reasoning_steps.append(f"🎯 Identified {len(consistent_patterns)} consistent patterns")
        
        # Step 2: Transformation Discovery Phase
        reasoning_steps.append("🔄 Discovering transformations between input-output pairs")
        
        example_transformations = []
        for i, example in enumerate(task.train_examples):
            transformations = self.transformation_engine.infer_transformation(
                example.input_grid, example.output_grid
            )
            example_transformations.append(transformations)
            
            if transformations:
                reasoning_steps.append(f"   Example {i+1}: Found {len(transformations)} possible transformations")
                for trans in transformations[:2]:  # Show top 2
                    reasoning_steps.append(f"      - {trans.description} (confidence: {trans.confidence:.2f})")
        
        # Find consistent transformations
        consistent_transformations = self._find_consistent_transformations(example_transformations)
        applied_transformations.extend(consistent_transformations)
        
        reasoning_steps.append(f"⚡ Identified {len(consistent_transformations)} consistent transformations")
        
        # Step 3: Rule Synthesis Phase
        reasoning_steps.append("🧠 Synthesizing abstract rules from patterns and transformations")
        
        abstract_rule = self._synthesize_rule(consistent_patterns, consistent_transformations)
        reasoning_steps.append(f"📋 Synthesized rule: {abstract_rule['description']}")
        
        # Step 4: Application Phase
        reasoning_steps.append("🎯 Applying discovered rule to test input")
        
        # Generate primary solution
        primary_solution = self._apply_rule(task.test_input, abstract_rule)
        
        # Generate alternative solutions
        for trans in consistent_transformations[:3]:  # Try top 3 transformations
            try:
                alt_grid = trans.rule(task.test_input.grid)
                if not np.array_equal(alt_grid, primary_solution.grid):
                    alternative_solutions.append(ARCGrid(alt_grid))
            except:
                pass  # Skip failed transformations
        
        reasoning_steps.append(f"✨ Generated {1 + len(alternative_solutions)} possible solutions")
        
        # Step 5: Confidence Assessment
        confidence = self._assess_confidence(
            consistent_patterns, consistent_transformations, 
            len(task.train_examples), abstract_rule
        )
        
        reasoning_steps.append(f"📊 Solution confidence: {confidence:.1%}")
        
        return ARCSolution(
            task=task,
            predicted_output=primary_solution,
            confidence=confidence,
            discovered_patterns=discovered_patterns,
            applied_transformations=applied_transformations,
            reasoning_steps=reasoning_steps,
            alternative_solutions=alternative_solutions
        )
    
    def _find_consistent_patterns(self, example_patterns: List[Dict]) -> List[Pattern]:
        """Find patterns that appear consistently across examples"""
        consistent_patterns = []
        
        if not example_patterns:
            return consistent_patterns
        
        # Group patterns by type and description
        pattern_groups = defaultdict(list)
        
        for example in example_patterns:
            for pattern in example['input_patterns'] + example['output_patterns']:
                key = (pattern.pattern_type, pattern.description)
                pattern_groups[key].append(pattern)
        
        # Find patterns that appear in most examples
        min_appearances = max(1, len(example_patterns) // 2)  # At least half
        
        for (pattern_type, description), patterns in pattern_groups.items():
            if len(patterns) >= min_appearances:
                # Create consolidated pattern
                avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
                example_ids = list(set(p.applicable_examples for p in patterns))
                
                consistent_pattern = Pattern(
                    pattern_type=pattern_type,
                    description=description,
                    confidence=avg_confidence,
                    parameters=patterns[0].parameters,  # Use first pattern's parameters
                    applicable_examples=example_ids
                )
                consistent_patterns.append(consistent_pattern)
        
        return consistent_patterns
    
    def _find_consistent_transformations(self, example_transformations: List[List[Transformation]]) -> List[Transformation]:
        """Find transformations that work across multiple examples"""
        consistent_transformations = []
        
        if not example_transformations:
            return consistent_transformations
        
        # Group transformations by type and description
        transformation_groups = defaultdict(list)
        
        for i, transformations in enumerate(example_transformations):
            for trans in transformations:
                key = (trans.transformation_type, trans.description)
                transformation_groups[key].append((trans, i))
        
        # Find transformations that appear in multiple examples
        min_appearances = max(1, len(example_transformations) // 2)
        
        for (trans_type, description), trans_list in transformation_groups.items():
            if len(trans_list) >= min_appearances:
                # Use the transformation with highest confidence
                best_trans = max(trans_list, key=lambda x: x[0].confidence)[0]
                consistent_transformations.append(best_trans)
        
        return consistent_transformations
    
    def _synthesize_rule(self, patterns: List[Pattern], transformations: List[Transformation]) -> Dict[str, Any]:
        """Synthesize abstract rule from patterns and transformations"""
        
        if not patterns and not transformations:
            return {
                'description': 'Identity transformation (output = input)',
                'type': 'identity',
                'confidence': 0.5
            }
        
        # Combine insights from patterns and transformations
        rule_components = []
        
        # Add pattern-based components
        for pattern in patterns:
            if pattern.confidence > 0.7:
                rule_components.append(f"{pattern.pattern_type.value}: {pattern.description}")
        
        # Add transformation-based components
        for trans in transformations:
            if trans.confidence > 0.7:
                rule_components.append(f"Transform: {trans.description}")
        
        if rule_components:
            description = "Apply: " + " + ".join(rule_components[:3])  # Top 3 components
            confidence = sum(p.confidence for p in patterns) + sum(t.confidence for t in transformations)
            confidence = min(0.95, confidence / max(1, len(patterns) + len(transformations)))
        else:
            description = "Pattern-based transformation (details unclear)"
            confidence = 0.3
        
        return {
            'description': description,
            'type': 'composite',
            'patterns': patterns,
            'transformations': transformations,
            'confidence': confidence
        }
    
    def _apply_rule(self, test_input: ARCGrid, rule: Dict[str, Any]) -> ARCGrid:
        """Apply synthesized rule to test input"""
        
        result_grid = test_input.grid.copy()
        
        # Apply transformations in order
        for transformation in rule.get('transformations', []):
            try:
                result_grid = transformation.rule(result_grid)
            except Exception as e:
                # If transformation fails, continue with original
                print(f"Transformation failed: {e}")
                continue
        
        # If no transformations or all failed, try pattern-based prediction
        if np.array_equal(result_grid, test_input.grid):
            result_grid = self._pattern_based_prediction(test_input, rule.get('patterns', []))
        
        return ARCGrid(result_grid)
    
    def _pattern_based_prediction(self, test_input: ARCGrid, patterns: List[Pattern]) -> np.ndarray:
        """Make prediction based on discovered patterns"""
        
        result = test_input.grid.copy()
        
        # Simple pattern-based modifications
        for pattern in patterns:
            if pattern.pattern_type == PatternType.COLOR:
                # Apply color-based rule
                if 'dominant_color' in pattern.parameters:
                    dominant_color = pattern.parameters['dominant_color']
                    # Simple rule: change all non-zero pixels to dominant color
                    result[result != 0] = dominant_color
            
            elif pattern.pattern_type == PatternType.SYMMETRY:
                # Apply symmetry transformation
                symmetry_type = pattern.parameters.get('symmetry_type', '')
                if symmetry_type == 'horizontal':
                    result = np.flipud(result)
                elif symmetry_type == 'vertical':
                    result = np.fliplr(result)
                elif symmetry_type == 'rotation_90':
                    result = np.rot90(result)
        
        return result
    
    def _assess_confidence(self, patterns: List[Pattern], transformations: List[Transformation], 
                          num_examples: int, rule: Dict[str, Any]) -> float:
        """Assess confidence in the solution"""
        
        base_confidence = 0.6
        
        # Boost confidence for more training examples
        example_boost = min(0.2, num_examples * 0.05)
        
        # Boost confidence for high-confidence patterns
        pattern_boost = 0.0
        if patterns:
            avg_pattern_confidence = sum(p.confidence for p in patterns) / len(patterns)
            pattern_boost = (avg_pattern_confidence - 0.5) * 0.2
        
        # Boost confidence for high-confidence transformations
        transform_boost = 0.0
        if transformations:
            avg_transform_confidence = sum(t.confidence for t in transformations) / len(transformations)
            transform_boost = (avg_transform_confidence - 0.5) * 0.3
        
        # Boost confidence for clear, simple rules
        rule_boost = 0.0
        if rule['confidence'] > 0.8:
            rule_boost = 0.1
        
        total_confidence = base_confidence + example_boost + pattern_boost + transform_boost + rule_boost
        
        return max(0.1, min(0.95, total_confidence))
    
    def format_arc_solution(self, solution: ARCSolution) -> str:
        """Format ARC-AGI-2 solution for display"""
        
        formatted = f"🧩 **ARC-AGI-2 Abstract Reasoning Solution**\n\n"
        formatted += f"**Task ID:** {solution.task.task_id}\n"
        formatted += f"**Training Examples:** {len(solution.task.train_examples)}\n"
        formatted += f"**Confidence:** {solution.confidence:.1%}\n"
        formatted += f"**Alternative Solutions:** {len(solution.alternative_solutions)}\n\n"
        
        formatted += f"**Discovered Patterns ({len(solution.discovered_patterns)}):**\n"
        for i, pattern in enumerate(solution.discovered_patterns[:5], 1):
            formatted += f"   {i}. {pattern.pattern_type.value.title()}: {pattern.description} ({pattern.confidence:.1%})\n"
        
        if solution.applied_transformations:
            formatted += f"\n**Applied Transformations ({len(solution.applied_transformations)}):**\n"
            for i, trans in enumerate(solution.applied_transformations[:3], 1):
                formatted += f"   {i}. {trans.transformation_type.value.title()}: {trans.description} ({trans.confidence:.1%})\n"
        
        formatted += f"\n**Abstract Reasoning Steps:**\n"
        for i, step in enumerate(solution.reasoning_steps, 1):
            formatted += f"   {i}. {step}\n"
        
        # Grid dimensions
        formatted += f"\n**Output Grid:** {solution.predicted_output.height}×{solution.predicted_output.width} "
        formatted += f"({len(solution.predicted_output.colors)} colors)\n"
        
        return formatted

# Integration function for consciousness system
def integrate_arc_agi_2(consciousness_system, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Integrate ARC-AGI-2 solving with consciousness system"""
    
    solver = ARCAGI2Solver()
    
    # Parse and solve the task
    task = solver.parse_arc_task(task_data)
    solution = solver.solve_arc_task(task)
    
    # Format for consciousness integration
    arc_result = {
        'task_id': task.task_id,
        'training_examples': len(task.train_examples),
        'patterns_discovered': len(solution.discovered_patterns),
        'transformations_found': len(solution.applied_transformations),
        'confidence': solution.confidence,
        'output_dimensions': (solution.predicted_output.height, solution.predicted_output.width),
        'output_colors': len(solution.predicted_output.colors),
        'reasoning_steps': len(solution.reasoning_steps),
        'alternative_solutions': len(solution.alternative_solutions),
        'formatted_solution': solver.format_arc_solution(solution),
        'abstract_reasoning_quality': 'high' if solution.confidence > 0.8 else 'medium' if solution.confidence > 0.6 else 'low'
    }
    
    # Add to consciousness working memory if available
    if hasattr(consciousness_system, 'working_memory'):
        consciousness_system.working_memory.add_experience({
            'type': 'arc_agi_2_solution',
            'content': arc_result,
            'timestamp': consciousness_system.get_current_time() if hasattr(consciousness_system, 'get_current_time') else 0,
            'significance': solution.confidence
        })
    
    return arc_result