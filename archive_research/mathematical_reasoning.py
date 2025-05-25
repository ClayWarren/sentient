"""
Mathematical Reasoning System for Sentient AI
Specialized reasoning for mathematical problems and symbolic computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import math
import sympy as sp
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json

class MathProblemType(Enum):
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    GEOMETRY = "geometry"
    STATISTICS = "statistics"
    LOGIC = "logic"
    DISCRETE = "discrete"
    LINEAR_ALGEBRA = "linear_algebra"

@dataclass
class MathStep:
    step_number: int
    operation: str
    expression: str
    result: Union[str, float, int]
    explanation: str
    confidence: float
    problem_type: MathProblemType

@dataclass
class MathSolution:
    problem: str
    problem_type: MathProblemType
    steps: List[MathStep]
    final_answer: Union[str, float, int]
    confidence: float
    verification: Optional[Dict[str, Any]] = None

class MathematicalReasoningModule(nn.Module):
    """Neural module for mathematical reasoning"""
    
    def __init__(self, d_model: int = 768, num_problem_types: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_problem_types = num_problem_types
        
        # Problem type classifier
        self.problem_classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_problem_types),
            nn.Softmax(dim=-1)
        )
        
        # Step generation encoder
        self.step_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=d_model*2),
            num_layers=4
        )
        
        # Operation predictor
        self.operation_predictor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 64),  # Common math operations
            nn.Softmax(dim=-1)
        )
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = input_embedding.size(0)
        
        # Classify problem type
        problem_type_probs = self.problem_classifier(input_embedding)
        
        # Generate step encoding
        step_encoding = self.step_encoder(input_embedding.unsqueeze(1)).squeeze(1)
        
        # Predict operations
        operation_probs = self.operation_predictor(step_encoding)
        
        # Predict confidence
        confidence = self.confidence_predictor(step_encoding).squeeze(-1)
        
        return {
            'problem_type_probs': problem_type_probs,
            'step_encoding': step_encoding,
            'operation_probs': operation_probs,
            'confidence': confidence
        }

class SymbolicMathEngine:
    """Symbolic mathematics engine using SymPy"""
    
    def __init__(self):
        self.symbol_cache = {}
        
    def parse_expression(self, expr_str: str) -> sp.Expr:
        """Parse string expression into SymPy expression"""
        try:
            # Clean and prepare expression
            expr_str = self._clean_expression(expr_str)
            
            # Parse with SymPy
            expr = sp.sympify(expr_str)
            return expr
        except Exception as e:
            raise ValueError(f"Could not parse expression '{expr_str}': {e}")
    
    def _clean_expression(self, expr: str) -> str:
        """Clean and standardize mathematical expression"""
        # Replace common variations
        expr = expr.replace('^', '**')  # Power notation
        expr = expr.replace('ln', 'log')  # Natural log
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)  # Implicit multiplication
        expr = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expr)  # Implicit multiplication
        
        return expr
    
    def solve_equation(self, equation: str, variable: str = 'x') -> List[sp.Expr]:
        """Solve an equation for a variable"""
        try:
            # Parse equation
            if '=' in equation:
                left, right = equation.split('=')
                expr = sp.sympify(left) - sp.sympify(right)
            else:
                expr = sp.sympify(equation)
            
            # Define variable
            var = sp.Symbol(variable)
            
            # Solve
            solutions = sp.solve(expr, var)
            return solutions
        except Exception as e:
            raise ValueError(f"Could not solve equation '{equation}': {e}")
    
    def differentiate(self, expr: str, variable: str = 'x') -> sp.Expr:
        """Compute derivative of expression"""
        try:
            expr_obj = self.parse_expression(expr)
            var = sp.Symbol(variable)
            derivative = sp.diff(expr_obj, var)
            return derivative
        except Exception as e:
            raise ValueError(f"Could not differentiate '{expr}': {e}")
    
    def integrate(self, expr: str, variable: str = 'x', limits: Optional[Tuple] = None) -> sp.Expr:
        """Compute integral of expression"""
        try:
            expr_obj = self.parse_expression(expr)
            var = sp.Symbol(variable)
            
            if limits:
                integral = sp.integrate(expr_obj, (var, limits[0], limits[1]))
            else:
                integral = sp.integrate(expr_obj, var)
                
            return integral
        except Exception as e:
            raise ValueError(f"Could not integrate '{expr}': {e}")
    
    def simplify(self, expr: str) -> sp.Expr:
        """Simplify mathematical expression"""
        try:
            expr_obj = self.parse_expression(expr)
            simplified = sp.simplify(expr_obj)
            return simplified
        except Exception as e:
            raise ValueError(f"Could not simplify '{expr}': {e}")
    
    def factor(self, expr: str) -> sp.Expr:
        """Factor polynomial expression"""
        try:
            expr_obj = self.parse_expression(expr)
            factored = sp.factor(expr_obj)
            return factored
        except Exception as e:
            raise ValueError(f"Could not factor '{expr}': {e}")
    
    def expand(self, expr: str) -> sp.Expr:
        """Expand mathematical expression"""
        try:
            expr_obj = self.parse_expression(expr)
            expanded = sp.expand(expr_obj)
            return expanded
        except Exception as e:
            raise ValueError(f"Could not expand '{expr}': {e}")

class MathematicalReasoner:
    """High-level mathematical reasoning system"""
    
    def __init__(self):
        self.symbolic_engine = SymbolicMathEngine()
        self.math_module = MathematicalReasoningModule()
        
        # Problem patterns
        self.problem_patterns = {
            MathProblemType.ARITHMETIC: [
                r'calculate\s+(.+)',
                r'what\s+is\s+(.+)',
                r'compute\s+(.+)',
                r'(\d+\s*[+\-*/]\s*\d+.*)'
            ],
            MathProblemType.ALGEBRA: [
                r'solve\s+(.+?)\s+for\s+(\w+)',
                r'find\s+(\w+)\s+when\s+(.+)',
                r'(.+?)\s*=\s*(.+)'
            ],
            MathProblemType.CALCULUS: [
                r'derivative\s+of\s+(.+)',
                r'differentiate\s+(.+)',
                r'integrate\s+(.+)',
                r'integral\s+of\s+(.+)'
            ],
            MathProblemType.GEOMETRY: [
                r'area\s+of\s+(.+)',
                r'perimeter\s+of\s+(.+)',
                r'volume\s+of\s+(.+)'
            ]
        }
        
        # Operation mappings
        self.operations = {
            'add': lambda a, b: a + b,
            'subtract': lambda a, b: a - b,
            'multiply': lambda a, b: a * b,
            'divide': lambda a, b: a / b if b != 0 else float('inf'),
            'power': lambda a, b: a ** b,
            'sqrt': lambda a: math.sqrt(a) if a >= 0 else complex(0, math.sqrt(-a)),
            'log': lambda a: math.log(a) if a > 0 else float('-inf'),
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan
        }
    
    def identify_problem_type(self, problem: str) -> MathProblemType:
        """Identify the type of mathematical problem"""
        problem_lower = problem.lower().strip()
        
        # Check patterns for each problem type
        for problem_type, patterns in self.problem_patterns.items():
            for pattern in patterns:
                if re.search(pattern, problem_lower):
                    return problem_type
        
        # Default classification based on keywords
        if any(word in problem_lower for word in ['solve', 'equation', 'variable', 'x', 'y']):
            return MathProblemType.ALGEBRA
        elif any(word in problem_lower for word in ['derivative', 'integral', 'limit', 'differentiate']):
            return MathProblemType.CALCULUS
        elif any(word in problem_lower for word in ['area', 'volume', 'perimeter', 'angle']):
            return MathProblemType.GEOMETRY
        elif any(word in problem_lower for word in ['mean', 'median', 'probability', 'distribution']):
            return MathProblemType.STATISTICS
        else:
            return MathProblemType.ARITHMETIC
    
    def solve_problem(self, problem: str) -> MathSolution:
        """Solve a mathematical problem with step-by-step reasoning"""
        problem_type = self.identify_problem_type(problem)
        
        if problem_type == MathProblemType.ARITHMETIC:
            return self._solve_arithmetic(problem)
        elif problem_type == MathProblemType.ALGEBRA:
            return self._solve_algebra(problem)
        elif problem_type == MathProblemType.CALCULUS:
            return self._solve_calculus(problem)
        elif problem_type == MathProblemType.GEOMETRY:
            return self._solve_geometry(problem)
        else:
            return self._solve_general(problem, problem_type)
    
    def _solve_arithmetic(self, problem: str) -> MathSolution:
        """Solve arithmetic problems"""
        steps = []
        
        # Extract numerical expression
        expr_match = re.search(r'[\d+\-*/().\s]+', problem)
        if not expr_match:
            return MathSolution(problem, MathProblemType.ARITHMETIC, [], "Unable to parse", 0.0)
        
        expression = expr_match.group().strip()
        
        try:
            # Step 1: Parse expression
            steps.append(MathStep(
                step_number=1,
                operation="parse",
                expression=expression,
                result=expression,
                explanation=f"Identified arithmetic expression: {expression}",
                confidence=0.9,
                problem_type=MathProblemType.ARITHMETIC
            ))
            
            # Step 2: Evaluate
            result = eval(expression)
            steps.append(MathStep(
                step_number=2,
                operation="evaluate",
                expression=expression,
                result=result,
                explanation=f"Computed the result: {result}",
                confidence=0.95,
                problem_type=MathProblemType.ARITHMETIC
            ))
            
            return MathSolution(
                problem=problem,
                problem_type=MathProblemType.ARITHMETIC,
                steps=steps,
                final_answer=result,
                confidence=0.92
            )
            
        except Exception as e:
            steps.append(MathStep(
                step_number=len(steps) + 1,
                operation="error",
                expression=expression,
                result=f"Error: {e}",
                explanation=f"Failed to evaluate expression: {e}",
                confidence=0.1,
                problem_type=MathProblemType.ARITHMETIC
            ))
            
            return MathSolution(
                problem=problem,
                problem_type=MathProblemType.ARITHMETIC,
                steps=steps,
                final_answer=f"Error: {e}",
                confidence=0.1
            )
    
    def _solve_algebra(self, problem: str) -> MathSolution:
        """Solve algebraic equations"""
        steps = []
        
        try:
            # Step 1: Identify equation and variable
            if ' for ' in problem.lower():
                equation_part, variable_part = problem.lower().split(' for ')
                variable = variable_part.strip().split()[0]
            else:
                variable = 'x'  # Default variable
                equation_part = problem
            
            # Extract equation
            equation_match = re.search(r'[^a-zA-Z]*([^.!?]*)', equation_part)
            if equation_match:
                equation = equation_match.group(1).strip()
            else:
                equation = equation_part.strip()
            
            steps.append(MathStep(
                step_number=1,
                operation="identify",
                expression=equation,
                result=f"Variable: {variable}",
                explanation=f"Identified equation '{equation}' to solve for '{variable}'",
                confidence=0.8,
                problem_type=MathProblemType.ALGEBRA
            ))
            
            # Step 2: Solve using symbolic engine
            solutions = self.symbolic_engine.solve_equation(equation, variable)
            
            steps.append(MathStep(
                step_number=2,
                operation="solve",
                expression=equation,
                result=str(solutions),
                explanation=f"Applied algebraic methods to solve for {variable}",
                confidence=0.9,
                problem_type=MathProblemType.ALGEBRA
            ))
            
            # Step 3: Format result
            if solutions:
                if len(solutions) == 1:
                    final_answer = f"{variable} = {solutions[0]}"
                else:
                    final_answer = f"{variable} = {solutions}"
            else:
                final_answer = "No solution found"
            
            steps.append(MathStep(
                step_number=3,
                operation="format",
                expression=str(solutions),
                result=final_answer,
                explanation=f"Formatted final answer: {final_answer}",
                confidence=0.9,
                problem_type=MathProblemType.ALGEBRA
            ))
            
            return MathSolution(
                problem=problem,
                problem_type=MathProblemType.ALGEBRA,
                steps=steps,
                final_answer=final_answer,
                confidence=0.87
            )
            
        except Exception as e:
            steps.append(MathStep(
                step_number=len(steps) + 1,
                operation="error",
                expression="",
                result=f"Error: {e}",
                explanation=f"Failed to solve algebraic problem: {e}",
                confidence=0.1,
                problem_type=MathProblemType.ALGEBRA
            ))
            
            return MathSolution(
                problem=problem,
                problem_type=MathProblemType.ALGEBRA,
                steps=steps,
                final_answer=f"Error: {e}",
                confidence=0.1
            )
    
    def _solve_calculus(self, problem: str) -> MathSolution:
        """Solve calculus problems"""
        steps = []
        
        try:
            problem_lower = problem.lower()
            
            # Determine operation type
            if 'derivative' in problem_lower or 'differentiate' in problem_lower:
                operation_type = 'derivative'
            elif 'integral' in problem_lower or 'integrate' in problem_lower:
                operation_type = 'integral'
            else:
                operation_type = 'unknown'
            
            # Extract expression and variable
            expr_pattern = r'(?:of\s+|integrate\s+|differentiate\s+)([^,\s]+(?:\s*[+\-*/]\s*[^,\s]+)*)'
            expr_match = re.search(expr_pattern, problem_lower)
            
            if expr_match:
                expression = expr_match.group(1).strip()
            else:
                # Fallback pattern
                expr_match = re.search(r'([x\d+\-*/^().\s]+)', problem)
                expression = expr_match.group(1).strip() if expr_match else 'x'
            
            # Extract variable
            var_match = re.search(r'with respect to (\w+)|d/d(\w+)', problem_lower)
            if var_match:
                variable = var_match.group(1) or var_match.group(2)
            else:
                variable = 'x'  # Default
            
            steps.append(MathStep(
                step_number=1,
                operation="identify",
                expression=expression,
                result=f"Operation: {operation_type}, Variable: {variable}",
                explanation=f"Identified {operation_type} of '{expression}' with respect to '{variable}'",
                confidence=0.8,
                problem_type=MathProblemType.CALCULUS
            ))
            
            # Perform calculus operation
            if operation_type == 'derivative':
                result = self.symbolic_engine.differentiate(expression, variable)
                explanation = f"Applied differentiation rules to find the derivative"
            elif operation_type == 'integral':
                result = self.symbolic_engine.integrate(expression, variable)
                explanation = f"Applied integration techniques to find the antiderivative"
            else:
                result = f"Unknown operation: {operation_type}"
                explanation = f"Could not determine calculus operation"
            
            steps.append(MathStep(
                step_number=2,
                operation=operation_type,
                expression=expression,
                result=str(result),
                explanation=explanation,
                confidence=0.9,
                problem_type=MathProblemType.CALCULUS
            ))
            
            return MathSolution(
                problem=problem,
                problem_type=MathProblemType.CALCULUS,
                steps=steps,
                final_answer=str(result),
                confidence=0.85
            )
            
        except Exception as e:
            steps.append(MathStep(
                step_number=len(steps) + 1,
                operation="error",
                expression="",
                result=f"Error: {e}",
                explanation=f"Failed to solve calculus problem: {e}",
                confidence=0.1,
                problem_type=MathProblemType.CALCULUS
            ))
            
            return MathSolution(
                problem=problem,
                problem_type=MathProblemType.CALCULUS,
                steps=steps,
                final_answer=f"Error: {e}",
                confidence=0.1
            )
    
    def _solve_geometry(self, problem: str) -> MathSolution:
        """Solve geometry problems"""
        steps = []
        problem_lower = problem.lower()
        
        try:
            # Identify shape and operation
            if 'circle' in problem_lower:
                shape = 'circle'
            elif 'rectangle' in problem_lower or 'square' in problem_lower:
                shape = 'rectangle'
            elif 'triangle' in problem_lower:
                shape = 'triangle'
            else:
                shape = 'unknown'
            
            if 'area' in problem_lower:
                operation = 'area'
            elif 'perimeter' in problem_lower:
                operation = 'perimeter'
            elif 'volume' in problem_lower:
                operation = 'volume'
            else:
                operation = 'unknown'
            
            steps.append(MathStep(
                step_number=1,
                operation="identify",
                expression=f"{shape} {operation}",
                result=f"Shape: {shape}, Operation: {operation}",
                explanation=f"Identified geometric problem: find {operation} of {shape}",
                confidence=0.8,
                problem_type=MathProblemType.GEOMETRY
            ))
            
            # Extract numerical values
            numbers = re.findall(r'\d+(?:\.\d+)?', problem)
            numbers = [float(n) for n in numbers]
            
            # Apply geometric formulas
            if shape == 'circle' and operation == 'area' and len(numbers) >= 1:
                radius = numbers[0]
                result = math.pi * radius ** 2
                formula = f"π × r² = π × {radius}² = {result:.2f}"
            elif shape == 'circle' and operation == 'perimeter' and len(numbers) >= 1:
                radius = numbers[0]
                result = 2 * math.pi * radius
                formula = f"2π × r = 2π × {radius} = {result:.2f}"
            elif shape == 'rectangle' and operation == 'area' and len(numbers) >= 2:
                length, width = numbers[0], numbers[1]
                result = length * width
                formula = f"length × width = {length} × {width} = {result}"
            elif shape == 'rectangle' and operation == 'perimeter' and len(numbers) >= 2:
                length, width = numbers[0], numbers[1]
                result = 2 * (length + width)
                formula = f"2(length + width) = 2({length} + {width}) = {result}"
            else:
                result = "Insufficient information or unsupported geometry problem"
                formula = "Could not apply geometric formula"
            
            steps.append(MathStep(
                step_number=2,
                operation="calculate",
                expression=formula,
                result=result,
                explanation=f"Applied geometric formula: {formula}",
                confidence=0.9 if isinstance(result, (int, float)) else 0.3,
                problem_type=MathProblemType.GEOMETRY
            ))
            
            return MathSolution(
                problem=problem,
                problem_type=MathProblemType.GEOMETRY,
                steps=steps,
                final_answer=result,
                confidence=0.85 if isinstance(result, (int, float)) else 0.3
            )
            
        except Exception as e:
            steps.append(MathStep(
                step_number=len(steps) + 1,
                operation="error",
                expression="",
                result=f"Error: {e}",
                explanation=f"Failed to solve geometry problem: {e}",
                confidence=0.1,
                problem_type=MathProblemType.GEOMETRY
            ))
            
            return MathSolution(
                problem=problem,
                problem_type=MathProblemType.GEOMETRY,
                steps=steps,
                final_answer=f"Error: {e}",
                confidence=0.1
            )
    
    def _solve_general(self, problem: str, problem_type: MathProblemType) -> MathSolution:
        """General mathematical problem solver"""
        steps = []
        
        steps.append(MathStep(
            step_number=1,
            operation="analyze",
            expression=problem,
            result=f"Problem type: {problem_type.value}",
            explanation=f"Analyzed problem as {problem_type.value} type",
            confidence=0.7,
            problem_type=problem_type
        ))
        
        # Try to extract and evaluate any expressions
        try:
            expr_matches = re.findall(r'[\d+\-*/().\s]+', problem)
            if expr_matches:
                for i, expr in enumerate(expr_matches):
                    try:
                        result = eval(expr.strip())
                        steps.append(MathStep(
                            step_number=len(steps) + 1,
                            operation="evaluate",
                            expression=expr.strip(),
                            result=result,
                            explanation=f"Evaluated expression: {expr.strip()} = {result}",
                            confidence=0.8,
                            problem_type=problem_type
                        ))
                    except:
                        continue
        except:
            pass
        
        if len(steps) == 1:
            final_answer = f"Unable to solve {problem_type.value} problem automatically"
            confidence = 0.3
        else:
            final_answer = steps[-1].result
            confidence = sum(step.confidence for step in steps) / len(steps)
        
        return MathSolution(
            problem=problem,
            problem_type=problem_type,
            steps=steps,
            final_answer=final_answer,
            confidence=confidence
        )
    
    def format_solution(self, solution: MathSolution) -> str:
        """Format mathematical solution for display"""
        formatted = f"📐 **Mathematical Reasoning Solution**\n\n"
        formatted += f"**Problem:** {solution.problem}\n"
        formatted += f"**Type:** {solution.problem_type.value.title()}\n"
        formatted += f"**Confidence:** {solution.confidence:.1%}\n\n"
        
        formatted += "**Step-by-Step Solution:**\n\n"
        
        for step in solution.steps:
            confidence_bar = "▓" * int(step.confidence * 10) + "░" * (10 - int(step.confidence * 10))
            formatted += f"**Step {step.step_number}:** {step.operation.title()}\n"
            formatted += f"   Expression: `{step.expression}`\n"
            formatted += f"   Result: `{step.result}`\n"
            formatted += f"   Explanation: {step.explanation}\n"
            formatted += f"   Confidence: [{confidence_bar}] {step.confidence:.1%}\n\n"
        
        formatted += f"🎯 **Final Answer:** {solution.final_answer}\n"
        
        return formatted

# Integration function for consciousness system
def integrate_mathematical_reasoning(consciousness_system, problem: str) -> Dict[str, Any]:
    """Integrate mathematical reasoning with consciousness system"""
    
    reasoner = MathematicalReasoner()
    
    # Solve the mathematical problem
    solution = reasoner.solve_problem(problem)
    
    # Format for consciousness integration
    math_result = {
        'problem': problem,
        'problem_type': solution.problem_type.value,
        'solution': solution,
        'formatted_solution': reasoner.format_solution(solution),
        'final_answer': solution.final_answer,
        'confidence': solution.confidence,
        'num_steps': len(solution.steps)
    }
    
    # Add to consciousness working memory if available
    if hasattr(consciousness_system, 'working_memory'):
        consciousness_system.working_memory.add_experience({
            'type': 'mathematical_reasoning',
            'content': math_result,
            'timestamp': consciousness_system.get_current_time() if hasattr(consciousness_system, 'get_current_time') else 0,
            'significance': solution.confidence
        })
    
    return math_result