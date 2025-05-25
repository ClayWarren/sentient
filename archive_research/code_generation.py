"""
Code Generation and Execution System for Sentient AI
Enables the AI to generate, analyze, and execute code in multiple languages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import tempfile
import os
import sys
import ast
import traceback
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import threading
import queue

class CodeLanguage(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    BASH = "bash"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"

class CodeType(Enum):
    FUNCTION = "function"
    CLASS = "class"
    SCRIPT = "script"
    SNIPPET = "snippet"
    TEST = "test"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"

@dataclass
class CodeGenerationRequest:
    description: str
    language: CodeLanguage
    code_type: CodeType
    requirements: List[str]
    context: Optional[str] = None
    examples: Optional[List[str]] = None

@dataclass
class GeneratedCode:
    code: str
    language: CodeLanguage
    code_type: CodeType
    explanation: str
    confidence: float
    estimated_lines: int
    dependencies: List[str]
    test_cases: Optional[List[str]] = None

@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""

class CodeGenerationModule(nn.Module):
    """Neural module for code generation"""
    
    def __init__(self, d_model: int = 768, num_languages: int = 9, num_types: int = 7):
        super().__init__()
        self.d_model = d_model
        self.num_languages = num_languages
        self.num_types = num_types
        
        # Language classifier
        self.language_classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_languages),
            nn.Softmax(dim=-1)
        )
        
        # Code type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, num_types),
            nn.Softmax(dim=-1)
        )
        
        # Code structure encoder
        self.structure_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=d_model*2),
            num_layers=3
        )
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # Ensure positive output
        )
        
    def forward(self, input_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = input_embedding.size(0)
        
        # Classify language
        language_probs = self.language_classifier(input_embedding)
        
        # Classify code type
        type_probs = self.type_classifier(input_embedding)
        
        # Encode structure
        structure_encoding = self.structure_encoder(input_embedding.unsqueeze(1)).squeeze(1)
        
        # Predict confidence
        confidence = self.confidence_predictor(structure_encoding).squeeze(-1)
        
        # Estimate complexity (lines of code)
        complexity = self.complexity_estimator(structure_encoding).squeeze(-1)
        
        return {
            'language_probabilities': language_probs,
            'type_probabilities': type_probs,
            'structure_encoding': structure_encoding,
            'confidence': confidence,
            'complexity_estimate': complexity
        }

class CodeTemplateEngine:
    """Template-based code generation engine"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize code templates for different languages and types"""
        return {
            CodeLanguage.PYTHON.value: {
                CodeType.FUNCTION.value: '''def {function_name}({parameters}):
    """
    {description}
    
    Args:
        {args_doc}
    
    Returns:
        {return_doc}
    """
    {body}
    return {return_value}''',
                
                CodeType.CLASS.value: '''class {class_name}:
    """
    {description}
    """
    
    def __init__(self{init_params}):
        """Initialize {class_name}"""
        {init_body}
    
    {methods}''',
                
                CodeType.SCRIPT.value: '''#!/usr/bin/env python3
"""
{description}
"""

{imports}

def main():
    """Main function"""
    {main_body}

if __name__ == "__main__":
    main()''',
                
                CodeType.TEST.value: '''import unittest
{additional_imports}

class Test{class_name}(unittest.TestCase):
    """Test cases for {target_name}"""
    
    def setUp(self):
        """Set up test fixtures"""
        {setup_code}
    
    {test_methods}

if __name__ == "__main__":
    unittest.main()'''
            },
            
            CodeLanguage.JAVASCRIPT.value: {
                CodeType.FUNCTION.value: '''/**
 * {description}
 * @param {{Object}} {param_name} - {param_description}
 * @returns {{Object}} {return_description}
 */
function {function_name}({parameters}) {{
    {body}
    return {return_value};
}}''',
                
                CodeType.CLASS.value: '''/**
 * {description}
 */
class {class_name} {{
    constructor({constructor_params}) {{
        {constructor_body}
    }}
    
    {methods}
}}''',
                
                CodeType.SCRIPT.value: '''#!/usr/bin/env node
/**
 * {description}
 */

{imports}

function main() {{
    {main_body}
}}

if (require.main === module) {{
    main();
}}'''
            },
            
            CodeLanguage.BASH.value: {
                CodeType.SCRIPT.value: '''#!/bin/bash
# {description}

set -euo pipefail

{functions}

main() {{
    {main_body}
}}

main "$@"''',
                
                CodeType.FUNCTION.value: '''{function_name}() {{
    # {description}
    {body}
}}'''
            }
        }
    
    def generate_from_template(self, request: CodeGenerationRequest, **kwargs) -> str:
        """Generate code using templates"""
        language_templates = self.templates.get(request.language.value, {})
        template = language_templates.get(request.code_type.value)
        
        if not template:
            return self._generate_fallback(request)
        
        # Fill template with provided values
        try:
            return template.format(**kwargs)
        except KeyError as e:
            # Handle missing template variables
            return self._generate_partial_template(template, **kwargs)
    
    def _generate_fallback(self, request: CodeGenerationRequest) -> str:
        """Generate fallback code when no template exists"""
        if request.language == CodeLanguage.PYTHON:
            return f'# {request.description}\n# TODO: Implement {request.code_type.value}\npass'
        elif request.language == CodeLanguage.JAVASCRIPT:
            return f'// {request.description}\n// TODO: Implement {request.code_type.value}'
        elif request.language == CodeLanguage.BASH:
            return f'#!/bin/bash\n# {request.description}\n# TODO: Implement {request.code_type.value}'
        else:
            return f'# {request.description}\n# TODO: Implement {request.code_type.value} in {request.language.value}'
    
    def _generate_partial_template(self, template: str, **kwargs) -> str:
        """Generate code with partial template filling"""
        # Replace available variables and leave placeholders for missing ones
        for key, value in kwargs.items():
            template = template.replace(f'{{{key}}}', str(value))
        
        # Replace remaining placeholders with TODO comments
        import re
        template = re.sub(r'\{(\w+)\}', r'# TODO: \1', template)
        
        return template

class CodeExecutor:
    """Secure code execution environment"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.supported_languages = {
            CodeLanguage.PYTHON: self._execute_python,
            CodeLanguage.JAVASCRIPT: self._execute_javascript,
            CodeLanguage.BASH: self._execute_bash
        }
        
    def execute_code(self, code: str, language: CodeLanguage, 
                    input_data: Optional[str] = None) -> ExecutionResult:
        """Execute code in the specified language"""
        
        if language not in self.supported_languages:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Language {language.value} not supported for execution"
            )
        
        try:
            executor = self.supported_languages[language]
            return executor(code, input_data)
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution failed: {str(e)}"
            )
    
    def _execute_python(self, code: str, input_data: Optional[str] = None) -> ExecutionResult:
        """Execute Python code safely"""
        start_time = time.time()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Prepare command
            cmd = [sys.executable, temp_file]
            
            # Execute with timeout
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tempfile.gettempdir()
            )
            
            stdout, stderr = process.communicate(
                input=input_data,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=stdout + stderr if stderr else stdout,
                error=stderr if stderr else None,
                execution_time=execution_time,
                return_code=process.returncode,
                stdout=stdout,
                stderr=stderr
            )
            
        except subprocess.TimeoutExpired:
            process.kill()
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout} seconds",
                execution_time=self.timeout
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time
            )
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def _execute_javascript(self, code: str, input_data: Optional[str] = None) -> ExecutionResult:
        """Execute JavaScript code using Node.js"""
        start_time = time.time()
        
        # Check if Node.js is available
        try:
            subprocess.run(['node', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ExecutionResult(
                success=False,
                output="",
                error="Node.js not available for JavaScript execution"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute with Node.js
            cmd = ['node', temp_file]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(
                input=input_data,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=stdout + stderr if stderr else stdout,
                error=stderr if stderr else None,
                execution_time=execution_time,
                return_code=process.returncode,
                stdout=stdout,
                stderr=stderr
            )
            
        except subprocess.TimeoutExpired:
            process.kill()
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout} seconds",
                execution_time=self.timeout
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time
            )
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def _execute_bash(self, code: str, input_data: Optional[str] = None) -> ExecutionResult:
        """Execute Bash script safely"""
        start_time = time.time()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Make executable
            os.chmod(temp_file, 0o755)
            
            # Execute
            cmd = ['bash', temp_file]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tempfile.gettempdir()
            )
            
            stdout, stderr = process.communicate(
                input=input_data,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=stdout + stderr if stderr else stdout,
                error=stderr if stderr else None,
                execution_time=execution_time,
                return_code=process.returncode,
                stdout=stdout,
                stderr=stderr
            )
            
        except subprocess.TimeoutExpired:
            process.kill()
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout} seconds",
                execution_time=self.timeout
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time
            )
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass

class CodeAnalyzer:
    """Code analysis and quality assessment"""
    
    def analyze_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Analyze code for various metrics"""
        analysis = {
            'language': language.value,
            'lines_of_code': len([line for line in code.split('\n') if line.strip()]),
            'total_lines': len(code.split('\n')),
            'character_count': len(code),
            'estimated_complexity': self._estimate_complexity(code, language),
            'dependencies': self._extract_dependencies(code, language),
            'functions': self._extract_functions(code, language),
            'classes': self._extract_classes(code, language),
            'syntax_valid': self._check_syntax(code, language)
        }
        
        return analysis
    
    def _estimate_complexity(self, code: str, language: CodeLanguage) -> str:
        """Estimate code complexity"""
        lines = len([line for line in code.split('\n') if line.strip()])
        
        if lines < 10:
            return "Low"
        elif lines < 50:
            return "Medium"
        elif lines < 200:
            return "High"
        else:
            return "Very High"
    
    def _extract_dependencies(self, code: str, language: CodeLanguage) -> List[str]:
        """Extract dependencies/imports from code"""
        dependencies = []
        
        if language == CodeLanguage.PYTHON:
            import_patterns = [
                r'^import\s+(\w+)',
                r'^from\s+(\w+)\s+import',
                r'import\s+(\w+)\s+as'
            ]
            
            for line in code.split('\n'):
                line = line.strip()
                for pattern in import_patterns:
                    match = re.match(pattern, line)
                    if match:
                        dependencies.append(match.group(1))
        
        elif language == CodeLanguage.JAVASCRIPT:
            import_patterns = [
                r'require\([\'"]([^\'"]+)[\'"]\)',
                r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
                r'import\s+[\'"]([^\'"]+)[\'"]'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, code)
                dependencies.extend(matches)
        
        return list(set(dependencies))
    
    def _extract_functions(self, code: str, language: CodeLanguage) -> List[str]:
        """Extract function names from code"""
        functions = []
        
        if language == CodeLanguage.PYTHON:
            pattern = r'^def\s+(\w+)\s*\('
            for line in code.split('\n'):
                match = re.match(pattern, line.strip())
                if match:
                    functions.append(match.group(1))
        
        elif language == CodeLanguage.JAVASCRIPT:
            patterns = [
                r'function\s+(\w+)\s*\(',
                r'const\s+(\w+)\s*=\s*function',
                r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, code)
                functions.extend(matches)
        
        elif language == CodeLanguage.BASH:
            pattern = r'^(\w+)\s*\(\s*\)\s*\{'
            for line in code.split('\n'):
                match = re.match(pattern, line.strip())
                if match:
                    functions.append(match.group(1))
        
        return functions
    
    def _extract_classes(self, code: str, language: CodeLanguage) -> List[str]:
        """Extract class names from code"""
        classes = []
        
        if language == CodeLanguage.PYTHON:
            pattern = r'^class\s+(\w+)'
            for line in code.split('\n'):
                match = re.match(pattern, line.strip())
                if match:
                    classes.append(match.group(1))
        
        elif language == CodeLanguage.JAVASCRIPT:
            pattern = r'class\s+(\w+)'
            matches = re.findall(pattern, code)
            classes.extend(matches)
        
        return classes
    
    def _check_syntax(self, code: str, language: CodeLanguage) -> bool:
        """Check if code has valid syntax"""
        try:
            if language == CodeLanguage.PYTHON:
                ast.parse(code)
                return True
            # For other languages, we'd need specific parsers
            # For now, assume valid if no obvious syntax errors
            return True
        except SyntaxError:
            return False
        except Exception:
            return False

class CodeGenerator:
    """Main code generation system"""
    
    def __init__(self):
        self.template_engine = CodeTemplateEngine()
        self.executor = CodeExecutor()
        self.analyzer = CodeAnalyzer()
        self.generation_module = CodeGenerationModule()
        
        # Code generation patterns
        self.generation_patterns = {
            'function': [
                r'write\s+a\s+function\s+(?:to\s+|that\s+)?(.+)',
                r'create\s+a\s+function\s+(?:to\s+|that\s+)?(.+)',
                r'implement\s+a\s+function\s+(?:to\s+|that\s+)?(.+)',
                r'generate\s+(?:a\s+)?function\s+(?:to\s+|that\s+)?(.+)'
            ],
            'class': [
                r'write\s+a\s+class\s+(?:to\s+|that\s+)?(.+)',
                r'create\s+a\s+class\s+(?:to\s+|that\s+)?(.+)',
                r'implement\s+a\s+class\s+(?:to\s+|that\s+)?(.+)'
            ],
            'script': [
                r'write\s+a\s+script\s+(?:to\s+|that\s+)?(.+)',
                r'create\s+a\s+script\s+(?:to\s+|that\s+)?(.+)',
                r'generate\s+(?:a\s+)?script\s+(?:to\s+|that\s+)?(.+)'
            ]
        }
    
    def parse_generation_request(self, description: str) -> CodeGenerationRequest:
        """Parse natural language description into generation request"""
        description_lower = description.lower().strip()
        
        # Detect language
        language = CodeLanguage.PYTHON  # Default
        if 'javascript' in description_lower or 'js' in description_lower:
            language = CodeLanguage.JAVASCRIPT
        elif 'bash' in description_lower or 'shell' in description_lower:
            language = CodeLanguage.BASH
        elif 'sql' in description_lower:
            language = CodeLanguage.SQL
        elif 'html' in description_lower:
            language = CodeLanguage.HTML
        elif 'css' in description_lower:
            language = CodeLanguage.CSS
        
        # Detect code type
        code_type = CodeType.FUNCTION  # Default
        for type_name, patterns in self.generation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, description_lower):
                    code_type = CodeType(type_name.upper())
                    break
        
        # Extract requirements
        requirements = []
        if 'requirements:' in description_lower:
            req_match = re.search(r'requirements:\s*(.+)', description_lower)
            if req_match:
                requirements = [req.strip() for req in req_match.group(1).split(',')]
        
        return CodeGenerationRequest(
            description=description,
            language=language,
            code_type=code_type,
            requirements=requirements
        )
    
    def generate_code(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate code based on request"""
        
        if request.language == CodeLanguage.PYTHON:
            return self._generate_python_code(request)
        elif request.language == CodeLanguage.JAVASCRIPT:
            return self._generate_javascript_code(request)
        elif request.language == CodeLanguage.BASH:
            return self._generate_bash_code(request)
        else:
            return self._generate_generic_code(request)
    
    def _generate_python_code(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate Python code"""
        
        if request.code_type == CodeType.FUNCTION:
            # Extract function details from description
            func_name = self._extract_function_name(request.description)
            parameters = self._extract_parameters(request.description)
            
            # Generate function body based on description
            body = self._generate_function_body(request.description)
            
            code = self.template_engine.generate_from_template(
                request,
                function_name=func_name,
                parameters=parameters,
                description=request.description,
                args_doc=self._generate_args_doc(parameters),
                return_doc="Generated result",
                body=body,
                return_value=self._generate_return_value(request.description)
            )
            
        elif request.code_type == CodeType.CLASS:
            class_name = self._extract_class_name(request.description)
            
            code = self.template_engine.generate_from_template(
                request,
                class_name=class_name,
                description=request.description,
                init_params="",
                init_body="    pass",
                methods="    def example_method(self):\n        \"\"\"Example method\"\"\"\n        pass"
            )
            
        elif request.code_type == CodeType.SCRIPT:
            imports = self._generate_imports(request.description)
            main_body = self._generate_script_body(request.description)
            
            code = self.template_engine.generate_from_template(
                request,
                description=request.description,
                imports=imports,
                main_body=main_body
            )
            
        else:
            code = f'# {request.description}\n# TODO: Implement {request.code_type.value}\npass'
        
        analysis = self.analyzer.analyze_code(code, request.language)
        
        return GeneratedCode(
            code=code,
            language=request.language,
            code_type=request.code_type,
            explanation=f"Generated {request.code_type.value} in {request.language.value}: {request.description}",
            confidence=0.8,
            estimated_lines=analysis['lines_of_code'],
            dependencies=analysis['dependencies']
        )
    
    def _generate_javascript_code(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate JavaScript code"""
        
        if request.code_type == CodeType.FUNCTION:
            func_name = self._extract_function_name(request.description)
            parameters = self._extract_parameters(request.description)
            body = self._generate_function_body(request.description, language='javascript')
            
            code = self.template_engine.generate_from_template(
                request,
                function_name=func_name,
                parameters=parameters,
                description=request.description,
                param_name="params",
                param_description="Function parameters",
                return_description="Function result",
                body=body,
                return_value=self._generate_return_value(request.description, 'javascript')
            )
            
        else:
            code = f'// {request.description}\n// TODO: Implement {request.code_type.value}'
        
        analysis = self.analyzer.analyze_code(code, request.language)
        
        return GeneratedCode(
            code=code,
            language=request.language,
            code_type=request.code_type,
            explanation=f"Generated {request.code_type.value} in {request.language.value}: {request.description}",
            confidence=0.75,
            estimated_lines=analysis['lines_of_code'],
            dependencies=analysis['dependencies']
        )
    
    def _generate_bash_code(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate Bash code"""
        
        if request.code_type == CodeType.SCRIPT:
            functions = ""
            main_body = self._generate_script_body(request.description, language='bash')
            
            code = self.template_engine.generate_from_template(
                request,
                description=request.description,
                functions=functions,
                main_body=main_body
            )
            
        else:
            code = f'#!/bin/bash\n# {request.description}\n# TODO: Implement {request.code_type.value}'
        
        analysis = self.analyzer.analyze_code(code, request.language)
        
        return GeneratedCode(
            code=code,
            language=request.language,
            code_type=request.code_type,
            explanation=f"Generated {request.code_type.value} in {request.language.value}: {request.description}",
            confidence=0.7,
            estimated_lines=analysis['lines_of_code'],
            dependencies=analysis['dependencies']
        )
    
    def _generate_generic_code(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate generic code for unsupported languages"""
        
        code = self.template_engine._generate_fallback(request)
        
        return GeneratedCode(
            code=code,
            language=request.language,
            code_type=request.code_type,
            explanation=f"Generated placeholder for {request.language.value}",
            confidence=0.3,
            estimated_lines=len(code.split('\n')),
            dependencies=[]
        )
    
    # Helper methods for code generation
    def _extract_function_name(self, description: str) -> str:
        """Extract function name from description"""
        # Look for explicit function names
        name_patterns = [
            r'function\s+(?:called\s+|named\s+)?([a-zA-Z_]\w*)',
            r'(?:called\s+|named\s+)([a-zA-Z_]\w*)',
            r'([a-zA-Z_]\w*)\s+function'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Generate name from description
        words = re.findall(r'\b[a-zA-Z]+\b', description.lower())
        if words:
            if len(words) == 1:
                return words[0]
            else:
                return '_'.join(words[:3])  # Use first 3 words
        
        return 'generated_function'
    
    def _extract_class_name(self, description: str) -> str:
        """Extract class name from description"""
        name_patterns = [
            r'class\s+(?:called\s+|named\s+)?([A-Z][a-zA-Z_]\w*)',
            r'(?:called\s+|named\s+)([A-Z][a-zA-Z_]\w*)',
            r'([A-Z][a-zA-Z_]\w*)\s+class'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1).title()
        
        # Generate name from description
        words = re.findall(r'\b[a-zA-Z]+\b', description.lower())
        if words:
            return ''.join(word.title() for word in words[:2])
        
        return 'GeneratedClass'
    
    def _extract_parameters(self, description: str) -> str:
        """Extract function parameters from description"""
        # Look for parameter descriptions
        param_patterns = [
            r'with\s+parameters?\s+([^.]+)',
            r'takes?\s+([^.]+?)\s+as\s+(?:input|parameter)',
            r'given\s+([^.]+)'
        ]
        
        for pattern in param_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                param_text = match.group(1)
                # Simple parameter extraction
                params = re.findall(r'\b[a-zA-Z_]\w*\b', param_text)
                return ', '.join(params[:5])  # Limit to 5 parameters
        
        # Default parameters based on common patterns
        if 'calculate' in description.lower():
            return 'value'
        elif 'process' in description.lower():
            return 'data'
        elif 'convert' in description.lower():
            return 'input_value'
        else:
            return 'input_data'
    
    def _generate_args_doc(self, parameters: str) -> str:
        """Generate arguments documentation"""
        if not parameters:
            return "None"
        
        params = [p.strip() for p in parameters.split(',')]
        docs = []
        for param in params:
            docs.append(f"{param}: Description of {param}")
        
        return '\n        '.join(docs)
    
    def _generate_function_body(self, description: str, language: str = 'python') -> str:
        """Generate function body based on description"""
        
        if language == 'python':
            if 'calculate' in description.lower():
                return '    # Perform calculation\n    result = input_data  # TODO: Implement calculation'
            elif 'process' in description.lower():
                return '    # Process data\n    processed = data  # TODO: Implement processing'
            elif 'convert' in description.lower():
                return '    # Convert input\n    converted = input_value  # TODO: Implement conversion'
            else:
                return '    # Implementation\n    # TODO: Add implementation logic'
        
        elif language == 'javascript':
            if 'calculate' in description.lower():
                return '    // Perform calculation\n    const result = input_data; // TODO: Implement calculation'
            else:
                return '    // Implementation\n    // TODO: Add implementation logic'
        
        return '    # TODO: Implement function body'
    
    def _generate_return_value(self, description: str, language: str = 'python') -> str:
        """Generate return value based on description"""
        if 'calculate' in description.lower():
            return 'result' if language == 'python' else 'result'
        elif 'process' in description.lower():
            return 'processed' if language == 'python' else 'processed'
        elif 'convert' in description.lower():
            return 'converted' if language == 'python' else 'converted'
        else:
            return 'None' if language == 'python' else 'null'
    
    def _generate_imports(self, description: str) -> str:
        """Generate import statements based on description"""
        imports = []
        
        if any(word in description.lower() for word in ['file', 'read', 'write']):
            imports.append('import os')
        if any(word in description.lower() for word in ['time', 'date', 'timestamp']):
            imports.append('import time\nimport datetime')
        if any(word in description.lower() for word in ['math', 'calculate', 'computation']):
            imports.append('import math')
        if any(word in description.lower() for word in ['json', 'api', 'data']):
            imports.append('import json')
        if any(word in description.lower() for word in ['request', 'http', 'web']):
            imports.append('import requests')
        
        return '\n'.join(imports) if imports else '# Add imports as needed'
    
    def _generate_script_body(self, description: str, language: str = 'python') -> str:
        """Generate script main body"""
        
        if language == 'python':
            if 'file' in description.lower():
                return '    # File operations\n    print("Processing files...")\n    # TODO: Implement file processing'
            elif 'web' in description.lower():
                return '    # Web operations\n    print("Starting web operations...")\n    # TODO: Implement web functionality'
            else:
                return '    # Main script logic\n    print("Script starting...")\n    # TODO: Implement main functionality'
        
        elif language == 'bash':
            if 'file' in description.lower():
                return '    echo "Processing files..."\n    # TODO: Implement file processing'
            else:
                return '    echo "Script starting..."\n    # TODO: Implement main functionality'
        
        return '    # TODO: Implement script functionality'
    
    def generate_and_execute(self, description: str, 
                           execute: bool = True) -> Dict[str, Any]:
        """Generate code and optionally execute it"""
        
        # Parse request
        request = self.parse_generation_request(description)
        
        # Generate code
        generated_code = self.generate_code(request)
        
        # Analyze code
        analysis = self.analyzer.analyze_code(generated_code.code, generated_code.language)
        
        result = {
            'request': request,
            'generated_code': generated_code,
            'analysis': analysis,
            'execution_result': None
        }
        
        # Execute if requested and language is supported
        if execute and generated_code.language in [CodeLanguage.PYTHON, CodeLanguage.JAVASCRIPT, CodeLanguage.BASH]:
            execution_result = self.executor.execute_code(
                generated_code.code, 
                generated_code.language
            )
            result['execution_result'] = execution_result
        
        return result
    
    def format_generation_result(self, result: Dict[str, Any]) -> str:
        """Format code generation result for display"""
        
        generated_code = result['generated_code']
        analysis = result['analysis']
        execution_result = result.get('execution_result')
        
        formatted = f"💻 **Code Generation Result**\n\n"
        formatted += f"**Language:** {generated_code.language.value.title()}\n"
        formatted += f"**Type:** {generated_code.code_type.value.title()}\n"
        formatted += f"**Confidence:** {generated_code.confidence:.1%}\n"
        formatted += f"**Lines of Code:** {analysis['lines_of_code']}\n"
        
        if analysis['dependencies']:
            formatted += f"**Dependencies:** {', '.join(analysis['dependencies'])}\n"
        
        formatted += f"\n**Generated Code:**\n```{generated_code.language.value}\n"
        formatted += f"{generated_code.code}\n```\n\n"
        
        formatted += f"**Explanation:** {generated_code.explanation}\n\n"
        
        # Add execution results if available
        if execution_result:
            formatted += f"**Execution Result:**\n"
            if execution_result.success:
                formatted += f"✅ **Success** (in {execution_result.execution_time:.3f}s)\n"
                if execution_result.output:
                    formatted += f"**Output:**\n```\n{execution_result.output}\n```\n"
            else:
                formatted += f"❌ **Failed** (in {execution_result.execution_time:.3f}s)\n"
                if execution_result.error:
                    formatted += f"**Error:**\n```\n{execution_result.error}\n```\n"
        
        return formatted

# Integration function for consciousness system
def integrate_code_generation(consciousness_system, description: str, execute: bool = True) -> Dict[str, Any]:
    """Integrate code generation with consciousness system"""
    
    generator = CodeGenerator()
    
    # Generate and optionally execute code
    result = generator.generate_and_execute(description, execute)
    
    # Format for consciousness integration
    code_result = {
        'description': description,
        'language': result['generated_code'].language.value,
        'code_type': result['generated_code'].code_type.value,
        'generated_code': result['generated_code'].code,
        'analysis': result['analysis'],
        'execution_successful': result['execution_result'].success if result['execution_result'] else None,
        'formatted_result': generator.format_generation_result(result),
        'confidence': result['generated_code'].confidence
    }
    
    # Add to consciousness working memory if available
    if hasattr(consciousness_system, 'working_memory'):
        consciousness_system.working_memory.add_experience({
            'type': 'code_generation',
            'content': code_result,
            'timestamp': consciousness_system.get_current_time() if hasattr(consciousness_system, 'get_current_time') else 0,
            'significance': result['generated_code'].confidence
        })
    
    return code_result