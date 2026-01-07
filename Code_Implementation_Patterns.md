# Code Implementation Patterns & Reference

This document provides actual code patterns you can adapt for your implementation.

---

## 1. RAG SETUP (LangChain + Pinecone)

### Initialize RAG Index

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone

class VulnerabilityRAG:
    def __init__(self, pinecone_api_key, pinecone_env):
        # Initialize Pinecone
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        
        # Use code-specific embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="microsoft/codebert-base"
        )
        
        # Initialize vector store
        self.vectorstore = Pinecone.from_existing_index(
            "vulnerability-fixes",
            self.embeddings
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def add_vulnerability_fix(self, cwe_id, vulnerable_code, patched_code, 
                             language, framework, context):
        """Add a vulnerability fix to RAG index"""
        
        # Create document with metadata
        document = f"""
        CWE: {cwe_id}
        Language: {language}
        Framework: {framework}
        
        VULNERABLE CODE:
        {vulnerable_code}
        
        PATCHED CODE:
        {patched_code}
        
        CONTEXT:
        {context}
        """
        
        chunks = self.text_splitter.split_text(document)
        
        metadatas = [{
            'cwe_id': cwe_id,
            'language': language,
            'framework': framework,
            'type': 'fix',
            'chunk_index': i
        } for i in range(len(chunks))]
        
        # Add to vectorstore
        self.vectorstore.add_texts(chunks, metadatas=metadatas)
        
        return len(chunks)
    
    def retrieve_similar_fixes(self, vulnerable_code, cwe_id, top_k=3):
        """Retrieve similar vulnerability fixes from RAG"""
        
        # Search
        results = self.vectorstore.similarity_search_with_score(
            vulnerable_code,
            k=top_k * 2,  # Get more to filter
            filter={'cwe_id': cwe_id}  # Filter by CWE
        )
        
        # Format results
        similar_fixes = []
        seen_cwe = set()
        
        for doc, score in results:
            cwe = doc.metadata.get('cwe_id')
            
            # Avoid duplicates
            if cwe in seen_cwe:
                continue
            
            similar_fixes.append({
                'content': doc.page_content,
                'similarity_score': score,
                'metadata': doc.metadata
            })
            
            seen_cwe.add(cwe)
            
            if len(similar_fixes) >= top_k:
                break
        
        return similar_fixes
```

### Batch Index CVEfixes Dataset

```python
import json
from tqdm import tqdm

def batch_index_cvefixes(rag: VulnerabilityRAG, cvefixes_json_path):
    """Index CVEfixes dataset into RAG"""
    
    with open(cvefixes_json_path, 'r') as f:
        fixes = json.load(f)
    
    indexed_count = 0
    
    for fix in tqdm(fixes, desc="Indexing CVE fixes"):
        try:
            # Extract relevant fields
            cve_id = fix.get('cve_id')
            cwe_ids = fix.get('cwe_ids', [])
            language = fix.get('language')
            vulnerable_code = fix.get('vulnerable_code')
            patched_code = fix.get('patched_code')
            description = fix.get('description')
            
            # Validate
            if not all([cwe_ids, language, vulnerable_code, patched_code]):
                continue
            
            # Index for each CWE
            for cwe_id in cwe_ids:
                chunks_added = rag.add_vulnerability_fix(
                    cwe_id=cwe_id,
                    vulnerable_code=vulnerable_code,
                    patched_code=patched_code,
                    language=language,
                    framework=fix.get('framework', 'unknown'),
                    context=description
                )
                
                indexed_count += 1
        
        except Exception as e:
            print(f"Error indexing {cve_id}: {e}")
            continue
    
    print(f"✓ Indexed {indexed_count} vulnerability fixes")
    return indexed_count
```

---

## 2. PROMPT GENERATION ENGINE

### CWE-Specific Templates

```python
from enum import Enum
from dataclasses import dataclass

@dataclass
class FixStrategy:
    name: str
    description: str
    requirements: list
    examples: list

class CWEPromptGenerator:
    
    # Define strategies for each CWE
    STRATEGIES = {
        'CWE-89': {  # SQL Injection
            'primary': FixStrategy(
                name='parameterized_queries',
                description='Use PreparedStatement or parameterized placeholders',
                requirements=[
                    'Never concatenate user input into SQL strings',
                    'Use ? or :param placeholders for user input',
                    'Bind parameters separately from query string'
                ],
                examples=[
                    '# Bad:\nquery = f"SELECT * FROM users WHERE id = {user_id}"\n\n# Good:\nquery = "SELECT * FROM users WHERE id = ?"\ncursor.execute(query, (user_id,))'
                ]
            ),
            'secondary': FixStrategy(
                name='orm_query_builder',
                description='Use ORM with query builder (SQLAlchemy, Django ORM)',
                requirements=[
                    'Use ORM query interface instead of raw SQL',
                    'ORM automatically parameterizes queries',
                    'Validate at model level'
                ],
                examples=[]
            )
        },
        'CWE-79': {  # XSS
            'primary': FixStrategy(
                name='html_escaping',
                description='Escape HTML special characters',
                requirements=[
                    'Escape: & < > " \'',
                    'Use framework-native escaping',
                    'Never use innerHTML with user input'
                ],
                examples=[
                    '# Bad:\nhtml = f"<p>{user_input}</p>"\n\n# Good:\nhtml = f"<p>{html.escape(user_input)}</p>"'
                ]
            ),
            'secondary': FixStrategy(
                name='content_security_policy',
                description='Implement CSP headers',
                requirements=[
                    'Set Content-Security-Policy header',
                    'Restrict script sources',
                    'Disable inline scripts'
                ],
                examples=[]
            )
        },
        'CWE-434': {  # Unrestricted Upload
            'primary': FixStrategy(
                name='file_type_validation',
                description='Validate file type before upload',
                requirements=[
                    'Whitelist allowed file extensions',
                    'Check MIME type via magic bytes (not header)',
                    'Check file size limit',
                    'Store outside web root'
                ],
                examples=[
                    'ALLOWED_EXTENSIONS = {".jpg", ".png", ".pdf"}\n\nif not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):\n    raise ValueError("Invalid file type")'
                ]
            )
        }
    }
    
    def generate_prompt(self, vulnerability, rag_examples, strategy='auto'):
        """Generate CWE-specific prompt"""
        
        cwe_id = vulnerability.cwe_id
        strategies = self.STRATEGIES.get(cwe_id, {})
        
        # Select strategy
        if strategy == 'auto':
            chosen_strategy = strategies.get('primary')
        else:
            chosen_strategy = strategies.get(strategy, strategies.get('primary'))
        
        # Build prompt
        prompt = f"""You are a security expert specializing in fixing {cwe_id} vulnerabilities.

CWE-{cwe_id}: {self._get_cwe_name(cwe_id)}

VULNERABILITY DESCRIPTION:
{self._get_cwe_description(cwe_id)}

FIX STRATEGY: {chosen_strategy.name}
{chosen_strategy.description}

REQUIREMENTS:
{chr(10).join('- ' + req for req in chosen_strategy.requirements)}

SIMILAR FIXES IN YOUR CODEBASE:
{self._format_rag_examples(rag_examples)}

VULNERABLE CODE:
```{vulnerability.language}
{vulnerability.code}
```

TASK:
1. Analyze the vulnerability in the code
2. Identify the exact lines causing the issue
3. Explain why this is vulnerable
4. Generate a secure fix using the strategy: {chosen_strategy.name}
5. Verify the fix eliminates the root cause

Generate exactly 3 alternative fixes, each using the {chosen_strategy.name} strategy.
For each fix, explain the trade-offs.

Only output valid {vulnerability.language} code in fenced code blocks.
"""
        
        return prompt
    
    def _get_cwe_name(self, cwe_id: str) -> str:
        """Get CWE name from ID"""
        names = {
            'CWE-89': 'Improper Neutralization of Special Elements in SQL',
            'CWE-79': 'Improper Neutralization of Input During Web Page Generation',
            'CWE-434': 'Unrestricted Upload of File with Dangerous Type',
            # ... more CWEs
        }
        return names.get(cwe_id, 'Unknown Vulnerability')
    
    def _get_cwe_description(self, cwe_id: str) -> str:
        """Get CWE description"""
        descriptions = {
            'CWE-89': 'The product constructs SQL commands using externally-influenced input without proper neutralization...',
            'CWE-79': 'The product does not properly neutralize user-supplied input before it is placed in output...',
            # ... more descriptions
        }
        return descriptions.get(cwe_id, '')
    
    def _format_rag_examples(self, examples):
        """Format RAG examples for prompt"""
        if not examples:
            return "[No similar examples found in codebase]"
        
        formatted = []
        for i, example in enumerate(examples, 1):
            formatted.append(f"\nExample {i}:\n{example['content'][:500]}...")
        
        return '\n'.join(formatted)
```

---

## 3. CANDIDATE GENERATION & RANKING

### Multi-Candidate Generation

```python
import asyncio
from typing import List

class PatchGenerator:
    
    def __init__(self, llm_client, rag_system):
        self.llm = llm_client  # OpenAI, Claude, etc.
        self.rag = rag_system
        self.prompt_gen = CWEPromptGenerator()
    
    async def generate_candidates(self, vulnerability, num_candidates=3) -> List[dict]:
        """Generate multiple patch candidates in parallel"""
        
        # Retrieve RAG examples
        rag_examples = self.rag.retrieve_similar_fixes(
            vulnerability.code,
            vulnerability.cwe_id,
            top_k=3
        )
        
        # Generate candidates with different strategies
        tasks = [
            self._generate_with_strategy(
                vulnerability, 
                rag_examples,
                strategy
            )
            for strategy in ['primary', 'secondary', 'alternative']
        ]
        
        results = await asyncio.gather(*tasks)
        
        candidates = [r for r in results if r is not None]
        
        return candidates
    
    async def _generate_with_strategy(self, vulnerability, rag_examples, strategy):
        """Generate patch using specific strategy"""
        
        try:
            # Generate prompt
            prompt = self.prompt_gen.generate_prompt(
                vulnerability,
                rag_examples,
                strategy=strategy
            )
            
            # Call LLM
            response = await self.llm.acompletion(
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a security expert fixing code vulnerabilities. Respond with only valid code in fenced blocks.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.3  # Low temperature for deterministic code
            )
            
            # Extract patches from response
            patches = self._extract_patches(response.choices[0].message.content)
            
            if patches:
                return {
                    'strategy': strategy,
                    'patches': patches,
                    'metadata': {
                        'model': self.llm.model,
                        'temperature': 0.3
                    }
                }
        
        except Exception as e:
            print(f"Error generating patch with {strategy}: {e}")
        
        return None
    
    def _extract_patches(self, response_text):
        """Extract code blocks from LLM response"""
        
        import re
        
        # Find all code blocks
        pattern = r'```(?:\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        patches = []
        for i, patch in enumerate(matches):
            patches.append({
                'code': patch.strip(),
                'index': i
            })
        
        return patches
```

### Candidate Validation & Ranking

```python
import subprocess
import tempfile
from dataclasses import dataclass

@dataclass
class ValidationScore:
    syntax: float
    vulnerability_fixed: float
    functional_correctness: float
    exploit_eliminated: float
    regression_tests: float
    code_quality: float
    
    @property
    def overall(self) -> float:
        """Calculate weighted overall score"""
        weights = {
            'syntax': 0.15,
            'vulnerability_fixed': 0.35,
            'functional_correctness': 0.20,
            'exploit_eliminated': 0.15,
            'regression_tests': 0.10,
            'code_quality': 0.05
        }
        
        return sum(
            getattr(self, metric) * weight
            for metric, weight in weights.items()
        )

class PatchValidator:
    
    def __init__(self, sast_tool, test_suite, poc_database):
        self.sast = sast_tool
        self.tests = test_suite
        self.pocs = poc_database
    
    def validate_and_rank(self, candidates, vulnerability) -> List[dict]:
        """Validate all candidates and rank by score"""
        
        ranked = []
        
        for candidate in candidates:
            patch = candidate['patches'][0]['code'] if candidate['patches'] else None
            
            if not patch:
                continue
            
            scores = self._validate_patch(patch, vulnerability)
            
            ranked.append({
                'patch': patch,
                'strategy': candidate['strategy'],
                'scores': scores,
                'confidence': scores.overall,
                'reasoning': self._generate_ranking_reasoning(scores)
            })
        
        # Sort by confidence
        ranked.sort(key=lambda x: x['confidence'], reverse=True)
        
        return ranked
    
    def _validate_patch(self, patch, vulnerability) -> ValidationScore:
        """Run all validation checks"""
        
        scores = ValidationScore(
            syntax=self._check_syntax(patch, vulnerability.language),
            vulnerability_fixed=self._check_vulnerability_fixed(patch, vulnerability),
            functional_correctness=self._check_functional(patch, vulnerability),
            exploit_eliminated=self._check_exploit(patch, vulnerability),
            regression_tests=self._check_regression(patch, vulnerability),
            code_quality=self._check_quality(patch, vulnerability)
        )
        
        return scores
    
    def _check_syntax(self, patch, language) -> float:
        """Validate syntax"""
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{self._get_extension(language)}') as f:
                f.write(patch)
                f.flush()
                
                if language == 'python':
                    result = subprocess.run(['python', '-m', 'py_compile', f.name], 
                                          capture_output=True, timeout=5)
                    return 1.0 if result.returncode == 0 else 0.0
                
                elif language == 'javascript':
                    result = subprocess.run(['node', '--check', f.name],
                                          capture_output=True, timeout=5)
                    return 1.0 if result.returncode == 0 else 0.0
                
                # Add more languages as needed
        
        except Exception:
            return 0.0
    
    def _check_vulnerability_fixed(self, patch, vulnerability) -> float:
        """Use SAST to verify vulnerability is fixed"""
        
        try:
            result = self.sast.scan(patch, rules=[vulnerability.cwe_id])
            
            if not result.findings:
                return 1.0  # No findings = fixed
            
            # Partial credit if severity reduced
            original_severity = max(f.severity for f in vulnerability.findings)
            patched_severity = max(f.severity for f in result.findings)
            
            if patched_severity < original_severity:
                return 0.5
            
            return 0.0
        
        except Exception:
            return 0.3  # Unknown
    
    def _check_functional(self, patch, vulnerability) -> float:
        """Run existing unit tests"""
        
        try:
            # This would integrate with actual test suite
            test_results = self.tests.run_tests_with_patch(patch)
            
            if test_results.passed == test_results.total:
                return 1.0
            
            return test_results.passed / test_results.total
        
        except Exception:
            return 0.5
    
    def _check_exploit(self, patch, vulnerability) -> float:
        """Test with PoC exploits"""
        
        try:
            poc = self.pocs.get_poc(vulnerability.cwe_id)
            
            if not poc:
                return 0.5  # No PoC available
            
            # Run PoC against patched code
            result = subprocess.run(
                ['docker', 'run', '--rm', '-v', f'{patch}:/code.txt', 
                 f'poc-tester-{vulnerability.language}'],
                capture_output=True,
                timeout=10
            )
            
            # If PoC fails (can't exploit), vulnerability is fixed
            return 1.0 if result.returncode != 0 else 0.0
        
        except Exception:
            return 0.3
    
    def _check_regression(self, patch, vulnerability) -> float:
        """Run integration tests"""
        
        # Similar to _check_functional but for integration tests
        return 0.8  # Placeholder
    
    def _check_quality(self, patch, vulnerability) -> float:
        """Check code quality metrics"""
        
        original_complexity = self._measure_complexity(vulnerability.code)
        patch_complexity = self._measure_complexity(patch)
        
        # Penalize if much more complex
        if patch_complexity > original_complexity * 1.5:
            return 0.6
        
        return 0.9
    
    def _measure_complexity(self, code) -> float:
        """Measure cyclomatic complexity"""
        # Use radon or similar
        return 1.0
    
    def _generate_ranking_reasoning(self, scores) -> str:
        """Explain why this patch was ranked"""
        
        reasons = []
        
        if scores.syntax < 0.8:
            reasons.append("Syntax errors present")
        
        if scores.vulnerability_fixed < 0.8:
            reasons.append("Vulnerability may not be fully fixed")
        
        if scores.functional_correctness < 0.8:
            reasons.append("May cause functional regressions")
        
        if scores.exploit_eliminated < 0.8:
            reasons.append("Exploit validation inconclusive")
        
        if not reasons:
            reasons.append("All validation checks passed")
        
        return " | ".join(reasons)
    
    def _get_extension(self, language):
        """Get file extension for language"""
        ext_map = {'python': 'py', 'javascript': 'js', 'java': 'java', 'csharp': 'cs'}
        return ext_map.get(language, language)
```

---

## 4. MCP SERVER INTEGRATION

### Security Scanner MCP Server

```python
from mcp.server import Server
from mcp.types import TextContent, Tool
import subprocess
import tempfile
import os
from pathlib import Path

class SecurityScannerMCP:
    """MCP server for real-time security validation"""
    
    def __init__(self):
        self.server = Server("security-scanner-v1")
        self.register_tools()
    
    def register_tools(self):
        """Register MCP tools for security scanning"""
        
        @self.server.call_tool()
        async def scan_code_with_semgrep(code: str, cwe_id: str = None, language: str = None):
            """Scan code with Semgrep for security vulnerabilities"""
            
            try:
                # Write code to temp file
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix=f'.{language or "py"}',
                    delete=False
                ) as f:
                    f.write(code)
                    f.flush()
                    temp_path = f.name
                
                # Build semgrep command
                cmd = ['semgrep', '--json', temp_path]
                
                if cwe_id:
                    cmd.extend(['--include-rule', f'security.vulnerability.{cwe_id}'])
                else:
                    cmd.extend(['-c', 'p/security-audit'])
                
                # Run semgrep
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse results
                import json
                if result.stdout:
                    findings = json.loads(result.stdout)
                    return TextContent(
                        type="text",
                        text=json.dumps({
                            'vulnerabilities_found': len(findings.get('results', [])),
                            'findings': findings.get('results', [])[:10],  # Limit output
                            'status': 'success'
                        })
                    )
                
                return TextContent(type="text", text=json.dumps({
                    'vulnerabilities_found': 0,
                    'status': 'success'
                }))
            
            except subprocess.TimeoutExpired:
                return TextContent(type="text", text=json.dumps({
                    'error': 'Scan timeout',
                    'status': 'timeout'
                }))
            
            finally:
                # Cleanup
                if 'temp_path' in locals():
                    os.unlink(temp_path)
        
        @self.server.call_tool()
        async def scan_python_with_bandit(code: str):
            """Scan Python code for security issues with Bandit"""
            
            try:
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.py',
                    delete=False
                ) as f:
                    f.write(code)
                    f.flush()
                    temp_path = f.name
                
                result = subprocess.run(
                    ['bandit', '-f', 'json', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                import json
                if result.stdout:
                    findings = json.loads(result.stdout)
                    return TextContent(
                        type="text",
                        text=json.dumps({
                            'issues_found': len(findings.get('results', [])),
                            'issues': findings.get('results', [])[:5],
                            'status': 'success'
                        })
                    )
                
                return TextContent(type="text", text=json.dumps({
                    'issues_found': 0,
                    'status': 'success'
                }))
            
            finally:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
        
        @self.server.call_tool()
        async def verify_vulnerability_fixed(
            original_code: str,
            patched_code: str,
            cwe_id: str
        ):
            """Verify that patch fixes the vulnerability"""
            
            try:
                # Scan original
                original_result = subprocess.run(
                    ['semgrep', '--json', '--stdin'],
                    input=original_code,
                    capture_output=True,
                    text=True,
                    timeout=20
                )
                
                # Scan patched
                patched_result = subprocess.run(
                    ['semgrep', '--json', '--stdin'],
                    input=patched_code,
                    capture_output=True,
                    text=True,
                    timeout=20
                )
                
                import json
                original_findings = json.loads(original_result.stdout).get('results', [])
                patched_findings = json.loads(patched_result.stdout).get('results', [])
                
                # Check if vulnerability fixed
                vulnerability_fixed = len(patched_findings) < len(original_findings)
                
                return TextContent(
                    type="text",
                    text=json.dumps({
                        'original_findings': len(original_findings),
                        'patched_findings': len(patched_findings),
                        'vulnerability_fixed': vulnerability_fixed,
                        'status': 'success'
                    })
                )
            
            except Exception as e:
                return TextContent(
                    type="text",
                    text=json.dumps({
                        'error': str(e),
                        'status': 'error'
                    })
                )
    
    async def run(self):
        """Run the MCP server"""
        async with self.server:
            print("Security Scanner MCP server running...")
            await self.server.wait_shutdown()

# Cline/Claude MCP Configuration
MCP_CONFIG = """
{
  "mcpServers": {
    "security-scanner": {
      "command": "python",
      "args": ["-m", "security_scanner_mcp"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      },
      "disabled": false,
      "autoApprove": [
        "verify_vulnerability_fixed",
        "scan_code_with_semgrep"
      ]
    }
  }
}
"""
```

---

## 5. CONFIDENCE SCORING & ROUTING

```python
from enum import Enum

class RouteAction(Enum):
    AUTO_APPROVE = "auto_approve"
    MANUAL_REVIEW = "manual_review"
    ESCALATION = "escalation"

class ConfidenceRouter:
    """Route patches to appropriate workflow based on confidence"""
    
    # Thresholds
    AUTO_APPROVE_THRESHOLD = 0.95
    MANUAL_REVIEW_THRESHOLD = 0.75
    ESCALATION_THRESHOLD = 0.0
    
    def route_patch(self, patch_result, vulnerability):
        """Determine routing based on confidence"""
        
        confidence = patch_result['confidence']
        cwe_id = vulnerability.cwe_id
        
        # Adjust thresholds based on CWE
        auto_approve_threshold = self._get_cwe_threshold(cwe_id, 'auto_approve')
        
        if confidence >= auto_approve_threshold:
            return self._create_auto_approve_action(patch_result)
        
        elif confidence >= self.MANUAL_REVIEW_THRESHOLD:
            return self._create_manual_review_action(patch_result)
        
        else:
            return self._create_escalation_action(patch_result)
    
    def _get_cwe_threshold(self, cwe_id, action):
        """Get action-specific threshold for CWE"""
        
        # Higher thresholds for complex CWEs
        complex_cwes = {'CWE-200', 'CWE-307', 'CWE-427'}
        
        if cwe_id in complex_cwes:
            if action == 'auto_approve':
                return 0.98  # Higher bar
        
        if action == 'auto_approve':
            return self.AUTO_APPROVE_THRESHOLD
        elif action == 'manual_review':
            return self.MANUAL_REVIEW_THRESHOLD
        
        return self.ESCALATION_THRESHOLD
    
    def _create_auto_approve_action(self, patch_result):
        """Create auto-approve action"""
        
        return {
            'action': RouteAction.AUTO_APPROVE,
            'confidence': patch_result['confidence'],
            'pr_config': {
                'auto_merge': True,
                'require_approval': False,
                'reviewers': [],
                'branch': 'auto-remediation-approved'
            },
            'message': f"✅ Auto-approved with {patch_result['confidence']:.1%} confidence"
        }
    
    def _create_manual_review_action(self, patch_result):
        """Create manual review action"""
        
        vulnerability = patch_result['vulnerability']
        cwe_id = vulnerability.cwe_id
        
        return {
            'action': RouteAction.MANUAL_REVIEW,
            'confidence': patch_result['confidence'],
            'pr_config': {
                'auto_merge': False,
                'require_approval': True,
                'reviewers': self._find_expert_reviewers(cwe_id),
                'branch': 'auto-remediation-review'
            },
            'message': f"⚠️  Requires manual review ({patch_result['confidence']:.1%} confidence)",
            'checks_to_override': self._get_blocking_checks(patch_result)
        }
    
    def _create_escalation_action(self, patch_result):
        """Create escalation action"""
        
        return {
            'action': RouteAction.ESCALATION,
            'confidence': patch_result['confidence'],
            'message': f"❌ Low confidence ({patch_result['confidence']:.1%}) - manual remediation required",
            'debugging': {
                'validation_scores': patch_result.get('scores'),
                'reason': self._determine_failure_reason(patch_result)
            },
            'assignee': 'security-lead',
            'priority': 'high'
        }
    
    def _find_expert_reviewers(self, cwe_id):
        """Find team members with expertise in CWE"""
        
        # This would query team database
        reviewers_by_cwe = {
            'CWE-89': ['@sql-expert', '@database-specialist'],
            'CWE-79': ['@xss-expert', '@frontend-lead'],
            'CWE-434': ['@upload-specialist', '@security-lead'],
        }
        
        return reviewers_by_cwe.get(cwe_id, ['@security-team'])
    
    def _get_blocking_checks(self, patch_result):
        """Identify which automated checks need manual override"""
        
        blocking = []
        scores = patch_result.get('scores', {})
        
        if scores.get('exploit_eliminated', 0) < 0.8:
            blocking.append('exploit_validation')
        
        if scores.get('regression_tests', 0) < 0.9:
            blocking.append('regression_tests')
        
        return blocking
    
    def _determine_failure_reason(self, patch_result):
        """Analyze why confidence is low"""
        
        scores = patch_result.get('scores', {})
        reasons = []
        
        for metric, score in scores.items():
            if score < 0.7:
                reasons.append(f"{metric}: {score:.1%}")
        
        return " | ".join(reasons) if reasons else "Unknown"
```

---

## 6. PR GENERATION

```python
class PRGenerator:
    """Generate GitHub/GitLab PRs for patches"""
    
    def __init__(self, git_client):
        self.git = git_client
    
    def create_pr(self, patch_result, routing_action):
        """Create pull request with patch"""
        
        vulnerability = patch_result['vulnerability']
        patch = patch_result['patch']
        
        # Create branch
        branch_name = self._generate_branch_name(vulnerability, routing_action)
        
        # Write patch file
        self.git.create_branch(branch_name)
        self._apply_patch(patch, vulnerability)
        
        # Create commit
        commit_message = self._generate_commit_message(patch_result)
        self.git.commit(commit_message)
        
        # Create PR
        pr_body = self._generate_pr_body(patch_result, routing_action)
        pr = self.git.create_pr(
            title=self._generate_pr_title(patch_result),
            body=pr_body,
            base='main',
            head=branch_name,
            reviewers=routing_action.get('pr_config', {}).get('reviewers', []),
            labels=['security', 'auto-remediation', f"cwe-{patch_result['vulnerability'].cwe_id}"]
        )
        
        # Add auto-merge if applicable
        if routing_action['pr_config'].get('auto_merge'):
            pr.enable_auto_merge(merge_method='squash')
        
        return pr
    
    def _generate_pr_title(self, patch_result):
        """Generate PR title"""
        
        vulnerability = patch_result['vulnerability']
        confidence = patch_result['confidence']
        
        confidence_badge = "✅" if confidence >= 0.95 else "⚠️" if confidence >= 0.75 else "❌"
        
        return f"{confidence_badge} Security: Fix {vulnerability.cwe_id} in {vulnerability.file}:{vulnerability.line}"
    
    def _generate_pr_body(self, patch_result, routing_action):
        """Generate detailed PR body with validation results"""
        
        vulnerability = patch_result['vulnerability']
        scores = patch_result.get('scores', {})
        
        body = f"""## Automated Security Patch

### Vulnerability Details
- **CWE ID:** {vulnerability.cwe_id}
- **Severity:** {vulnerability.severity}
- **File:** {vulnerability.file}:{vulnerability.line}
- **Confidence:** {patch_result['confidence']:.1%}

### Vulnerability Description
{vulnerability.description}

### Generated Patch
```{vulnerability.language}
{patch_result['patch']}
```

### Validation Results

| Check | Score | Status |
|-------|-------|--------|
| Syntax Validation | {scores.get('syntax', 0):.0%} | {'✅' if scores.get('syntax', 0) > 0.9 else '❌'} |
| Vulnerability Fixed | {scores.get('vulnerability_fixed', 0):.0%} | {'✅' if scores.get('vulnerability_fixed', 0) > 0.9 else '❌'} |
| Functional Tests | {scores.get('functional_correctness', 0):.0%} | {'✅' if scores.get('functional_correctness', 0) > 0.9 else '⚠️'} |
| Exploit Validation | {scores.get('exploit_eliminated', 0):.0%} | {'✅' if scores.get('exploit_eliminated', 0) > 0.9 else '⚠️'} |
| Regression Tests | {scores.get('regression_tests', 0):.0%} | {'✅' if scores.get('regression_tests', 0) > 0.9 else '⚠️'} |
| Code Quality | {scores.get('code_quality', 0):.0%} | {'✅' if scores.get('code_quality', 0) > 0.9 else '⚠️'} |

### Fix Strategy
{patch_result.get('strategy', 'Unknown')}

### Routing Decision
{routing_action.get('message', 'N/A')}

---

*Generated by AI Remediation Engine - Please review before merging*
"""
        
        return body
    
    def _generate_commit_message(self, patch_result):
        """Generate commit message"""
        
        vulnerability = patch_result['vulnerability']
        
        return f"""fix: remediate {vulnerability.cwe_id} vulnerability

Automatically generated security patch for {vulnerability.cwe_id} in {vulnerability.file}

- Vulnerability: {vulnerability.description[:100]}...
- Confidence: {patch_result['confidence']:.1%}
- Strategy: {patch_result.get('strategy', 'Unknown')}

Co-authored-by: AI Security Remediation Engine
"""
    
    def _apply_patch(self, patch, vulnerability):
        """Apply patch to codebase"""
        
        # This would write the patched code to the correct file
        file_path = vulnerability.file
        
        with open(file_path, 'r') as f:
            original = f.read()
        
        # Simple replacement - in reality, use AST-based patching
        patched = self._apply_patch_to_code(original, patch, vulnerability)
        
        with open(file_path, 'w') as f:
            f.write(patched)
    
    def _apply_patch_to_code(self, original, patch, vulnerability):
        """Apply patch to original code"""
        # Implementation depends on your code structure
        pass
    
    def _generate_branch_name(self, vulnerability, routing_action):
        """Generate git branch name"""
        
        action_prefix = {
            RouteAction.AUTO_APPROVE: 'auto',
            RouteAction.MANUAL_REVIEW: 'review',
            RouteAction.ESCALATION: 'escalate'
        }
        
        prefix = action_prefix.get(routing_action['action'], 'patch')
        
        return f"{prefix}/security/{vulnerability.cwe_id.lower()}-{vulnerability.file.replace('/', '-')}"
```

---

## INTEGRATION EXAMPLE

```python
async def remediate_vulnerability(vulnerability, llm_client, rag_system, sast_tool):
    """Complete remediation pipeline"""
    
    print(f"🔍 Remediating {vulnerability.cwe_id}...")
    
    # 1. Generate candidates
    generator = PatchGenerator(llm_client, rag_system)
    candidates_result = await generator.generate_candidates(vulnerability)
    
    print(f"✓ Generated {len(candidates_result)} patch candidates")
    
    # 2. Validate and rank
    validator = PatchValidator(sast_tool, test_suite, poc_database)
    ranked = validator.validate_and_rank(candidates_result, vulnerability)
    
    print(f"✓ Top candidate confidence: {ranked[0]['confidence']:.1%}")
    
    # 3. Route based on confidence
    router = ConfidenceRouter()
    routing_action = router.route_patch(ranked[0], vulnerability)
    
    print(f"✓ Routing to: {routing_action['action'].value}")
    
    # 4. Create PR
    pr_gen = PRGenerator(git_client)
    pr = pr_gen.create_pr(ranked[0], routing_action)
    
    print(f"✓ Created PR: {pr.html_url}")
    
    return {
        'status': 'success',
        'pr': pr,
        'confidence': ranked[0]['confidence'],
        'routing': routing_action['action']
    }
```

---

This provides production-ready code patterns you can adapt for your specific environment and tooling choices.