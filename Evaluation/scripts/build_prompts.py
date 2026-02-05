#!/usr/bin/env python3
"""
Read data/*/cases + templates to generate judge prompts.
Render final judge prompts from Task Card (YAML) and templates.
"""

import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.path_config import load_global_config, build_path_settings, resolve_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build final prompts from templates and case data"""
    
    def __init__(self, project_root: str = ".", global_config: str = "configs/global.yaml"):
        self.project_root = Path(project_root)
        self.global_config = load_global_config(self.project_root, global_config)
        self.path_settings = build_path_settings(self.global_config)
        self.paths = resolve_paths(self.project_root, self.path_settings)
        self.data_root = self.paths["data_root"]
        self.templates_root = self.paths["templates_root"]
        self.prompts_root = self.paths["prompts_root"]
        self.generations_root_str = self.path_settings["generations_root"]
        self.qa_root = self.templates_root / "Q&A"
        
        # Load judge prompt template
        self.judge_template = self._load_judge_template()
        
        # Load task-specific Q&A
        self.task_qa_map = self._load_task_qa_templates()
        
    def _load_judge_template(self) -> str:
        """Load judge prompt template"""
        template_path = self.templates_root / "judge_prompt.template.txt"
        if not template_path.exists():
            logger.warning(f"Judge template not found: {template_path}, using default template")
            return self._get_default_judge_template()
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _get_default_judge_template(self) -> str:
        """Default judge template"""
        return """System:
You are a professional multimodal evaluation specialist tasked with assessing AI-generated images.

CRITICAL: Your response MUST be valid JSON only. Return ONLY the raw JSON object.

User:
[Task Type]
{TASK_TYPE}

[Reference Images]
{REFS_BLOCK}

[Original Generation Instruction]
{GEN_PROMPT_BLOCK}

[Candidate Image to Evaluate]
{CANDIDATE_PATH}

[Task Requirements]
{GOAL_AND_CONSTRAINTS}

[Evaluation Principles]

1. Strict Comparison:
   Always compare against reference images, the generation instruction, and task requirements.

2. Binary Assessment:
   For each checkpoint, output only:
   - 1 = PASS (requirement clearly met)
   - 0 = FAIL (requirement not met or partially met)

3. Hard Constraint Enforcement:
   If a hard constraint fails, mark it as 0 in "hard_constraint_results".
   Any dimension affected is capped ≤2 points (handled programmatically).

4. Evidence-based Judging:
   Base decisions strictly on observable evidence in the candidate image.
   Be strict but fair — only mark PASS when clearly satisfied.

{TASK_QA_BLOCK}

[Output JSON Format]
You MUST output JSON in this exact structure:

{CHECKPOINT_JSON_FORMAT}

Remember: Return ONLY the JSON object, nothing else.
"""
    
    def _load_task_qa_templates(self) -> Dict[str, str]:
        qa_map = {}
        
        task_qa_files = {
            "object_composition": "object_composition.qa.txt",
            "spatial_geometric_constraints": "spatial_constraints.qa.txt",
            "local_element_transfer": "part_transfer.qa.txt",
            "attribute_style_decoupling": "style_disentangle.qa.txt",
            "foreground_background_composition": "fg_bg_replace.qa.txt",
            "story_infer": "causal_story.qa.txt"
        }
        
        for task_type, qa_file in task_qa_files.items():
            qa_path = self.qa_root / qa_file
            if qa_path.exists():
                with open(qa_path, 'r', encoding='utf-8') as f:
                    qa_map[task_type] = f.read().strip()
            else:
                logger.warning(f"Q&A file not found: {qa_path}")
                qa_map[task_type] = "No specific Q&A checks for this task."
        
        return qa_map
    
    def load_case(self, case_path: Path) -> Optional[Dict]:
        """Load a single case YAML file"""
        try:
            with open(case_path, 'r', encoding='utf-8') as f:
                case_data = yaml.safe_load(f)
            return case_data
        except Exception as e:
            logger.error(f"Failed to load case {case_path}: {e}")
            return None
    
    def build_judge_prompt(self, case_data: Dict, task_type: str) -> str:
        """Build judge prompt for a single case"""
        
        # Extract case info
        case_id = case_data.get('case_id', 'unknown')
        refs = case_data.get('refs', {})
        generation_prompt = case_data.get('generation_prompt', '')
        goal = case_data.get('goal', '')
        constraints_hard = case_data.get('constraints_hard', [])
        
        # Build blocks
        # 1. Reference images block
        refs_block = self._format_refs_block(refs)
        
        # 2. Generation prompt block
        gen_prompt_block = f'"{generation_prompt}"'
        
        # 3. Goal and constraints block
        goal_constraints = self._format_goal_constraints(goal, constraints_hard)
        
        # 4. Candidate image path (placeholder, replaced at runtime)
        candidate_path = f"{self.generations_root_str}/<model>/{task_type}/{case_id}.png"
        
        # 5. Task-specific Q&A (verification checkpoints)
        task_qa = self.task_qa_map.get(task_type, "[Verification Checkpoints]\nNo specific checkpoints defined.")
        
        # 5.5. Add task-specific extra constraints (before TASK_QA_BLOCK)
        task_specific_constraints = self._get_task_specific_constraints(task_type)
        if task_specific_constraints:
            task_qa = task_specific_constraints + "\n\n" + task_qa
        
        # 6. JSON output format (generated from checkpoints)
        checkpoint_json_format = self._generate_checkpoint_json_format(task_qa, constraints_hard)
        
        prompt = self.judge_template.format(
            TASK_TYPE=task_type,
            REFS_BLOCK=refs_block,
            GEN_PROMPT_BLOCK=gen_prompt_block,
            GOAL_AND_CONSTRAINTS=goal_constraints,
            CANDIDATE_PATH=candidate_path,
            TASK_QA_BLOCK=task_qa,
            CHECKPOINT_JSON_FORMAT=checkpoint_json_format
        )
        
        return prompt
    
    def _get_task_specific_constraints(self, task_type: str) -> str:
        """Get task-specific extra constraints"""
        constraints_map = {
            "object_composition": """[Additional Evaluation Guidelines for Object Composition]
- This task checks if all referenced subjects are correctly included, arranged, and integrated.
- Each checkpoint must be evaluated strictly as:
  1 = requirement fully and clearly satisfied
  0 = missing, unclear, distorted, or partially satisfied
- Do NOT assign 1 if any subject is missing, merged, or poorly placed.
- Visual realism or artistic quality does NOT compensate for incorrect content.
- Typical FAIL cases:
  - One or more referenced objects missing or deformed.
  - Subjects overlapping unnaturally or floating.
  - Wrong background or spatial positioning.""",
            
            "spatial_geometric_constraints": """[Additional Evaluation Guidelines for Spatial Geometric Constraints]
- This task checks spatial arrangement and geometric correctness.
- Each checkpoint must be scored as:
  1 = all spatial relationships, sizes, and positions perfectly match the instruction
  0 = any mismatch, implausible geometry, or unclear arrangement
- Do NOT assign 1 if positions are reversed, scales differ, or contact physics fail.
- Typical FAIL cases:
  - Subject in wrong position or facing wrong direction.
  - Misaligned perspective or inconsistent shadow direction.
  - Objects intersecting or floating unnaturally.""",
            
            "local_element_transfer": """[Additional Evaluation Guidelines for Local Element Transfer]
- This task checks precise transfer of specified local elements or attributes.
- Each checkpoint must be scored as:
  1 = element fully transferred with correct position and appearance
  0 = element missing, distorted, or transferred incorrectly
- Do NOT assign 1 if identity of target subject is altered.
- Typical FAIL cases:
  - Element not transferred or placed on wrong subject.
  - Transferred item loses shape, color, or material fidelity.
  - New irrelevant elements appear.""",
            
            "attribute_style_decoupling": """[Additional Evaluation Guidelines for Attribute Style Decoupling]
- This task checks whether subject, style, and background are correctly disentangled.
- Each checkpoint must be evaluated as:
  1 = specific content applied exactly to intended scope
  0 = any leakage, mixing, or missing component
- Do NOT assign 1 if style object leaks into structure or if subject identity is altered.
- Typical FAIL cases:
  - Style from reference B introduces new shapes.
  - Background inconsistent with reference C.
  - Subject replaced or partially lost during style transfer.""",
            
            "foreground_background_composition": """[Additional Evaluation Guidelines for Foreground Background Composition]
- This task checks separation and recombination of foreground and background.
- Each checkpoint must be scored as:
  1 = clear extraction, clean boundary, correct placement
  0 = any boundary artifacts, blending issues, or background mismatch
- Do NOT assign 1 if lighting, color tone, or shadows are inconsistent.
- Typical FAIL cases:
  - Wrong or mismatched background.
  - Foreground edges blurred, haloed, or incomplete.
  - Background detail overwritten by foreground spill.""",
            
            "story_infer": """[Additional Evaluation Guidelines for Story Inference]
- This task checks logical continuity and causal reasoning.
- Each checkpoint must be scored as:
  1 = the event logically follows previous frames and obeys physics
  0 = any implausibility, contradiction, or skipped step
- Do NOT assign 1 if sequence is ambiguous or partially correct.
- Typical FAIL cases:
  - Physically impossible event (object teleports or vanishes).
  - Reversed cause-effect relationship.
  - Missing intermediate step required for continuity."""
        }
        
        return constraints_map.get(task_type, "")
    
    def _format_refs_block(self, refs: Dict[str, str]) -> str:
        """Format reference images block"""
        if not refs:
            return "No reference images."
        
        lines = []
        for ref_name, ref_path in refs.items():
            lines.append(f"- {ref_name}: {ref_path}")
        return "\n".join(lines)
    
    def _format_goal_constraints(self, goal: str, hard: List[str]) -> str:
        """Format goal and constraints"""
        lines = [f"Goal: {goal}"]
        
        if hard:
            lines.append("Hard Constraints (if violated, relevant dimension scores capped ≤2):")
            for i, constraint in enumerate(hard, 1):
                lines.append(f"- H{i}: {constraint}")
        
        return "\n".join(lines)
    
    def _generate_checkpoint_json_format(self, task_qa: str, hard_constraints: List[str]) -> str:
        """Generate JSON format example from task Q&A"""
        import re
        
        # Extract checkpoint IDs and hard-constraint links (e.g., A_check_1 (H1), B_check_2)
        checkpoint_pattern = r'([A-G])_check_(\d+)(?:\s*\(H(\d+)\))?'
        checkpoints = re.findall(checkpoint_pattern, task_qa)
        
        # Build checkpoint_results section
        checkpoint_lines = []
        for dim, num, hard_id in checkpoints:
            if hard_id:
                checkpoint_lines.append(f'    "{dim}_check_{num}": {{"pass": 0 or 1, "hard_id": "H{hard_id}"}},')
            else:
                checkpoint_lines.append(f'    "{dim}_check_{num}": {{"pass": 0 or 1}},')
        
        # Remove trailing comma
        if checkpoint_lines:
            checkpoint_lines[-1] = checkpoint_lines[-1].rstrip(',')
        
        # Build hard_constraint_results section
        hard_constraint_lines = []
        for i in range(1, len(hard_constraints) + 1):
            hard_constraint_lines.append(f'    "H{i}": 0 or 1,')
        
        # Remove trailing comma
        if hard_constraint_lines:
            hard_constraint_lines[-1] = hard_constraint_lines[-1].rstrip(',')
        
        # Assemble full JSON format
        json_format = "{\n"
        json_format += '  "checkpoint_results": {\n'
        json_format += '\n'.join(checkpoint_lines) if checkpoint_lines else '    // No checkpoints defined'
        json_format += '\n  },\n'
        json_format += '  "hard_constraint_results": {\n'
        json_format += '\n'.join(hard_constraint_lines) if hard_constraint_lines else '    // No hard constraints'
        json_format += '\n  },\n'
        json_format += '  "rationale_short": "<<=200 chars>>",\n'
        json_format += '  "flags": [\n'
        json_format += '    "..."\n'
        json_format += '  ]\n'
        json_format += '}'
        
        return json_format
    
    def process_task_type(self, task_type: str) -> int:
        """Process all cases for a task type"""
        task_dir = self.data_root / task_type / "cases"
        
        if not task_dir.exists():
            logger.warning(f"Task directory not found: {task_dir}")
            return 0
        
        # Create output directory
        output_dir = self.prompts_root / "judge" / task_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all case files
        case_files = sorted(task_dir.glob("*.yaml"))
        processed = 0
        
        for case_file in case_files:
            case_data = self.load_case(case_file)
            if not case_data:
                continue
            
            # Build prompt
            prompt = self.build_judge_prompt(case_data, task_type)
            
            # Save prompt
            case_id = case_data.get('case_id', case_file.stem)
            output_file = output_dir / f"{case_id}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            processed += 1
            if processed % 10 == 0:
                logger.info(f"Processed {task_type}: {processed}/{len(case_files)}")
        
        logger.info(f"Task {task_type} complete: {processed} cases")
        return processed
    
    def process_all_tasks(self) -> Dict[str, int]:
        """Process all task types"""
        # Supported task types
        task_types = [
            "object_composition",
            "spatial_geometric_constraints",
            "local_element_transfer",
            "attribute_style_decoupling",
            "foreground_background_composition",
            "story_infer"
        ]
        
        results = {}
        for task_type in task_types:
            count = self.process_task_type(task_type)
            results[task_type] = count
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Build judge prompts from templates and case data")
    parser.add_argument("--project-root", type=str, default=".",
                       help="Project root directory (default: current directory)")
    parser.add_argument("--global-config", type=str, default="configs/global.yaml",
                       help="Path to global.yaml (default: configs/global.yaml)")
    parser.add_argument("--task-type", type=str, default="all",
                       help="Specific task type to process, or 'all' for all tasks")
    
    args = parser.parse_args()
    
    builder = PromptBuilder(project_root=args.project_root, global_config=args.global_config)
    
    if args.task_type == "all":
        results = builder.process_all_tasks()
        print("\n=== Prompt Building Summary ===")
        total = 0
        for task_type, count in results.items():
            print(f"{task_type:40s}: {count:4d} prompts")
            total += count
        print(f"{'Total':40s}: {total:4d} prompts")
    else:
        count = builder.process_task_type(args.task_type)
        print(f"Processed {count} cases for {args.task_type}")


if __name__ == "__main__":
    main()