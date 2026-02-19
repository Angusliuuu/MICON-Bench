import os
import sys
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluators.dimensions import UNIFIED_DIMENSIONS
from evaluators.candidate_evaluator import CandidateAnswerEvaluator, compute_hybrid_story_score
from utils.path_config import load_global_config, build_path_settings, resolve_paths, resolve_data_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("openai package not installed. Story inference evaluation will use default scores.")


# Checkpoint aggregation config for Object Composition task

OBJECT_COMPOSITION_CONFIG = {
    "HARD_CAP": 0.4,
    "AFFECTS": {"H1": ["A"], "H2": ["B"]},
    "checkpoints": {
        "A": ["A_check_1", "A_check_2", "A_check_3"],
        "B": ["B_check_1", "B_check_2", "B_check_3"],
        "C": ["C_check_1", "C_check_2", "C_check_3"],
        "D": ["D_check_1", "D_check_2", "D_check_3", "D_check_4"],
        "G": ["G_check_1", "G_check_2"]
    },
    "hard_checkpoints": {
        "A_check_1": "H1",  # A_check_1 is hard checkpoint H1, affects dimension A only
        "B_check_1": "H2"   # B_check_1 is hard checkpoint H2, affects dimension B only
    }
}


# Checkpoint aggregation config for Spatial Geometric Constraints task

SPATIAL_GEOMETRIC_CONSTRAINTS_CONFIG = {
    "HARD_CAP": 0.4,
    "AFFECTS": {"H1": ["A"], "H2": ["C"]},
    "checkpoints": {
        "A": ["A_check_1", "A_check_2", "A_check_3"],  # Instruction Following
        "B": ["B_check_1", "B_check_2", "B_check_3"],  # Identity / Fidelity
        "C": ["C_check_1", "C_check_2", "C_check_3", "C_check_4"],  # Structure / Geometry
        "D": ["D_check_1", "D_check_2", "D_check_3"],  # Cross-Reference Consistency
        "G": ["G_check_1", "G_check_2", "G_check_3"]   # Overall Usability
    },
    "hard_checkpoints": {
        "A_check_1": "H1",  # A_check_1 is hard checkpoint H1, affects dimension A
        "C_check_1": "H2"   # C_check_1 is hard checkpoint H2, affects dimension C
    }
}


# Checkpoint aggregation config for Attribute Style Decoupling task

ATTRIBUTE_STYLE_DECOUPLING_CONFIG = {
    "HARD_CAP": 0.4,
    "AFFECTS": {"H1": ["A"], "H2": ["B"], "H3": ["C"], "H4": ["D"]},
    "checkpoints": {
        "A": ["A_check_1", "A_check_2", "A_check_3"],  # Instruction Following
        "B": ["B_check_1", "B_check_2", "B_check_3"],  # Identity / Fidelity
        "C": ["C_check_1", "C_check_2", "C_check_3"],  # Structure / Geometry
        "D": ["D_check_1", "D_check_2", "D_check_3"],  # Cross-Reference Consistency
        "G": ["G_check_1", "G_check_2", "G_check_3"]   # Overall Usability
    },
    "hard_checkpoints": {
        "A_check_1": "H1",  # A_check_1 is hard checkpoint H1, subject must come from image A
        "B_check_1": "H2",  # B_check_1 is hard checkpoint H2, style must come from image B (texture/color only)
        "C_check_1": "H3",  # C_check_1 is hard checkpoint H3, background must come from image C
        "D_check_1": "H4"   # D_check_1 is hard checkpoint H4, do not import objects/layout from RB
    }
}


# Checkpoint aggregation config for Local Element Transfer task

LOCAL_ELEMENT_TRANSFER_CONFIG = {
    "HARD_CAP": 0.4,
    "AFFECTS": {"H1": ["A"], "H2": ["B"], "H3": ["C"]},
    "checkpoints": {
        "A": ["A_check_1", "A_check_2", "A_check_3"],  # Instruction Following
        "B": ["B_check_1", "B_check_2", "B_check_3"],  # Identity / Fidelity
        "C": ["C_check_1", "C_check_2", "C_check_3"],  # Structure / Geometry
        "D": ["D_check_1", "D_check_2", "D_check_3"],  # Cross-Reference Consistency
        "G": ["G_check_1", "G_check_2", "G_check_3"]   # Overall Usability
    },
    "hard_checkpoints": {
        "A_check_1": "H1",  # A_check_1 is hard checkpoint H1, all required subjects must exist
        "B_check_2": "H2",  # B_check_2 is hard checkpoint H2, target identity must stay intact
        "C_check_1": "H3"   # C_check_1 is hard checkpoint H3, local element must transfer correctly
    }
}


# Checkpoint aggregation config for Foreground Background Composition task

FOREGROUND_BACKGROUND_COMPOSITION_CONFIG = {
    "HARD_CAP": 0.4,
    "AFFECTS": {"H1": ["B"], "H2": ["D"]},
    "checkpoints": {
        "A": ["A_check_1", "A_check_2", "A_check_3"],  # Instruction Following
        "B": ["B_check_1", "B_check_2", "B_check_3"],  # Identity / Fidelity
        "C": ["C_check_1", "C_check_2", "C_check_3"],  # Structure / Geometry
        "D": ["D_check_1", "D_check_2", "D_check_3"],  # Cross-Reference Consistency
        "G": ["G_check_1", "G_check_2", "G_check_3"]   # Overall Usability
    },
    "hard_checkpoints": {
        "B_check_1": "H1",  # B_check_1 is hard checkpoint H1, foreground identity must be preserved
        "D_check_1": "H2"   # D_check_1 is hard checkpoint H2, background of image B must be preserved
    }
}


# Checkpoint aggregation config for Story Inference task

STORY_INFER_CONFIG = {
    "HARD_CAP": 0.4,
    "AFFECTS": {"H1": ["A"], "H2": ["B"], "H3": ["E"]},
    "checkpoints": {
        "A": ["A_check_1", "A_check_2", "A_check_3"],  # Instruction Following
        "B": ["B_check_1", "B_check_2", "B_check_3"],  # Identity / Fidelity
        "C": ["C_check_1", "C_check_2", "C_check_3"],  # Structure / Geometry
        "D": ["D_check_1", "D_check_2", "D_check_3"],  # Cross-Reference Consistency
        "E": ["E_check_1", "E_check_2", "E_check_3"],  # Causality / Temporal Logic
        "G": ["G_check_1", "G_check_2", "G_check_3"]   # Overall Usability
    },
    "hard_checkpoints": {
        "A_check_1": "H1",  # A_check_1 is hard checkpoint H1, generated image must show the next step
        "B_check_1": "H2",  # B_check_1 is hard checkpoint H2, main character must stay consistent
        "E_check_1": "H3"   # E_check_1 is hard checkpoint H3, next event must be causally valid
    }
}


def compute_dim_score(passes: List[int], hard_fail: bool = False, hard_cap: float = 0.4) -> float:
    """
    Compute dimension score (0-1 scale).
    base = pass rate (0-1).
    If hard_fail=True, the score is capped by hard_cap.
    """
    n = len(passes)
    if n == 0:
        return 0.0
    
    base = sum(passes) / n  # pass rate, 0-1 scale
    
    if hard_fail:
        return min(base, hard_cap)
    
    return base


def aggregate_checkpoints_to_scores(checkpoints: Dict[str, int], task_type: str) -> Dict[str, float]:
    """
    Aggregate checkpoints to dimension scores (0/1 -> 0-1).
    
    Args:
        checkpoints: checkpoint results, e.g. {"A1": 1, "A2": 0, "A3": 1, ...}
        task_type: task type
    
    Returns:
        dimension scores (0-1), e.g. {"A": 0.8, "B": 0.67, ...}
    """
    # Load task config
    if task_type == "object_composition":
        config = OBJECT_COMPOSITION_CONFIG
    elif task_type == "spatial_geometric_constraints":
        config = SPATIAL_GEOMETRIC_CONSTRAINTS_CONFIG
    elif task_type == "attribute_style_decoupling":
        config = ATTRIBUTE_STYLE_DECOUPLING_CONFIG
    elif task_type == "local_element_transfer":
        config = LOCAL_ELEMENT_TRANSFER_CONFIG
    elif task_type == "foreground_background_composition":
        config = FOREGROUND_BACKGROUND_COMPOSITION_CONFIG
    elif task_type == "story_infer":
        config = STORY_INFER_CONFIG
    else:
        logger.warning(f"Task type {task_type} is not supported for checkpoint aggregation")
        return {}
    
    dimension_scores = {}
    
    # Check hard constraints
    hard_failures = {}
    for checkpoint, hard_id in config["hard_checkpoints"].items():
        if checkpoints.get(checkpoint, 0) == 0:
            # Hard checkpoint failed, affects related dimensions
            affected_dims = config["AFFECTS"].get(hard_id, [])
            for dim in affected_dims:
                hard_failures[dim] = True
    
    # Compute dimension scores
    for dim_code, checkpoint_list in config["checkpoints"].items():
        passes = [checkpoints.get(cp, 0) for cp in checkpoint_list]
        hard_fail = hard_failures.get(dim_code, False)
        dimension_scores[dim_code] = compute_dim_score(
            passes, 
            hard_fail=hard_fail, 
            hard_cap=config["HARD_CAP"]
        )
    
    return dimension_scores


class ScoreAggregator:
    """Score aggregator"""
    
    def __init__(self, project_root: str = ".",
                 use_candidate_eval: bool = True, mllm_client=None, judge_config_path: str = None,
                 timestamp_dir: Optional[str] = None, model_name: Optional[str] = None,
                 output_base_dir: Optional[str] = None, global_config: str = "configs/global.yaml"):
        self.project_root = Path(project_root)
        self.global_config = load_global_config(self.project_root, global_config)
        self.path_settings = build_path_settings(self.global_config)
        self.paths = resolve_paths(self.project_root, self.path_settings)
        self.data_root = self.paths["data_root"]
        self.generations_root = self.paths["generations_root"]
        
        # New directory structure: outputs/eval_res/ModelName/ModelName_TIMESTAMP/
        if output_base_dir and model_name and timestamp_dir:
            base_path = self.project_root / output_base_dir / model_name / timestamp_dir
            self.judgments_root = base_path / "judgments"
            self.reports_root = base_path / "reports"
        elif timestamp_dir:
            # Backward compatible legacy structure
            self.judgments_root = self.paths["outputs_root"] / timestamp_dir / "judgments"
            self.reports_root = self.paths["outputs_root"] / timestamp_dir / "reports"
        else:
            self.judgments_root = self.paths["outputs_root"] / "judgments"
            self.reports_root = self.paths["reports_root"]
        
        self.reports_root.mkdir(parents=True, exist_ok=True)
        self.use_candidate_eval = use_candidate_eval
        
        # Initialize MLLM client
        if mllm_client is None and use_candidate_eval:
            self.mllm_client = self._init_mllm_client(judge_config_path)
        else:
            self.mllm_client = mllm_client
        
        # Initialize candidate answer evaluator (story_infer)
        self.candidate_evaluator = None
        if use_candidate_eval:
            template_path = self.data_root / "story_infer" / "simple_candidate_template.json"
            if template_path.exists():
                try:
                    self.candidate_evaluator = CandidateAnswerEvaluator(
                        template_path=template_path,
                        mllm_client=self.mllm_client
                    )
                    logger.info("Candidate answer evaluator initialized for story_infer")
                except Exception as e:
                    logger.warning(f"Failed to initialize candidate evaluator: {e}")
                    self.candidate_evaluator = None
            else:
                logger.warning(f"Candidate template not found: {template_path}")
    
    def _init_mllm_client(self, judge_config_path: Optional[str] = None):
 
        if OpenAI is None:
            logger.warning("OpenAI package not installed, MLLM client unavailable")
            return None
        
        if judge_config_path is None:
            judge_config_path = self.project_root / "configs" / "judge_models.yaml"
        else:
            judge_config_path = Path(judge_config_path)
        
        if not judge_config_path.exists():
            logger.warning(f"Judge config not found: {judge_config_path}")
            return None
        
        try:
            with open(judge_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            routing = config.get('routing', {})
            per_task = routing.get('per_task', {})
            judge_name = per_task.get('story_infer') or routing.get('default_judge')
            
            if not judge_name:
                logger.warning("No judge configured for story_infer task")
                return None
            
            judges = config.get('judges', [])
            judge_config = None
            for judge in judges:
                if judge.get('name') == judge_name:
                    judge_config = judge
                    break
            
            if not judge_config:
                logger.warning(f"Judge config not found for: {judge_name}")
                return None
            
            endpoint = judge_config.get('endpoint', 'http://localhost:8000/v1')
            api_key_env = judge_config.get('api_key_env', '')
            api_key = os.getenv(api_key_env, 'dummy') if api_key_env else 'dummy'
            model = judge_config.get('model', 'default')
            
            client = OpenAI(base_url=endpoint, api_key=api_key)
            
            client.model_name = model
            client.temperature = judge_config.get('temperature', 0.1)
            client.max_tokens = judge_config.get('max_tokens', 800)
            
            logger.info(f"MLLM client initialized: {judge_name} @ {endpoint}")
            return client
        
        except Exception as e:
            logger.error(f"Failed to initialize MLLM client: {e}")
            return None
    
    def collect_judgments(self, model_name: str, task_type: Optional[str] = None) -> List[Dict]:
        judgments = []
        
        task_types = [task_type] if task_type else [
            "object_composition",
            "spatial_geometric_constraints",
            "local_element_transfer", "attribute_style_decoupling",
            "foreground_background_composition",
            "story_infer"
        ]
        
        for tt in task_types:
            task_dir = self.judgments_root / tt
            if not task_dir.exists():
                continue
            
            for case_dir in sorted(task_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                
                judgment_file = case_dir / f"{model_name}.json"
                if judgment_file.exists():
                    try:
                        with open(judgment_file, 'r', encoding='utf-8') as f:
                            judgment = json.load(f)
                            judgment['task_type'] = tt
                            judgment['case_id'] = case_dir.name
                            
                            judgment = self._process_checkpoints(judgment, tt, model_name)
                            judgments.append(judgment)
                    except Exception as e:
                        logger.error(f"Failed to read judgment file {judgment_file}: {e}")
        
        return judgments
    
    def _get_generated_image_path(self, model_name: str, case_id: str, task_type: str) -> Optional[Path]:
        """Get generated image path"""
        # Format: outputs/generations/<model>/<task>/<case_id>.png
        gen_dir = self.generations_root / model_name / task_type
        
        for ext in [".png", ".jpg", ".jpeg"]:
            gen_path = gen_dir / f"{case_id}{ext}"
            if gen_path.exists():
                return gen_path
        
        return None
    
    def _get_reference_images(self, case_id: str, task_type: str) -> List[Path]:
        """Get reference image paths"""
        case_yaml = self.data_root / task_type / "cases" / f"{case_id}.yaml"
        
        if not case_yaml.exists():
            return []
        
        try:
            with open(case_yaml, 'r', encoding='utf-8') as f:
                case_data = yaml.safe_load(f)
                ref_images = []
                
                refs = case_data.get('refs')
                if not isinstance(refs, dict):
                    return []

                for key in ['RA', 'RB', 'RC']:
                    if key in refs:
                        ref_path = resolve_data_path(refs[key], self.data_root)
                        if ref_path.exists():
                            ref_images.append(ref_path)
                
                return ref_images
        except Exception as e:
            logger.warning(f"Failed to read reference images from {case_yaml}: {e}")
            return []
    
    def _compute_candidate_score(self, model_name: str, case_id: str, task_type: str) -> Dict[str, any]:
        """Compute candidate-answer-based score"""
        if not self.use_candidate_eval or self.candidate_evaluator is None:
            return {}
        
        # Get generated and reference images
        gen_path = self._get_generated_image_path(model_name, case_id, task_type)
        ref_images = self._get_reference_images(case_id, task_type)
        
        if gen_path is None or not ref_images:
            logger.warning(f"Cannot find images for candidate evaluation: {case_id}")
            return {}
        
        try:
            # Run candidate answer evaluator
            result = self.candidate_evaluator.evaluate_case(
                case_id=case_id,
                generated_image_path=gen_path,
                reference_images=ref_images
            )
            self._save_candidate_evaluation_details(model_name, case_id, task_type, result, gen_path, ref_images)
            
            return {
                'candidate_score': result.get('score', 0.0),
                'candidate_reasoning': result.get('reasoning', ''),
                'matches_positive': result.get('matches_positive', False),
                'matches_negative': result.get('matches_negative', False)
            }
        except Exception as e:
            logger.error(f"Failed to compute candidate score for {case_id}: {e}")
            return {}
    
    def _save_candidate_evaluation_details(self, model_name: str, case_id: str, task_type: str, 
                                          result: Dict, gen_path: Path, ref_images: List[Path]):
        """Save candidate evaluation details to a standalone JSON file"""
        try:
            # Create candidate evaluation details directory
            details_dir = self.judgments_root / task_type / case_id
            details_dir.mkdir(parents=True, exist_ok=True)
            
            details_file = details_dir / f"{model_name}_candidate_eval_details.json"
            
            # Load candidate template info
            candidate_data = self.candidate_evaluator.candidates_data.get(case_id, {})
            
            details = {
                "case_id": case_id,
                "task_type": task_type,
                "model_name": model_name,
                "generated_image": str(gen_path),
                "reference_images": [str(img) for img in ref_images],
                "background_summary": candidate_data.get('background_summary', ''),
                "positive_candidates": candidate_data.get('positive_candidates', []),
                "negative_candidates": candidate_data.get('negative_candidates', []),
                "evaluation_result": {
                    "score": result.get('score', 0.0),
                    "reasoning": result.get('reasoning', ''),
                    "matches_positive": result.get('matches_positive', False),
                    "matches_negative": result.get('matches_negative', False),
                    "error": result.get('error', None),
                    "parse_error": result.get('parse_error', None)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            with open(details_file, 'w', encoding='utf-8') as f:
                json.dump(details, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved candidate evaluation details to: {details_file}")
        
        except Exception as e:
            logger.error(f"Failed to save candidate evaluation details for {case_id}: {e}")
    
    def _extract_checkpoints_from_top_level(self, judgment: Dict) -> Dict:
        """
        Extract checkpoint data from the top-level judgment dict.
        This handles cases where the MLLM omits the checkpoint_results wrapper.
        
        Checkpoint format: A_check_1, A_check_2, B_check_1, etc.
        """
        import re
        checkpoint_pattern = re.compile(r'^[A-Z]_check_\d+$')
        
        extracted = {}
        for key, value in judgment.items():
            # Skip known non-checkpoint fields
            if key in ['case_id', 'task_type', 'model_name', 'checkpoint_results', 'checkpoints', 
                      'hard_constraint_results', 'rationale_short', 'flags', 'scores', 'overall',
                      'candidate_score', 'candidate_reasoning', 'matches_positive', 'matches_negative',
                      'traditional_mllm_score', 'score_calculation_details', 'timestamp']:
                continue
            
            # Check checkpoint pattern
            if checkpoint_pattern.match(key):
                extracted[key] = value
        
        return extracted if extracted else {}
    
    def _process_checkpoints(self, judgment: Dict, task_type: str, model_name: str = None) -> Dict:
        # Support three checkpoint formats:
        # 1. checkpoint_results (nested)
        # 2. checkpoints (legacy key name, nested)
        # 3. top-level checkpoints (when checkpoint_results wrapper is missing)
        checkpoint_results = judgment.get('checkpoint_results', judgment.get('checkpoints', {}))
        
        # If no nested checkpoints, try extracting from top-level
        if not checkpoint_results:
            # Check top-level checkpoint keys (e.g., A_check_1, B_check_1)
            checkpoint_results = self._extract_checkpoints_from_top_level(judgment)
            
            if checkpoint_results:
                logger.info(f"Case {judgment.get('case_id')}: Found checkpoints at top level (MLLM output without 'checkpoint_results' wrapper)")
            else:
                logger.warning(f"Case {judgment.get('case_id')}: no checkpoint data")
                return judgment
        
        # Flatten checkpoint_results
        # From {"A_check_1": {"pass": 1}} to {"A_check_1": 1}
        checkpoints = {}
        for key, value in checkpoint_results.items():
            if isinstance(value, dict) and 'pass' in value:
                checkpoints[key] = value['pass']
            else:
                checkpoints[key] = value
        
        # Step 1: aggregate checkpoints to dimension scores (0/1 -> 0-1)
        dimension_scores = aggregate_checkpoints_to_scores(checkpoints, task_type)
        
        if not dimension_scores:
            return judgment
        
        # Step 2: update dimension scores
        judgment['scores'] = dimension_scores
        
        # Step 3: compute MLLM overall score (0-1 -> 0-100)
        # overall = (Σ score_d / n) * 100 (simple average, no weights)
        avg_score = sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0.0
        mllm_overall = avg_score * 100.0
        
        # Step 4a: for story_infer, compute hybrid score using candidate answers
        if task_type == "story_infer" and self.use_candidate_eval and model_name:
            case_id = judgment.get('case_id')
            candidate_metrics = self._compute_candidate_score(model_name, case_id, task_type)
            
            if candidate_metrics and 'candidate_score' in candidate_metrics:
                # Store candidate score fields
                judgment['candidate_score'] = candidate_metrics['candidate_score']
                judgment['candidate_reasoning'] = candidate_metrics.get('candidate_reasoning', '')
                judgment['matches_positive'] = candidate_metrics.get('matches_positive', False)
                judgment['matches_negative'] = candidate_metrics.get('matches_negative', False)
                
                # Compute average dimension score (0-1 scale)
                avg_dimension_score = sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0.0
                
                # Hybrid overall: traditional MLLM (40%) + candidate answer (60%)
                # Traditional MLLM: 0-1 -> 0-40
                # Candidate answer: 0-10 -> 0-60
                hybrid_overall = compute_hybrid_story_score(
                    traditional_mllm_score=avg_dimension_score,
                    candidate_score=candidate_metrics['candidate_score'],
                    trad_weight=0.4,
                    cand_weight=0.6
                )
                
                judgment['traditional_mllm_score'] = round(avg_dimension_score, 2)
                judgment['overall'] = hybrid_overall
                
                # Save detailed score calculation
                judgment['score_calculation_details'] = self._create_score_calculation_details(
                    task_type=task_type,
                    checkpoints=checkpoints,
                    dimension_scores=dimension_scores,
                    mllm_overall=mllm_overall,
                    avg_dimension_score=avg_dimension_score,
                    candidate_score=candidate_metrics['candidate_score'],
                    hybrid_overall=hybrid_overall
                )
                
                logger.info(f"Case {case_id}: Traditional_MLLM={avg_dimension_score:.2f}/1.0, "
                           f"Candidate={candidate_metrics['candidate_score']:.2f}/10.0, "
                           f"Hybrid={hybrid_overall:.2f}/100.0")
            else:
                judgment['overall'] = round(mllm_overall, 2)
                # Save non-hybrid score calculation
                judgment['score_calculation_details'] = self._create_score_calculation_details(
                    task_type=task_type,
                    checkpoints=checkpoints,
                    dimension_scores=dimension_scores,
                    mllm_overall=mllm_overall
                )
                logger.warning(f"Case {case_id}: Using traditional MLLM score only")
        
        else:
            judgment['overall'] = round(mllm_overall, 2)
            # Save score calculation for other tasks
            if dimension_scores:
                judgment['score_calculation_details'] = self._create_score_calculation_details(
                    task_type=task_type,
                    checkpoints=checkpoints,
                    dimension_scores=dimension_scores,
                    mllm_overall=mllm_overall
                )
        
        return judgment
    
    def _create_score_calculation_details(self, task_type: str, checkpoints: Dict = None,
                                         dimension_scores: Dict = None,
                                         mllm_overall: float = None, avg_dimension_score: float = None,
                                         candidate_score: float = None, hybrid_overall: float = None) -> Dict:
        """Create detailed score calculation record"""
        details = {
            "task_type": task_type,
            "scoring_method": "traditional_mllm",
            "calculation_steps": []
        }
        
        # Step 1: checkpoint aggregation
        if checkpoints:
            checkpoint_step = {
                "step": 1,
                "description": "Checkpoint aggregation (0/1 -> dimension scores 0-1)",
                "checkpoints": checkpoints,
                "checkpoint_count": {
                    "total": len(checkpoints),
                    "passed": sum(1 for v in checkpoints.values() if v == 1),
                    "failed": sum(1 for v in checkpoints.values() if v == 0)
                }
            }
            details["calculation_steps"].append(checkpoint_step)
        
        # Step 2: dimension score calculation
        if dimension_scores:
            dimension_step = {
                "step": 2,
                "description": "Dimension scores (0-1 scale)",
                "dimension_scores": dimension_scores,
                "dimension_count": len(dimension_scores)
            }
            
            # Add per-dimension details
            if task_type in ["object_composition", "spatial_geometric_constraints",
                           "attribute_style_decoupling", "local_element_transfer",
                           "foreground_background_composition", "story_infer"]:
                dimension_details = []
                for dim_code, score in dimension_scores.items():
                    dim_info = {
                        "dimension": dim_code,
                        "dimension_name": UNIFIED_DIMENSIONS.get(dim_code, {}).name if dim_code in UNIFIED_DIMENSIONS else "Unknown",
                        "score": score,
                        "max_score": 5.0
                    }
                    dimension_details.append(dim_info)
                dimension_step["dimension_details"] = dimension_details
            
            details["calculation_steps"].append(dimension_step)
        
        # Step 3: average MLLM score (no weights)
        if dimension_scores and mllm_overall is not None:
            avg_step = {
                "step": 3,
                "description": "Average MLLM score calculation (0-100 scale)",
                "dimension_scores": dimension_scores,
                "scoring_components": []
            }
            
            score_sum = 0.0
            for dim, score in dimension_scores.items():
                score_sum += score
                avg_step["scoring_components"].append({
                    "dimension": dim,
                    "score": score
                })
            
            avg_score = score_sum / len(dimension_scores) if dimension_scores else 0.0
            avg_step["average_score"] = round(avg_score, 4)
            avg_step["mllm_score_0_100"] = round(mllm_overall, 2)
            avg_step["formula"] = "(Σ score_d / n) * 100 (simple average, scores are 0-1)"
            avg_step["note"] = "No weights; all dimensions are equally weighted"
            
            details["calculation_steps"].append(avg_step)
        
        # Step 4: story_infer hybrid scoring
        if task_type == "story_infer" and candidate_score is not None:
            details["scoring_method"] = "hybrid"
            
            # Traditional MLLM component
            if avg_dimension_score is not None:
                traditional_step = {
                    "step": 4,
                    "description": "Traditional MLLM component (average of dimension scores)",
                    "avg_dimension_score_0_1": round(avg_dimension_score, 4),
                    "normalized_to_40": round(avg_dimension_score * 40.0, 2),
                    "weight": 0.4,
                    "formula": "avg_dimension_score * 40 (0-1 scale)"
                }
                details["calculation_steps"].append(traditional_step)
            
            # Candidate answer component
            candidate_step = {
                "step": 5,
                "description": "Candidate answer component",
                "candidate_score_0_10": candidate_score,
                "normalized_to_60": round((candidate_score / 10.0) * 60.0, 2),
                "weight": 0.6,
                "formula": "(candidate_score / 10.0) * 60"
            }
            details["calculation_steps"].append(candidate_step)
            
            # Final hybrid score
            if hybrid_overall is not None:
                hybrid_step = {
                    "step": 6,
                    "description": "Final hybrid score",
                    "traditional_component": round(avg_dimension_score * 40.0, 2),
                    "candidate_component": round((candidate_score / 10.0) * 60.0, 2),
                    "final_score": hybrid_overall,
                    "formula": "traditional_component (0-1 * 40) + candidate_component (0-10/10 * 60)"
                }
                details["calculation_steps"].append(hybrid_step)
                details["final_score"] = hybrid_overall
        else:
            details["final_score"] = mllm_overall
        
        return details
    
    def aggregate_by_task(self, judgments: List[Dict]) -> Dict[str, Dict]:
        task_stats = defaultdict(lambda: {
            "cases": [],
            "total_count": 0,
            "avg_overall": 0.0,
            "avg_scores": {},
            "min_overall": 100.0,
            "max_overall": 0.0
        })
        
        for judgment in judgments:
            tt = judgment.get('task_type', 'unknown')
            overall = judgment.get('overall', 0.0)
            scores = judgment.get('scores', {})
            
            task_stats[tt]["cases"].append(judgment)
            task_stats[tt]["total_count"] += 1
            
            if overall < task_stats[tt]["min_overall"]:
                task_stats[tt]["min_overall"] = overall
            if overall > task_stats[tt]["max_overall"]:
                task_stats[tt]["max_overall"] = overall
            
            for dim, score in scores.items():
                if dim not in task_stats[tt]["avg_scores"]:
                    task_stats[tt]["avg_scores"][dim] = []
                task_stats[tt]["avg_scores"][dim].append(score)
        
        # Compute averages
        for tt, stats in task_stats.items():
            if stats["total_count"] > 0:
                stats["avg_overall"] = sum(c.get('overall', 0) for c in stats["cases"]) / stats["total_count"]
                for dim, scores_list in stats["avg_scores"].items():
                    stats["avg_scores"][dim] = sum(scores_list) / len(scores_list)
        
        return dict(task_stats)
    
    def generate_csv(self, judgments: List[Dict], model_name: str) -> str:
        csv_file = self.reports_root / f"{model_name}_summary.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = ["task_type", "case_id", "overall"]
            for dim_code in ["A", "B", "C", "D", "E", "F", "G"]:
                header.append(f"score_{dim_code}")
            header.extend([
                "traditional_mllm_score", "candidate_score", "candidate_reasoning",
                "matches_positive", "matches_negative",
                "rationale", "flags"
            ])
            writer.writerow(header)
            
            # Rows
            for judgment in judgments:
                row = [
                    judgment.get('task_type', ''),
                    judgment.get('case_id', ''),
                    judgment.get('overall', 0.0)
                ]
                
                scores = judgment.get('scores', {})
                for dim_code in ["A", "B", "C", "D", "E", "F", "G"]:
                    row.append(scores.get(dim_code, 0.0))
                
                # Add story_infer candidate score columns
                row.append(judgment.get('traditional_mllm_score', ''))
                row.append(judgment.get('candidate_score', ''))
                row.append(judgment.get('candidate_reasoning', ''))
                row.append(judgment.get('matches_positive', ''))
                row.append(judgment.get('matches_negative', ''))
                
                row.append(judgment.get('rationale_short', ''))
                row.append(';'.join(judgment.get('flags', [])))
                
                writer.writerow(row)
        
        logger.info(f"CSV report generated: {csv_file}")
        return str(csv_file)
    
    def generate_summary_report(self, judgments: List[Dict], task_stats: Dict, model_name: str) -> str:
        """Generate text summary report"""
        report_file = self.reports_root / f"{model_name}_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"MICON-Bench Evaluation Report\n")
            f.write(f"Model: {model_name}\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall stats
            total_cases = len(judgments)
            overall_avg = sum(j.get('overall', 0) for j in judgments) / total_cases if total_cases > 0 else 0
            
            f.write(f"Overall Statistics:\n")
            f.write(f"  Total Cases Evaluated: {total_cases}\n")
            f.write(f"  Average Overall Score: {overall_avg:.2f}/100\n\n")
            
            # Per-task stats
            f.write("Performance by Task Type:\n")
            f.write("-" * 80 + "\n")
            
            for tt in sorted(task_stats.keys()):
                stats = task_stats[tt]
                f.write(f"\n{tt}:\n")
                f.write(f"  Cases: {stats['total_count']}\n")
                f.write(f"  Avg Overall: {stats['avg_overall']:.2f}/100\n")
                f.write(f"  Range: [{stats['min_overall']:.2f}, {stats['max_overall']:.2f}]\n")
                f.write(f"  Dimension Scores:\n")
                for dim_code in ["A", "B", "C", "D", "E", "F", "G"]:
                    if dim_code in stats['avg_scores']:
                        dim_name = UNIFIED_DIMENSIONS[dim_code].name
                        f.write(f"    {dim_code} ({dim_name}): {stats['avg_scores'][dim_code]:.2f}/1\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Text report generated: {report_file}")
        return str(report_file)
    
    def run(self, model_name: str, task_type: Optional[str] = None):
        """Run full aggregation pipeline"""
        logger.info(f"Start aggregating scores for model {model_name}")
        
        # Collect judgments
        judgments = self.collect_judgments(model_name, task_type)
        
        if not judgments:
            logger.warning(f"No judgments found for model {model_name}")
            return
        
        logger.info(f"Collected {len(judgments)} judgments")
        
        # Aggregate by task
        task_stats = self.aggregate_by_task(judgments)
        
        # Generate reports
        csv_file = self.generate_csv(judgments, model_name)
        report_file = self.generate_summary_report(judgments, task_stats, model_name)
        
        logger.info("Aggregation complete!")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  Report: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation scores")
    parser.add_argument("--project-root", type=str, default=".", help="Project root")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--task-type", type=str, default=None, help="Specific task type (optional)")
    parser.add_argument("--judge-config", type=str, default=None,
                       help="Path to judge_models.yaml (default: configs/judge_models.yaml)")
    parser.add_argument("--global-config", type=str, default="configs/global.yaml",
                       help="Path to global.yaml (default: configs/global.yaml)")
    parser.add_argument("--timestamp-dir", type=str, default=None,
                       help="Timestamp directory to read judgments from (default: search for latest)")
    parser.add_argument("--output-base-dir", type=str, default=None,
                       help="Base output directory (e.g., 'outputs/eval_res')")
    
    args = parser.parse_args()
    project_root = Path(args.project_root)
    global_config = load_global_config(project_root, args.global_config)
    path_settings = build_path_settings(global_config)
    paths = resolve_paths(project_root, path_settings)
    
    # If timestamp_dir is not provided, try to find the latest one
    timestamp_dir = args.timestamp_dir
    if not timestamp_dir:
        # Search new directory structure first
        if args.output_base_dir:
            search_dir = Path(args.project_root) / args.output_base_dir / args.model_name
        else:
            search_dir = paths["outputs_root"]
        
        if search_dir.exists():
            # Find timestamp directories for the model name
            model_dirs = [d for d in search_dir.iterdir() 
                         if d.is_dir() and d.name.startswith(args.model_name)]
            if model_dirs:
                # Sort by modification time and take the latest
                timestamp_dir = sorted(model_dirs, key=lambda x: x.stat().st_mtime)[-1].name
                logger.info(f"Auto-detected latest timestamp directory: {timestamp_dir}")
    
    aggregator = ScoreAggregator(
        project_root=args.project_root,
        use_candidate_eval=True,
        mllm_client=None,  # auto-initialize from config
        judge_config_path=args.judge_config,
        timestamp_dir=timestamp_dir,
        model_name=args.model_name,
        output_base_dir=args.output_base_dir,
        global_config=args.global_config
    )
    aggregator.run(args.model_name, args.task_type)


if __name__ == "__main__":
    main()
