from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Dimension:
    """Scoring dimension definition"""
    code: str           
    name: str          
    description: str   
    score_range: tuple = (0, 1)  


# Unified seven dimensions (A-G)
UNIFIED_DIMENSIONS = {
    "A": Dimension(
        code="A",
        name="Instruction Following",
        description="Instruction adherence: how well the output follows the original prompt and task requirements"
    ),
    "B": Dimension(
        code="B",
        name="Identity/Fidelity",
        description="Identity/attribute fidelity: key objects and subjects match the references"
    ),
    "C": Dimension(
        code="C",
        name="Structure/Geometry",
        description="Geometry/structure: spatial relations, proportions, and perspective are plausible"
    ),
    "D": Dimension(
        code="D",
        name="Cross-Ref Consistency",
        description="Cross-reference consistency: information across references is coherent"
    ),
    "E": Dimension(
        code="E",
        name="Causality/Temporal",
        description="Causality/temporal: logical ordering and causal coherence in story progression"
    ),
    "F": Dimension(
        code="F",
        name="Text Grounding",
        description="Text grounding: text content in the image is accurate and clear"
    ),
    "G": Dimension(
        code="G",
        name="Overall Usability",
        description="Overall usability: aesthetics, usefulness, and visual harmony"
    )
}

DEFAULT_PARTICIPATING_DIMENSIONS = ["A", "B", "C", "D", "G"]


def get_dimension(code: str) -> Optional[Dimension]:
    return UNIFIED_DIMENSIONS.get(code.upper())


def get_all_dimensions() -> Dict[str, Dimension]:
    return UNIFIED_DIMENSIONS.copy()


def get_participating_dimensions(task_type: str, case_dimensions: Optional[List[str]] = None) -> List[str]:
    if case_dimensions:
        return [d.upper() for d in case_dimensions]
    # Return global default
    return DEFAULT_PARTICIPATING_DIMENSIONS.copy()


def compute_overall_score(dimension_scores: Dict[str, float]) -> float:
    """
    Compute overall score (simple average).
    Formula: overall = (Σ s_d / n) * 100 = average dimension score * 100
    Where s_d is a dimension score (0-1) and n is the number of dimensions.
    """
    if not dimension_scores:
        return 0.0
    
    # Average dimension scores (0-1)
    avg_score = sum(dimension_scores.values()) / len(dimension_scores)
    # Convert to 0-100 scale
    overall = avg_score * 100.0
    return round(overall, 2)


def format_dimension_prompt_block() -> str:
    lines = ["[Unified Scoring Dimensions]"]
    for code in ["A", "B", "C", "D", "E", "F", "G"]:
        dim = UNIFIED_DIMENSIONS[code]
        lines.append(f"{code}. {dim.name:25} ({dim.description})")
    return "\n".join(lines)