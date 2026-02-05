import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import base64
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


class CandidateAnswerEvaluator:

    def __init__(self, template_path: Union[str, Path], mllm_client=None):

        self.template_path = Path(template_path)
        self.mllm_client = mllm_client
        self.candidates_data = self._load_template()
        
        logger.info(f"CandidateAnswerEvaluator initialized with {len(self.candidates_data)} cases")
    
    def _load_template(self) -> Dict[str, Dict]:
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            candidates_dict = {}
            for case in data.get('cases', []):
                case_id = case.get('case_id')
                if case_id:
                    candidates_dict[case_id] = case
            
            return candidates_dict
        
        except Exception as e:
            logger.error(f"Failed to load candidate template: {e}")
            return {}
    
    def _encode_image_to_base64(self, image_path: Path) -> str:
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                return img_str
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise
    
    def _build_evaluation_prompt(self, case_id: str, generated_image_path: Path, 
                                 reference_images: List[Path]) -> str:

        candidate_data = self.candidates_data.get(case_id)
        if not candidate_data:
            raise ValueError(f"No candidate data found for case {case_id}")
        
        background = candidate_data.get('background_summary', '')
        positive_candidates = candidate_data.get('positive_candidates', [])
        negative_candidates = candidate_data.get('negative_candidates', [])
        
        prompt = f"""You are a professional image evaluator. You will be shown a sequence of images:

IMAGE ORGANIZATION:
- First {len(reference_images)} images: REFERENCE IMAGES showing the story sequence
- Last image: GENERATED IMAGE that should depict the next step in the story

IMPORTANT: Pay special attention to the differences between the LAST REFERENCE IMAGE and the GENERATED IMAGE. The generated image should show a clear progression from the last reference image, not just repeat it.

Background: {background}

Your task is to STRICTLY evaluate the generated image against the following HUMAN-ANNOTATED ANSWER SET:

POSITIVE EXAMPLES (Correct next steps - what the generated image MUST match):
"""
        for i, pos in enumerate(positive_candidates, 1):
            prompt += f"{i}. {pos}\n"
        
        prompt += f"""
NEGATIVE EXAMPLES (Incorrect next steps - what the generated image MUST avoid):
"""
        for i, neg in enumerate(negative_candidates, 1):
            prompt += f"{i}. {neg}\n"
        
        prompt += """
EVALUATION CRITERIA (STRICTLY follow the human-annotated answer set):

1. OBSERVE CAREFULLY:
   - Compare the LAST reference image with the GENERATED image
   - Identify what has CHANGED (this shows story progression)
   - Identify what remains CONSISTENT (character identity, scene coherence)

2. SCORING RULES:
   - Score 8-10: The generated image CLEARLY matches one or more positive examples AND avoids all negative examples. Shows obvious progression from the last reference image.
   - Score 5-7: The generated image PARTIALLY matches positive examples but has minor deviations OR shows some elements from negative examples.
   - Score 0-4: The generated image matches negative examples OR fails to show progression from the last reference image OR contradicts the positive examples.

3. KEY REQUIREMENT:
   - The evaluation MUST be based on the human-annotated answer set above
   - Do NOT accept images that merely repeat the last reference frame
   - Do NOT accept images that violate the positive examples
   - Do NOT accept images that exhibit characteristics from negative examples

CRITICAL: Your response MUST be valid JSON only. Return ONLY the raw JSON object with this exact format:
{
    "score": <number from 0-10>,
    "reasoning": "<brief explanation comparing the generated image to the positive/negative examples and noting changes from the last reference image>",
    "matches_positive": <true/false>,
    "matches_negative": <true/false>
}
"""
        return prompt
    
    def evaluate_case(self, case_id: str, generated_image_path: Union[str, Path],
                     reference_images: List[Union[str, Path]]) -> Dict[str, any]:

        generated_image_path = Path(generated_image_path)
        reference_images = [Path(img) for img in reference_images]
        
        if not generated_image_path.exists():
            logger.error(f"Generated image not found: {generated_image_path}")
            return {
                'score': 0.0,
                'reasoning': 'Generated image not found',
                'matches_positive': False,
                'matches_negative': False,
                'error': 'Image not found'
            }
        
        try:
            prompt = self._build_evaluation_prompt(case_id, generated_image_path, reference_images)

            if self.mllm_client is None:
                logger.warning("No MLLM client provided, returning default score")
                return {
                    'score': 5.0,
                    'reasoning': 'No MLLM client available for evaluation',
                    'matches_positive': False,
                    'matches_negative': False
                }
            
            images = reference_images + [generated_image_path]
            response = self._call_mllm(prompt, images)
            
            result = self._parse_response(response)
            
            return result
        
        except Exception as e:
            logger.error(f"Error evaluating case {case_id}: {e}")
            return {
                'score': 0.0,
                'reasoning': f'Evaluation error: {str(e)}',
                'matches_positive': False,
                'matches_negative': False,
                'error': str(e)
            }
    
    def _call_mllm(self, prompt: str, images: List[Path]) -> str:

        try:
            content = [{"type": "text", "text": prompt}]
            
            for img_path in images:
                img_base64 = self._encode_image_to_base64(img_path)
                if img_base64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    })
            
            model_name = getattr(self.mllm_client, 'model_name', 'default')
            temperature = getattr(self.mllm_client, 'temperature', 0.1)
            max_tokens = getattr(self.mllm_client, 'max_tokens', 800)
            
            response = self.mllm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional image evaluation expert."},
                    {"role": "user", "content": content}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"MLLM call failed: {e}")
            raise
    
    def _parse_response(self, response: str) -> Dict[str, any]:

        try:
            response = response.strip()
            
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                response = response[start:end].strip()
            
            result = json.loads(response)
            
            score = float(result.get('score', 0))
            score = max(0.0, min(10.0, score))
            
            return {
                'score': score,
                'reasoning': result.get('reasoning', ''),
                'matches_positive': result.get('matches_positive', False),
                'matches_negative': result.get('matches_negative', False)
            }
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MLLM response as JSON: {e}")
            logger.error(f"Response: {response}")
            
            import re
            score_match = re.search(r'score["\s:]+(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(10.0, score))
                return {
                    'score': score,
                    'reasoning': 'Extracted from non-JSON response',
                    'matches_positive': False,
                    'matches_negative': False
                }
            
            return {
                'score': 0.0,
                'reasoning': 'Failed to parse MLLM response',
                'matches_positive': False,
                'matches_negative': False,
                'parse_error': str(e)
            }
        
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            return {
                'score': 0.0,
                'reasoning': f'Parse error: {str(e)}',
                'matches_positive': False,
                'matches_negative': False,
                'error': str(e)
            }


def compute_hybrid_story_score(traditional_mllm_score: float, candidate_score: float,
                               trad_weight: float = 0.4, cand_weight: float = 0.6) -> float:
    """
    Compute hybrid score for story_infer task.
    Args:
        traditional_mllm_score: average dimension score from MLLM (0-1 scale)
        candidate_score: candidate-answer match score (0-10 scale)
        trad_weight: weight for MLLM score (default 0.4, i.e., 40%)
        cand_weight: weight for candidate score (default 0.6, i.e., 60%)
    Returns:
        hybrid overall score (0-100 scale)
    """
    # Traditional MLLM score: 0-1 -> 0-40
    trad_score_100 = traditional_mllm_score * 40.0
    
    # Candidate answer score: 0-10 -> 0-60
    cand_score_100 = (candidate_score / 10.0) * 60.0
    
    hybrid_score = trad_score_100 + cand_score_100
    
    return round(hybrid_score, 2)

