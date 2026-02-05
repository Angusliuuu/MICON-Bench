import os
import sys
import json
import base64
import argparse
import yaml
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.path_config import load_global_config, build_path_settings, resolve_paths, resolve_data_path

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JudgeRunner:
    
    def __init__(self, config_path: str = "configs/judge_models.yaml", project_root: str = ".", 
                 timestamp_dir: Optional[str] = None, model_name: Optional[str] = None,
                 output_base_dir: Optional[str] = None, global_config: str = "configs/global.yaml"):
        self.project_root = Path(project_root)
        self.config = self._load_config(config_path)
        self.global_config = load_global_config(self.project_root, global_config)
        self.path_settings = build_path_settings(self.global_config)
        self.paths = resolve_paths(self.project_root, self.path_settings)
        self.prompts_root = self.paths["prompts_root"] / "judge"
        self.generations_root = self.paths["generations_root"]
        
        # New directory structure: outputs/eval_res/ModelName/ModelName_TIMESTAMP/judgments
        if output_base_dir and model_name and timestamp_dir:
            self.outputs_root = self.project_root / output_base_dir / model_name / timestamp_dir / "judgments"
        elif timestamp_dir:
            # Backward compatible legacy structure
            self.outputs_root = self.paths["outputs_root"] / timestamp_dir / "judgments"
        else:
            self.outputs_root = self.paths["outputs_root"] / "judgments"
        
        self.data_root = self.paths["data_root"]
        
        self.clients = {}
        self._init_clients()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load judge model config"""
        full_path = self.project_root / config_path
        if not full_path.exists():
            logger.warning(f"Config file not found: {full_path}, using defaults")
            return self._get_default_config()
        
        with open(full_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict:
        """Default config"""
        return {
            "defaults": {
                "temperature": 0.1,
                "max_tokens": 800,
                "retries": 2,
                "concurrency": 1
            },
            "judges": [
                {
                    "name": "local-qwen",
                    "model": "Qwen2.5-VL-32B-Instruct",
                    "endpoint": "http://127.0.0.1:8000/v1",
                    "temperature": 0.1
                }
            ],
            "routing": {"default_judge": "local-qwen"}
        }
    
    def _init_clients(self):
        """Initialize OpenAI clients"""
        for judge_config in self.config.get("judges", []):
            name = judge_config["name"]
            endpoint = judge_config.get("endpoint", "http://127.0.0.1:8000/v1")
            api_key = os.getenv(judge_config.get("api_key_env", ""), "dummy") or "dummy"
            
            self.clients[name] = OpenAI(base_url=endpoint, api_key=api_key)
            logger.info(f"Initialized judge client: {name} @ {endpoint}")
    
    def get_judge_config(self, judge_name: Optional[str] = None) -> Dict:
        if judge_name is None:
            judge_name = self.config.get("routing", {}).get("default_judge", "local-qwen")
        
        for judge in self.config.get("judges", []):
            if judge["name"] == judge_name:
                return judge
        return self.config["judges"][0] if self.config.get("judges") else {}
    
    def encode_image(self, image_path: str) -> Optional[str]:
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None
    
    def call_judge(self, prompt_text: str, ref_images: List[str], candidate_image: str,
                   judge_name: Optional[str] = None, retry: int = 0) -> Tuple[Optional[Dict], str]:
        """Call judge MLLM, return: (parsed_json, raw_response)"""
        judge_config = self.get_judge_config(judge_name)
        client = self.clients.get(judge_config["name"])
        
        if not client:
            logger.error(f"Judge client not found: {judge_config['name']}")
            return None, ""
        
        # Prepend image order note if missing
        if "Image Order" not in prompt_text and "image order" not in prompt_text.lower():
            image_order_note = f"\n**IMAGE ORDER REMINDER**: You will see {len(ref_images)} reference image(s) FIRST, followed by 1 generated image (LAST). Please evaluate the LAST image.\n\n"
            prompt_text = image_order_note + prompt_text
        
        # Build message content
        content = [{"type": "text", "text": prompt_text}]
        
        # Add reference images (in order)
        for i, ref_path in enumerate(ref_images, 1):
            img_base64 = self.encode_image(ref_path)
            if img_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })
        
        # Add candidate image (generated, last)
        cand_base64 = self.encode_image(candidate_image)
        if cand_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{cand_base64}"}
            })
        
        # Call API
        try:
            defaults = self.config.get("defaults", {})
            temperature = judge_config.get("temperature", defaults.get("temperature", 0.1))
            max_tokens = judge_config.get("max_tokens", defaults.get("max_tokens", 800))
            
            response = client.chat.completions.create(
                model=judge_config["model"],
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            raw_response = response.choices[0].message.content
            parsed_json = self._parse_json_response(raw_response)
            
            if parsed_json is None and retry < defaults.get("retries", 2):
                logger.warning(f"JSON parse failed, retry {retry + 1}")
                time.sleep(1)
                return self.call_judge(prompt_text, ref_images, candidate_image, judge_name, retry + 1)
            
            return parsed_json, raw_response
            
        except Exception as e:
            logger.error(f"Judge API call failed: {e}")
            if retry < self.config.get("defaults", {}).get("retries", 2):
                time.sleep(2)
                return self.call_judge(prompt_text, ref_images, candidate_image, judge_name, retry + 1)
            return None, f"Error: {str(e)}"
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON response"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Extract JSON block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Extract any JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        logger.error(f"Unable to parse JSON: {response[:200]}")
        return None
    
    def judge_case(self, task_type: str, case_id: str, model_name: str,
                   judge_name: Optional[str] = None) -> bool:
        """Judge a single case"""
        prompt_file = self.prompts_root / task_type / f"{case_id}.txt"
        if not prompt_file.exists():
            logger.error(f"Prompt not found: {prompt_file}")
            return False
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
        
        case_file = self.data_root / task_type / "cases" / f"{case_id}.yaml"
        if not case_file.exists():
            logger.error(f"Case not found: {case_file}")
            return False
        
        with open(case_file, 'r', encoding='utf-8') as f:
            case_data = yaml.safe_load(f)
        
        # Get reference images
        refs = case_data.get("refs", {})
        ref_images = [str(resolve_data_path(ref_path, self.data_root)) for ref_path in refs.values()]
        
        # Find candidate image by case_id in the task folder
        # Format: outputs/generations/<model>/<task>/<case_id>.png
        gen_dir = self.generations_root / model_name / task_type
        candidate_image = None
        
        # Search by extension priority
        for ext in [".png", ".jpg", ".jpeg"]:
            test_path = gen_dir / f"{case_id}{ext}"
            if test_path.exists():
                candidate_image = str(test_path)
                break
        
        if not candidate_image:
            logger.error(f"Candidate image not found: {gen_dir}/{case_id}.png|jpg")
            logger.error(f"Ensure the file is named: {case_id}.png or {case_id}.jpg")
            return False
        
        logger.info(f"Judging: {task_type}/{case_id}")
        
        parsed_result, raw_response = self.call_judge(
            prompt_text, ref_images, candidate_image, judge_name
        )
        
        # Save result
        output_dir = self.outputs_root / task_type / case_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if parsed_result is None:
            logger.error(f"Judging failed: {task_type}/{case_id}")
            with open(output_dir / f"{model_name}_error.txt", 'w', encoding='utf-8') as f:
                f.write(raw_response)
            return False
        
        with open(output_dir / f"{model_name}.json", 'w', encoding='utf-8') as f:
            json.dump(parsed_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Done: {task_type}/{case_id}")
        return True
    
    def judge_task_type(self, task_type: str, model_name: str,
                       judge_name: Optional[str] = None, max_workers: int = 1) -> Dict[str, int]:
        """Judge all cases for a task type"""
        task_dir = self.data_root / task_type / "cases"
        if not task_dir.exists():
            logger.warning(f"Task directory not found: {task_dir}")
            return {"success": 0, "failed": 0, "skipped": 0}
        
        # Check generated images directory
        gen_dir = self.generations_root / model_name / task_type
        if not gen_dir.exists():
            logger.warning(f"🚫 Skip task {task_type}: generated images dir not found {gen_dir}")
            return {"success": 0, "failed": 0, "skipped": "all"}
        
        case_ids = [f.stem for f in sorted(task_dir.glob("*.yaml"))]
        success_count = failed_count = skipped_count = 0
        
        for case_id in case_ids:
            # Check candidate image exists
            candidate_exists = False
            for ext in [".png", ".jpg", ".jpeg"]:
                if (gen_dir / f"{case_id}{ext}").exists():
                    candidate_exists = True
                    break
            
            if not candidate_exists:
                logger.info(f"⏭️  Skip {task_type}/{case_id}: candidate image missing")
                skipped_count += 1
                continue
                
            if self.judge_case(task_type, case_id, model_name, judge_name):
                success_count += 1
            else:
                failed_count += 1
        
        logger.info(f"Task {task_type}: success={success_count}, failed={failed_count}, skipped={skipped_count}")
        return {"success": success_count, "failed": failed_count, "skipped": skipped_count}


def main():
    parser = argparse.ArgumentParser(description="Run MLLM judge")
    parser.add_argument("--project-root", type=str, default=".", help="Project root")
    parser.add_argument("--task-type", type=str, default="all", help="Task type or 'all'")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--judge-name", type=str, default=None, help="Judge name")
    parser.add_argument("--config", type=str, default="configs/judge_models.yaml", help="Config path")
    parser.add_argument("--global-config", type=str, default="configs/global.yaml",
                       help="Path to global.yaml (default: configs/global.yaml)")
    parser.add_argument("--timestamp-dir", type=str, default=None, 
                       help="Timestamp directory for outputs (default: auto-generate)")
    parser.add_argument("--output-base-dir", type=str, default=None,
                       help="Base output directory (e.g., 'outputs/eval_res')")
    
    args = parser.parse_args()
    
    # If timestamp_dir is not provided, auto-generate one
    timestamp_dir = args.timestamp_dir
    if not timestamp_dir:
        timestamp_dir = f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Auto-generated timestamp directory: {timestamp_dir}")
    
    runner = JudgeRunner(config_path=args.config, project_root=args.project_root, 
                        timestamp_dir=timestamp_dir, model_name=args.model_name,
                        output_base_dir=args.output_base_dir, global_config=args.global_config)
    
    task_types = ["object_composition", "spatial_geometric_constraints",
                  "local_element_transfer", "attribute_style_decoupling",
                  "foreground_background_composition", "story_infer"]
    
    if args.task_type != "all":
        task_types = [args.task_type]
    
    print("\n=== Judge Summary ===")
    total_success = total_failed = total_skipped = 0
    
    for task_type in task_types:
        result = runner.judge_task_type(task_type, args.model_name, args.judge_name)
        skipped_display = result['skipped'] if result['skipped'] != "all" else "ALL"
        print(f"{task_type:40s}: Success={result['success']:4d}, Failed={result['failed']:4d}, Skipped={skipped_display}")
        total_success += result['success']
        total_failed += result['failed']
        if result['skipped'] != "all":
            total_skipped += result['skipped']
    
    print(f"{'Total':40s}: Success={total_success:4d}, Failed={total_failed:4d}, Skipped={total_skipped:4d}")


if __name__ == "__main__":
    main()
