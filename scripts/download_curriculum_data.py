import json
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

class CurriculumDataDownloader:
    def __init__(self, output_dir: str = "data/curriculum"):
        self.output_dir = output_dir
        self.subjects = ["computer_science", "science"]
        self.data = {subject: [] for subject in self.subjects}
        os.makedirs(output_dir, exist_ok=True)

        # Dataset URLs
        self.dataset_urls = {
            "scienceqa": "https://huggingface.co/datasets/scienceqa/resolve/main/data/train.json",
            "openbookqa": "https://huggingface.co/datasets/allenai/openbookqa/resolve/main/main/train.jsonl"
        }

    def download_json(self, url: str) -> Optional[List[Dict]]:
        try:
            print(f"Downloading {url}...")
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return None

    def download_jsonl(self, url: str) -> Optional[List[Dict]]:
        try:
            print(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            data = []
            for line in tqdm(response.iter_lines()):
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        continue
            return data
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return None

    def process_scienceqa(self, data: List[Dict]) -> List[Dict]:
        processed = []
        for item in data:
            if not item.get("question") or not item.get("choices") or not item.get("answer"):
                continue
            processed.append({
                "subject": "science",
                "topic": item.get("topic", "General Science"),
                "subtopic": item.get("category", "General"),
                "question": item["question"],
                "answer": item["choices"][item["answer"]] if isinstance(item["answer"], int) and isinstance(item["choices"], list) and item["answer"] < len(item["choices"]) else item["answer"],
                "difficulty": item.get("grade", "medium"),
                "type": "multiple_choice",
                "concepts": [item.get("topic", "General Science")],
                "prerequisites": [],
                "learning_objective": item.get("explanation", "")
            })
        return processed

    def process_openbookqa(self, data: List[Dict]) -> List[Dict]:
        processed = []
        for item in data:
            if not item.get("question_stem") or not item.get("choices") or not item.get("answerKey"):
                continue
            answer_idx = None
            choices = item["choices"]["text"] if isinstance(item["choices"], dict) else []
            labels = item["choices"]["label"] if isinstance(item["choices"], dict) else []
            if item["answerKey"] in labels:
                answer_idx = labels.index(item["answerKey"])
            processed.append({
                "subject": "computer_science",
                "topic": "General CS",
                "subtopic": "General",
                "question": item["question_stem"],
                "answer": choices[answer_idx] if answer_idx is not None and answer_idx < len(choices) else item["answerKey"],
                "difficulty": "medium",
                "type": "multiple_choice",
                "concepts": [],
                "prerequisites": [],
                "learning_objective": item.get("fact1", "")
            })
        return processed

    def download_and_process_datasets(self):
        # ScienceQA for science
        scienceqa_data = self.download_json(self.dataset_urls["scienceqa"])
        if scienceqa_data:
            self.data["science"].extend(self.process_scienceqa(scienceqa_data))
        # OpenBookQA for computer science
        openbookqa_data = self.download_jsonl(self.dataset_urls["openbookqa"])
        if openbookqa_data:
            self.data["computer_science"].extend(self.process_openbookqa(openbookqa_data))

    def save_data(self):
        for subject in self.subjects:
            output_file = os.path.join(self.output_dir, f"{subject}_curriculum.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.data[subject], f, indent=2, ensure_ascii=False)
            print(f"Saved {len(self.data[subject])} questions for {subject} to {output_file}")

def main():
    downloader = CurriculumDataDownloader()
    downloader.download_and_process_datasets()
    downloader.save_data()

if __name__ == "__main__":
    main() 