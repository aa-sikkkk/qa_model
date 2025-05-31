import json
import os
from typing import Dict, List, Set
from pathlib import Path

class CurriculumValidator:
    def __init__(self, data_dir: str = "data/curriculum"):
        self.data_dir = data_dir
        self.required_fields = {
            "subject", "topic", "subtopic", "question", "answer",
            "difficulty", "type", "concepts", "prerequisites",
            "learning_objective"
        }
        self.valid_difficulties = {"easy", "medium", "hard"}
        self.valid_types = {"multiple_choice", "short_answer", "long_answer", "true_false"}
        
    def validate_question(self, question_data: Dict) -> List[str]:
        """Validate a single question entry and return list of issues."""
        issues = []
        
        # Check required fields
        missing_fields = self.required_fields - set(question_data.keys())
        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
            
        # Validate field values
        if "difficulty" in question_data and question_data["difficulty"] not in self.valid_difficulties:
            issues.append(f"Invalid difficulty level: {question_data['difficulty']}")
            
        if "type" in question_data and question_data["type"] not in self.valid_types:
            issues.append(f"Invalid question type: {question_data['type']}")
            
        # Check content quality
        if "question" in question_data:
            if len(question_data["question"]) < 10:
                issues.append("Question is too short")
            if not question_data["question"].endswith("?"):
                issues.append("Question should end with a question mark")
                
        if "answer" in question_data:
            if len(question_data["answer"]) < 20:
                issues.append("Answer is too short")
                
        if "concepts" in question_data:
            if not isinstance(question_data["concepts"], list):
                issues.append("Concepts should be a list")
            elif not question_data["concepts"]:
                issues.append("No concepts specified")
                
        return issues
    
    def validate_dataset(self, subject: str) -> Dict[str, List[str]]:
        """Validate all questions for a subject and return issues by question index."""
        issues = {}
        file_path = os.path.join(self.data_dir, f"{subject}_curriculum.json")
        
        if not os.path.exists(file_path):
            return {"error": [f"File not found: {file_path}"]}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
                
            for i, question in enumerate(questions):
                question_issues = self.validate_question(question)
                if question_issues:
                    issues[f"Question {i+1}"] = question_issues
                    
            return issues
            
        except json.JSONDecodeError:
            return {"error": [f"Invalid JSON in file: {file_path}"]}
        except Exception as e:
            return {"error": [f"Error processing file: {str(e)}"]}
    
    def generate_report(self, subject: str) -> None:
        """Generate a validation report for a subject."""
        print(f"\nValidating {subject} curriculum data...")
        issues = self.validate_dataset(subject)
        
        if "error" in issues:
            print(f"Error: {issues['error'][0]}")
            return
            
        if not issues:
            print("All questions passed validation!")
            return
            
        print("\nValidation Issues Found:")
        for question, question_issues in issues.items():
            print(f"\n{question}:")
            for issue in question_issues:
                print(f"  - {issue}")

def main():
    validator = CurriculumValidator()
    
    # Validate both subjects
    for subject in ["computer_science", "science"]:
        validator.generate_report(subject)

if __name__ == "__main__":
    main() 