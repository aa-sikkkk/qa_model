import json
import os
from pathlib import Path
from typing import Dict, List, Optional

class CurriculumDataCollector:
    def __init__(self, output_dir: str = "data/curriculum"):
        self.output_dir = output_dir
        self.subjects = ["computer_science", "science"]
        self.data = {subject: [] for subject in self.subjects}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def add_question(self, 
                    subject: str,
                    topic: str,
                    subtopic: str,
                    question: str,
                    answer: str,
                    difficulty: str = "medium",
                    question_type: str = "short_answer",
                    concepts: List[str] = None,
                    prerequisites: List[str] = None,
                    learning_objective: str = None) -> None:
        """Add a question to the curriculum dataset."""
        if subject not in self.subjects:
            raise ValueError(f"Invalid subject: {subject}")
            
        question_data = {
            "subject": subject,
            "topic": topic,
            "subtopic": subtopic,
            "question": question,
            "answer": answer,
            "difficulty": difficulty,
            "type": question_type,
            "concepts": concepts or [],
            "prerequisites": prerequisites or [],
            "learning_objective": learning_objective
        }
        
        self.data[subject].append(question_data)
    
    def save_data(self) -> None:
        """Save the collected data to JSON files."""
        for subject in self.subjects:
            output_file = os.path.join(self.output_dir, f"{subject}_curriculum.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.data[subject], f, indent=2, ensure_ascii=False)
            print(f"Saved {len(self.data[subject])} questions for {subject} to {output_file}")

def main():
    # Initialize the collector
    collector = CurriculumDataCollector()
    
    # Computer Science Questions
    collector.add_question(
        subject="computer_science",
        topic="Networking",
        subtopic="Network Types",
        question="What is the main difference between LAN and WAN?",
        answer="LAN (Local Area Network) covers a small geographical area like a building or campus, while WAN (Wide Area Network) covers a large geographical area like cities or countries.",
        difficulty="medium",
        concepts=["LAN", "WAN", "networking", "geographical coverage"],
        prerequisites=["basic networking concepts"],
        learning_objective="Understand the different types of networks based on geographical coverage"
    )
    
    collector.add_question(
        subject="computer_science",
        topic="Programming",
        subtopic="Variables and Data Types",
        question="What is the difference between integer and floating-point data types in programming?",
        answer="Integer data types store whole numbers without decimal points, while floating-point data types can store numbers with decimal points. Integers are more memory efficient but have limited precision, while floating-point numbers can represent a wider range of values but may have precision limitations.",
        difficulty="medium",
        concepts=["data types", "integers", "floating-point", "memory", "precision"],
        prerequisites=["basic programming concepts"],
        learning_objective="Understand different numeric data types and their characteristics"
    )
    
    collector.add_question(
        subject="computer_science",
        topic="Algorithms",
        subtopic="Sorting",
        question="Which sorting algorithm is most efficient for small datasets?",
        answer="Insertion sort is most efficient for small datasets (typically less than 50 elements) because it has low overhead and performs well on nearly sorted data. However, for larger datasets, more complex algorithms like quicksort or mergesort are more efficient.",
        difficulty="medium",
        question_type="multiple_choice",
        concepts=["sorting", "algorithms", "efficiency", "insertion sort"],
        prerequisites=["basic understanding of algorithms"],
        learning_objective="Understand algorithm efficiency for different input sizes"
    )
    
    collector.add_question(
        subject="computer_science",
        topic="Databases",
        subtopic="SQL",
        question="What is the difference between INNER JOIN and LEFT JOIN in SQL?",
        answer="INNER JOIN returns only the matching records from both tables, while LEFT JOIN returns all records from the left table and matching records from the right table. If there's no match in the right table, NULL values are returned for those columns.",
        difficulty="hard",
        question_type="multiple_choice",
        concepts=["SQL", "joins", "databases", "querying"],
        prerequisites=["basic SQL knowledge"],
        learning_objective="Understand different types of SQL joins and their use cases"
    )
    
    # Science Questions
    collector.add_question(
        subject="science",
        topic="Physics",
        subtopic="Forces and Motion",
        question="What is Newton's First Law of Motion and how does it relate to inertia?",
        answer="Newton's First Law states that an object will remain at rest or in uniform motion unless acted upon by an external force. This property of objects to resist changes in their state of motion is called inertia. The law demonstrates that inertia is a fundamental property of matter.",
        difficulty="medium",
        concepts=["Newton's First Law", "inertia", "force", "motion"],
        prerequisites=["basic concepts of force and motion"],
        learning_objective="Understand the relationship between force, motion, and inertia"
    )
    
    collector.add_question(
        subject="science",
        topic="Chemistry",
        subtopic="Chemical Reactions",
        question="What is the difference between exothermic and endothermic reactions?",
        answer="Exothermic reactions release energy to the surroundings (usually as heat), while endothermic reactions absorb energy from the surroundings. In exothermic reactions, the products have less energy than the reactants, while in endothermic reactions, the products have more energy than the reactants.",
        difficulty="medium",
        concepts=["exothermic", "endothermic", "energy", "chemical reactions"],
        prerequisites=["basic understanding of chemical reactions"],
        learning_objective="Understand energy changes in chemical reactions"
    )
    
    collector.add_question(
        subject="science",
        topic="Biology",
        subtopic="Cell Structure",
        question="What is the function of the mitochondria in a cell?",
        answer="Mitochondria are known as the powerhouse of the cell. They generate most of the cell's supply of adenosine triphosphate (ATP), which is used as a source of chemical energy. They also play a role in cell signaling, cellular differentiation, and cell death.",
        difficulty="easy",
        question_type="multiple_choice",
        concepts=["cell biology", "mitochondria", "energy", "cellular respiration"],
        prerequisites=["basic cell biology"],
        learning_objective="Understand the role of mitochondria in cellular function"
    )
    
    collector.add_question(
        subject="science",
        topic="Earth Science",
        subtopic="Plate Tectonics",
        question="How do convergent plate boundaries contribute to mountain formation?",
        answer="Convergent plate boundaries contribute to mountain formation through a process called orogeny. When two continental plates collide, neither plate can be subducted due to their low density. Instead, the crust is compressed, folded, and uplifted, forming mountain ranges like the Himalayas.",
        difficulty="hard",
        question_type="multiple_choice",
        concepts=["plate tectonics", "mountain formation", "convergent boundaries", "orogeny"],
        prerequisites=["basic understanding of plate tectonics"],
        learning_objective="Understand the relationship between plate boundaries and mountain formation"
    )
    
    # Save the collected data
    collector.save_data()

if __name__ == "__main__":
    main() 