import spacy
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import networkx as nx
from tqdm import tqdm
import re

# This class extracts concepts and relationships from text
class ConceptExtractor:
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the concept extractor with a spaCy model."""
        # Load the spaCy language model
        self.nlp = spacy.load(model_name)
        # Set to store unique concepts
        self.concepts: Set[str] = set()
        # List to store relationships between concepts
        self.relationships: List[Tuple[str, str, str]] = []
        # Directed graph to represent the concept map
        self.graph = nx.DiGraph()
        
        # Words and phrases to ignore while extracting concepts
        self.stop_phrases = {
            "chapter", "section", "unit", "page", "figure", "table",
            "example", "note", "exercise", "question", "answer"
        }
        # Question words to ignore as concepts
        self.question_words = {
            "what", "which", "when", "where", "who", "whom", "whose", "why", "how"
        }
        # Common verbs to ignore as relationships
        self.generic_verbs = {
            "has", "have", "had", "made", "make", "take", "write", "states", "called", "consists", "having", "give", "comes", "produced", "explains", "heat", "meet", "obtain", "provided", "bounded", "classified", "reared", "secretes", "is", "are", "was", "were", "be", "been", "get", "put", "set", "use", "used", "using", "form", "forms", "contain", "contains", "including", "include", "includes", "show", "shows", "showing", "see", "seen", "found", "find", "found", "keep", "kept", "become", "became", "becoming", "allow", "allows", "allowed", "let", "lets", "let's", "help", "helps", "helped", "support", "supports", "supported"
        }

    def clean_text(self, text: str) -> str:
        """Clean and preprocess the text."""
        # Remove page numbers and headers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        # Remove special characters but keep periods and commas
        text = re.sub(r'[^\w\s.,;:?!-]', ' ', text)
        return text.strip()

    def is_valid_concept(self, text: str) -> bool:
        """Check if a text is a valid concept."""
        text = text.lower().strip()
        # Ignore stop phrases
        if any(phrase in text for phrase in self.stop_phrases):
            return False
        # Ignore question words
        if text in self.question_words:
            return False
        # Ignore very short concepts
        if len(text.split()) < 2 and len(text) < 4:
            return False
        # Ignore concepts that are just numbers
        if text.replace('.', '').isdigit():
            return False
        return True

    def extract_from_text(self, text: str) -> None:
        """Extract concepts and relationships from text."""
        # Clean the text before processing
        text = self.clean_text(text)
        # Use spaCy to analyze the text
        doc = self.nlp(text)
        
        # Extract noun phrases and named entities as concepts
        for chunk in doc.noun_chunks:
            concept = chunk.text.lower().strip()
            if len(concept.split()) <= 4 and self.is_valid_concept(concept):
                self.concepts.add(concept)
        
        for ent in doc.ents:
            concept = ent.text.lower().strip()
            if self.is_valid_concept(concept):
                self.concepts.add(concept)

        # Extract relationships using dependency parsing
        for sent in doc.sents:
            # Find all valid concepts in the sentence first
            sentence_concepts = {}
            for chunk in sent.noun_chunks:
                 concept = chunk.text.lower().strip()
                 if len(concept.split()) <= 4 and self.is_valid_concept(concept):
                     # Store concept and its head token
                     sentence_concepts[concept] = chunk.root
            for ent in sent.ents:
                 concept = ent.text.lower().strip()
                 if self.is_valid_concept(concept):
                     sentence_concepts[concept] = ent.root
            
            # Iterate through tokens and find relationships
            for token in sent:
                # Subject-Verb relationship
                if "subj" in token.dep_:
                    subject = token.text.lower().strip()
                    verb = token.head.text.lower()
                    if verb and verb not in self.generic_verbs:
                         # Find if the subject is one of our extracted concepts
                         for concept, head_token in sentence_concepts.items():
                              # Check if the subject token is part of or related to a known concept
                              if token.text in concept or head_token == token:
                                  # Find object of the verb
                                  object_ = None
                                  for child in token.head.children:
                                      if "obj" in child.dep_:
                                          object_ = child.text.lower().strip()
                                          # Find if the object is one of our extracted concepts
                                          for obj_concept, obj_head_token in sentence_concepts.items():
                                              if child.text in obj_concept or obj_head_token == child:
                                                  if self.is_valid_concept(concept) and self.is_valid_concept(obj_concept):
                                                      self.relationships.append((concept, verb, obj_concept))
                                                      break # Found a valid object concept
                                          if object_:
                                              break # Found an object token
                                  
                                  # Add subject-verb relationship even without an object if valid concepts
                                  if object_ is None and self.is_valid_concept(concept):
                                       # Check if the verb's head is also in sentence_concepts as an object or target
                                      target_concept = None
                                      for tgt_concept, tgt_head_token in sentence_concepts.items():
                                          if token.head.text in tgt_concept or tgt_head_token == token.head:
                                              target_concept = tgt_concept
                                              break
                                      if target_concept and self.is_valid_concept(target_concept):
                                          self.relationships.append((concept, verb, target_concept))
                                      elif self.is_valid_concept(concept):
                                           pass # Simple subject-verb relationship
                                  
                                  break # Found a valid subject concept

                # Verb-Object relationship (catching cases where the subject might be implicit or already processed)
                elif "obj" in token.dep_:
                     object_ = token.text.lower().strip()
                     verb = token.head.text.lower()
                     subject = token.head.head.text.lower().strip() if token.head.head and "subj" in token.head.head.dep_ else None

                     if verb and verb not in self.generic_verbs:
                         # Find if the object is one of our extracted concepts
                         for concept, head_token in sentence_concepts.items():
                             if token.text in concept or head_token == token:
                                 # If there's a clear subject, use it. Otherwise, look for related concepts.
                                 source_concept = None
                                 if subject:
                                     for subj_concept, subj_head_token in sentence_concepts.items():
                                         if subject in subj_concept or subj_head_token == token.head.head:
                                             source_concept = subj_concept
                                             break
                                 
                                 # If no direct subject found, look for nearby concepts in the sentence
                                 if not source_concept:
                                     for child in token.head.children:
                                          if child != token:
                                              for src_concept, src_head_token in sentence_concepts.items():
                                                  if child.text in src_concept or src_head_token == child:
                                                       source_concept = src_concept
                                                       break
                                              if source_concept: break
                                     if not source_concept and token.head.head:
                                          for src_concept, src_head_token in sentence_concepts.items():
                                               if token.head.head.text in src_concept or src_head_token == token.head.head:
                                                    source_concept = src_concept
                                                    break

                                 if source_concept and self.is_valid_concept(source_concept) and self.is_valid_concept(concept):
                                     self.relationships.append((source_concept, verb, concept))
                                 
                                 break # Found a valid object concept

    def build_graph(self) -> None:
        """Build a directed graph from extracted concepts and relationships."""
        # Add edges to the graph based on relationships
        for source, rel, target in self.relationships:
            self.graph.add_edge(source, target, relationship=rel)

    def save_concept_map(self, output_file: str) -> None:
        """Save the concept map as a JSON file."""
        # Create a dictionary with concepts and relationships
        concept_map = {
            "concepts": sorted(list(self.concepts)),  # Sort for consistency
            "relationships": [
                {"source": s, "relationship": r, "target": t}
                for s, r, t in sorted(self.relationships)  # Sort for consistency
            ]
        }
        
        # Save the concept map to a file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(concept_map, f, indent=2, ensure_ascii=False)

def process_curriculum_file(file_path: str, output_dir: str) -> None:
    """Process a curriculum file and generate its concept map."""
    # Create an instance of the ConceptExtractor
    extractor = ConceptExtractor()
    
    print(f"Processing {file_path}...")
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split the text into chunks to avoid memory issues
    chunk_size = 100000  # characters
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Process each chunk to extract concepts and relationships
    for chunk in tqdm(chunks, desc="Processing chunks"):
        extractor.extract_from_text(chunk)
    
    # Build the graph from extracted relationships
    extractor.build_graph()
    
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save the concept map to a JSON file
    output_file = Path(output_dir) / f"{Path(file_path).stem}_concept_map.json"
    extractor.save_concept_map(str(output_file))
    print(f"Saved concept map to {output_file}")
    print(f"Extracted {len(extractor.concepts)} concepts and {len(extractor.relationships)} relationships")

def main():
    """Main function to process curriculum files."""
    # Directory containing curriculum files
    data_dir = "data/curriculum"
    # Process the science curriculum file
    process_curriculum_file("data/curriculum/sc.txt", data_dir)
    # Process the computer science curriculum file
    process_curriculum_file("data/curriculum/cs.txt", data_dir)

if __name__ == "__main__":
    main()