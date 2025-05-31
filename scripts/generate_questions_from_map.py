# Batch Hugging Face answer generation enabled for Colab.
import json
import sys
from pathlib import Path
import random
import re
import csv

# Import Hugging Face pipeline for answer generation
try:
    from transformers import pipeline
    hf_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    hf_available = True
except ImportError:
    hf_available = False
    hf_generator = None
    print("[WARN] Hugging Face transformers not installed. Answers will use template fallback.")
except Exception as e:
    hf_available = False
    hf_generator = None
    print(f"[WARN] Could not load Hugging Face model: {e}. Answers will use template fallback.")

# Trying to import NLTK's WordNetLemmatizer for better verb normalization
try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    nltk_available = True
except ImportError:
    nltk_available = False

# Trying to import TextBlob for spellchecking
try:
    from textblob import Word as TBWord
    textblob_available = True
except ImportError:
    textblob_available = False

# Fallback: a small set of common English verbs for basic validation
COMMON_ENGLISH_VERBS = set([
    "be", "have", "do", "say", "get", "make", "go", "know", "take", "see", "come", "think", "look", "want", "give", "use", "find", "tell", "ask", "work", "seem", "feel", "try", "leave", "call", "put", "keep", "let", "begin", "help", "talk", "turn", "start", "show", "hear", "play", "run", "move", "live", "believe", "bring", "write", "provide", "sit", "stand", "lose", "pay", "meet", "include", "continue", "set", "learn", "change", "lead", "understand", "watch", "follow", "stop", "create", "speak", "read", "allow", "add", "spend", "grow", "open", "walk", "win", "offer", "remember", "love", "consider", "appear", "buy", "wait", "serve", "die", "send", "expect", "build", "stay", "fall", "cut", "reach", "kill", "remain", "suggest", "raise", "pass", "sell", "require", "report", "decide", "pull", "return", "explain", "hope", "develop", "carry", "break", "receive", "agree", "support", "hit", "produce", "eat", "cover", "catch", "draw", "choose", "cause", "point", "listen", "realize", "place", "form", "join", "reduce", "establish", "act", "apply", "prepare", "teach", "contain", "control", "manage", "describe", "design", "test", "connect", "store", "relate", "indicate", "emit", "cross", "regulate", "generate", "weigh", "minimize", "protect", "memorize", "drop", "resemble", "calculate", "pump", "fuse", "dissociate", "corrode", "deplete", "neutralize", "absorb", "release", "supply", "surround", "destroy", "combine", "convert", "divide", "move", "occur", "produce", "result", "trigger", "affect", "lead", "influence", "contain", "include", "consist", "comprise", "use", "connect", "provide", "follow", "require"
])

def validate_directory_structure():
    """Validates and creates necessary directories."""
    base_dir = Path(__file__).parent.parent / "scripts/data"
    concept_maps_dir = base_dir / "concept_maps"
    generated_questions_dir = base_dir / "generated_questions"
    
    # Create directories if they don't exist
    concept_maps_dir.mkdir(parents=True, exist_ok=True)
    generated_questions_dir.mkdir(parents=True, exist_ok=True)
    
    return concept_maps_dir, generated_questions_dir

def load_concept_map(json_path):
    """Loads a concept map from a JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "relationships" not in data:
            print(f"Error: \"{json_path}\" does not contain a 'relationships' key.")
            print("Expected format: {\"relationships\": [[\"source\", \"verb\", \"target\"], ...]}")
            return None
        if not data["relationships"]:
            print(f"Error: No relationships found in \"{json_path}\".")
            return None
        return data
    except FileNotFoundError:
        print(f"Error: Concept map file not found at \"{json_path}\"")
        print("Please ensure the concept map JSON file exists in the scripts/data/concept_maps directory.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from \"{json_path}\". Invalid JSON format.")
        print("Please ensure the file contains valid JSON with a 'relationships' array.")
        return None

def lemmatize_verb(verb):
    if nltk_available:
        lemmatizer = WordNetLemmatizer()
        return lemmatizer.lemmatize(verb, 'v')
    # Fallback: remove common suffixes
    for suf in ["sses", "ies", "es", "s", "ed", "ing"]:
        if verb.endswith(suf):
            return verb[:-len(suf)]
    return verb

def is_valid_verb(verb):
    if nltk_available:
        from nltk.corpus import wordnet
        return bool(wordnet.synsets(verb, pos=wordnet.VERB))
    return verb in COMMON_ENGLISH_VERBS

def spellcheck_word(word):
    if textblob_available:
        w = TBWord(word)
        return str(w.correct())
    # Fallback: simple replacements for common errors
    corrections = {
        "copys": "copies", "supplys": "supplies", "focuss": "focuses", "crosseds": "crosses", "emitss": "emits", "identifys": "identifies"
    }
    return corrections.get(word, word)

def clean_concept(concept):
    # Remove newlines, excessive whitespace, and non-informative content
    c = re.sub(r'\s+', ' ', str(concept)).strip()
    # Remove leading/trailing punctuation
    c = c.strip('.,;:!?-"\'()[]{}')
    # Filter out concepts that are too short, too long, or mostly symbols/numbers
    if len(c) < 2 or len(c) > 60:
        return ''
    if re.fullmatch(r'[^a-zA-Z0-9]+', c):
        return ''
    if re.fullmatch(r'[0-9 ]+', c):
        return ''
    # Remove concepts with odd patterns (e.g., numbers+letters, fragments)
    if re.fullmatch(r'[a-zA-Z]* ?[0-9]+[a-zA-Z]*', c):
        return ''
    return c

def is_generic_concept(concept):
    GENERIC_WORDS = {"thing", "things", "something", "anything", "everything", "object", "item", "aspect", "that", "them", "itself", "it", "they", "this", "those"}
    return concept.lower() in GENERIC_WORDS

def is_tautology(source, target):
    s = source.strip().lower()
    t = target.strip().lower()
    return s == t or s.replace(' ', '') == t.replace(' ', '')

def is_incomplete_question(q):
    prepositions = set(["of", "to", "for", "with", "on", "at", "by", "from", "up", "about", "into", "over", "after", "beneath", "under", "above"])
    words = q.lower().split()
    if len(words) < 5:
        return True
    if words[-1] in prepositions:
        return True
    if any(w in ["ofs", "ins", "betweens", "acceptss", "intros", "returnss", "askss", "fors", "concernings", "passeds", "ivs", "ass", "ofs", "ons", "usess", "makess", "writess", "readss", "openss", "closess", "deletess", "prepares", "developings", "encourageds", "providess", "representss", "identifiess", "selectss", "declares", "executes", "supplies", "retrievess", "saves", "modifies", "edits", "acceptss", "returns", "asks", "maintainss", "leads", "regulates", "arranges", "controls", "enables", "displays", "calculates", "transfers", "sends", "receives", "assigns", "chooses", "lists", "sorts", "filters", "matches", "marks", "names", "organizes", "plans", "prints", "processes", "provides", "removes", "replaces", "resets", "restores", "retrieves", "runs", "searches", "shares", "shows", "starts", "stops", "stores", "supports", "switches", "tests", "trains", "updates", "uploads", "uses", "validates", "verifies", "views", "writes"] for w in words):
        return True
    return False

def spellcheck_question(q):
    # Only spellcheck verbs in the question (simple heuristic: looks for 'Who or what <verb> ...' and 'What does ... <verb> ...')
    q = re.sub(r'\b(Who or what|What|How|Why) ([a-zA-Z]+)s\b', lambda m: m.group(0).replace(m.group(2)+'s', spellcheck_word(m.group(2))+'s'), q)
    q = re.sub(r'\b(Who or what|What|How|Why) ([a-zA-Z]+)ed\b', lambda m: m.group(0).replace(m.group(2)+'ed', spellcheck_word(m.group(2))+'ed'), q)
    q = re.sub(r'\b(Who or what|What|How|Why) ([a-zA-Z]+)\b', lambda m: m.group(0).replace(m.group(2), spellcheck_word(m.group(2))), q)
    return q

def generate_hf_answers(questions, contexts, batch_size=16):
    """Generate answers in batches using Hugging Face model."""
    if not hf_available or hf_generator is None:
        return [None] * len(questions)
    prompts = [f"Question: {q}\nContext: {c}\nAnswer:" for q, c in zip(questions, contexts)]
    answers = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        try:
            results = hf_generator(batch_prompts, max_new_tokens=32)
            for res in results:
                answer = res.get('generated_text', res.get('text', '')).strip()
                if answer.lower().startswith('answer:'):
                    answer = answer[7:].strip()
                answers.append(answer)
        except Exception as e:
            print(f"[WARN] Hugging Face batch answer generation failed: {e}")
            answers.extend([None] * len(batch_prompts))
    return answers

def generate_questions(concept_map, num_questions=None):
    relationships = concept_map.get("relationships", [])
    questions = []  # Now a list of dicts: {question, answer, source, verb, target}
    if not relationships:
        print("No relationships found in the concept map.")
        return []
    blacklisted_verbs = set([
        "of", "from", "into", "by", "with", "on", "in", "at", "to", "for", "as", "c", "2s", "1s", "vs", "and", "or", "but", "if", "then", "else", "than", "so", "because", "although", "though", "while", "whereas", "given", "state", "law", "about", "between", "among", "during", "after", "before", "above", "below", "under", "over", "through", "across", "per", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "do", "does", "did", "done", "doing", "will", "would", "can", "could", "shall", "should", "may", "might", "must", "not", "no", "yes", "a", "an", "the", "this", "that", "these", "those", "it", "they", "we", "you", "i", "he", "she", "who", "whom", "whose", "which", "what", "where", "when", "why", "how"
    ])
    def is_blacklisted_verb(verb):
        return verb in blacklisted_verbs

    def select_templates_and_answer(source, verb, target):
        # Returns [(question, template_answer), ...]
        causal_verbs = ["causes", "affects", "leads to", "results in", "influences", "triggers"]
        comp_verbs = ["contains", "includes", "has", "consists of", "comprises"]
        def_verbs = ["is", "are", "was", "were", "be"]
        action_verbs = ["uses", "connects", "provides", "follows", "requires"]
        general_templates = [
            (f"What is the relationship between {source} and {target}?", f"{source} and {target} are related by {verb}.") ,
            (f"Explain the connection between {source} and {target}.", f"{source} {verb} {target}.") ,
            (f"How are {source} and {target} related?", f"{source} and {target} are related by {verb}.")
        ]
        pairs = []
        v = verb.lower()
        if v in causal_verbs:
            pairs = [
                (f"Why does {source} {verb} {target}?", f"Because {source} {verb} {target}.") ,
                (f"How does {source} {verb} {target}?", f"{source} {verb} {target}.") ,
                (f"What effect does {source} have on {target}?", f"{source} {verb} {target}.")
            ]
        elif v in comp_verbs:
            pairs = [
                (f"What does {source} {verb}?", f"{target}"),
                (f"List the components of {source}.", f"{source} {verb} {target}.") ,
                (f"Describe what {source} {verb}.", f"{source} {verb} {target}.")
            ]
        elif v in def_verbs:
            pairs = [
                (f"What is {source}?", f"{source} is {target}.") ,
                (f"Define {source}.", f"{source} is {target}.") ,
                (f"What are {source} and how are they related to {target}?", f"{source} is related to {target} by {verb}.")
            ]
        elif v in action_verbs:
            pairs = [
                (f"How does {source} {verb} {target}?", f"{source} {verb} {target}.") ,
                (f"Describe how {source} {verb} {target}.", f"{source} {verb} {target}.") ,
                (f"What is the role of {source} in relation to {target}?", f"{source} {verb} {target}.")
            ]
        else:
            pairs = general_templates
        # Add a few dependency-inspired variants
        pairs.extend([
            (f"What {verb}s {target}?", f"{source}"),
            (f"What does {source} {verb}?", f"{target}"),
            (f"Who or what {verb}s {target}?", f"{source}")
        ])
        return pairs

    random.shuffle(relationships)
    debug_count = 0
    all_questions = []
    all_contexts = []
    all_template_answers = []
    all_sources = []
    all_verbs = []
    all_targets = []

    for rel in relationships:
        if debug_count < 20:
            print(f"DEBUG: Relationship: {rel}")
            debug_count += 1
        if isinstance(rel, dict):
            source = clean_concept(rel.get("source", ""))
            verb = rel.get("relationship", rel.get("verb", "")).strip().lower()
            verb = lemmatize_verb(verb)
            target = clean_concept(rel.get("target", ""))
        elif isinstance(rel, (list, tuple)) and len(rel) == 3:
            source = clean_concept(rel[0])
            verb = lemmatize_verb(str(rel[1]).strip().lower())
            target = clean_concept(rel[2])
        else:
            continue
        if not source or not verb or not target:
            continue
        if is_generic_concept(source) or is_generic_concept(target):
            continue
        if is_blacklisted_verb(verb):
            continue
        if not is_valid_verb(verb):
            continue
        if is_tautology(source, target):
            continue
        pairs = select_templates_and_answer(source, verb, target)
        for q, template_a in pairs:
            q = spellcheck_question(q)
            if is_incomplete_question(q):
                continue
            context = f"{source} {verb} {target}."
            all_questions.append(q)
            all_contexts.append(context)
            all_template_answers.append(template_a)
            all_sources.append(source)
            all_verbs.append(verb)
            all_targets.append(target)
            if num_questions and len(all_questions) >= num_questions:
                break
        if num_questions and len(all_questions) >= num_questions:
            break

    # Batch generate answers
    hf_answers = generate_hf_answers(all_questions, all_contexts)

    # Build the final questions list
    questions = []
    for q, hf_a, template_a, source, verb, target in zip(all_questions, hf_answers, all_template_answers, all_sources, all_verbs, all_targets):
        answer = hf_a if hf_a else template_a
        questions.append({
            "question": q,
            "answer": answer,
            "source": source,
            "verb": verb,
            "target": target
        })
    return questions

def save_questions(questions, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, q in enumerate(questions):
                f.write(f"{i+1}. {q['question']}\n")
        print(f"Saved {len(questions)} questions to {output_file}")
    except IOError as e:
        print(f"Error saving questions to {output_file}: {e}")

def save_questions_csv(questions, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Question", "Answer", "Source", "Verb", "Target"])
            for q in questions:
                writer.writerow([q["question"], q["answer"], q["source"], q["verb"], q["target"]])
        print(f"Saved {len(questions)} questions (with answers) to {output_file}")
    except IOError as e:
        print(f"Error saving questions to {output_file}: {e}")

def main():
    concept_maps_dir, generated_questions_dir = validate_directory_structure()
    if len(sys.argv) < 2:
        json_files = list(concept_maps_dir.glob("*.json"))
        if not json_files:
            print(f"No .json files found in {concept_maps_dir}. Please add concept map files.")
            sys.exit(1)
        print(f"No specific file provided. Processing all {len(json_files)} concept map files in {concept_maps_dir}...")
        for json_path in json_files:
            print(f"Processing: {json_path.name}")
            concept_map = load_concept_map(str(json_path))
            if concept_map:
                questions = generate_questions(concept_map)
                if questions:
                    output_file = generated_questions_dir / f"{json_path.stem}_questions.txt"
                    save_questions(questions, str(output_file))
                    output_csv = generated_questions_dir / f"{json_path.stem}_questions.csv"
                    save_questions_csv(questions, str(output_csv))
                else:
                    print(f"No questions generated for {json_path.name}.")
            else:
                print(f"Skipping {json_path.name} due to errors.")
        print("\nDone processing all concept maps.")
    else:
        json_path = concept_maps_dir / sys.argv[1]
        num_questions = int(sys.argv[2]) if len(sys.argv) > 2 else None
        print(f"Processing: {json_path.name}")
        concept_map = load_concept_map(str(json_path))
        if concept_map:
            questions = generate_questions(concept_map, num_questions)
            if questions:
                output_file = generated_questions_dir / f"{json_path.stem}_questions.txt"
                save_questions(questions, str(output_file))
                output_csv = generated_questions_dir / f"{json_path.stem}_questions.csv"
                save_questions_csv(questions, str(output_csv))
            else:
                print("No questions generated. This could be due to:")
                print("1. All relationships were filtered out by the verb blacklist")
                print("2. All relationships were identified as tautologies")
                print("3. All generated questions were filtered as incomplete")
                print("\nTry running with debug output to see which relationships are being processed.")
        else:
            print(f"Skipping {json_path.name} due to errors.")

if __name__ == "__main__":
    main() 