# Concept-Mapping Model

This project is developed for reasearch purpose of analyzing curriculum data, generating concept maps, and creating questions based on the curriculum content. It's designed to help educators and students better understand and interact with educational materials for low-resource settings..

## Features

- Curriculum data collection and validation
- Concept extraction and mapping
- Question generation from concept maps
- Local QA inference system
- Visualization tools for concept maps

## Project Structure

```
.
├── scripts/                    # Main project scripts
│   ├── collect_curriculum_data.py
│   ├── validate_curriculum_data.py
│   ├── extract_concepts.py
│   ├── generate_questions_from_map.py
│   ├── local_qa_inference.py
│   ├── visualize_concept_map.py
│   └── data/                  # Data directory
├── env/                       # Virtual environment
├── requirements.txt           # Project dependencies
└── research.md               # Research documentation
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv env
.\env\Scripts\activate  # Windows
source env/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Collection
```bash
python scripts/collect_curriculum_data.py
```

### Concept Extraction
```bash
python scripts/extract_concepts.py
```

### Question Generation
```bash
python scripts/generate_questions_from_map.py
```

### Local QA Inference
```bash
python scripts/local_qa_inference.py
```

### Visualization
```bash
python scripts/visualize_concept_map.py
```

## Dependencies

- Python 3.x
- torch==2.7.0
- transformers==4.52.3
- spacy==3.8.7
- networkx==3.5
- matplotlib==3.8.2
- nltk==3.9.1
- textblob==0.19.0
- tqdm==4.67.1
- psutil==7.0.0
- Requests==2.32.3

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project uses various open-source libraries and tools
