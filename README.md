# Sysgen

Sysgen is a CLI tool that creates high-quality synthetic datasets using the Gemini API. It analyzes Markdown documents to generate realistic and diverse examples for machine learning, software testing, and data analysis.

## Features

- **Automated Q&A Generation**: Extracts questions and answers from Markdown files using AI
- **Semantic Duplicate Detection**: Automatically removes duplicate questions using sentence embeddings
- **Quality Scoring**: Ranks Q&A pairs based on question clarity and answer completeness
- **Customizable Question Count**: Define the number of questions per document
- **Multiple Runs**: Process each document multiple times to generate varied outputs
- **Similarity Threshold Control**: Adjust duplicate detection sensitivity
- **JSON Output Format**: Saves results in a structured JSON file

## Installation

### Install from pip
```bash
pip install sysgen
```

### Set Up Environment Variables
Before running sysgen, set the API key in your terminal:
```bash
# Windows
set GEMINI_API_KEY=your_gemini_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

Run the script with the following command:
```bash
sysgen --md-folder path/to/md --output output.json --num-questions 50 --repeat 1 --similarity-threshold 0.8
```

### Arguments

- `--md-folder`: Folder containing Markdown files (default: `md`)
- `--output`: Output JSON file (default: `output.json`)
- `--num-questions`: Number of questions per document (default: `100`)
- `--repeat`: Number of times to process each document (default: `1`)
- `--similarity-threshold`: Similarity threshold for duplicate detection, 0.0-1.0 (default: `0.8`)

### Quality Control Features

- **Answer Length Validation**: Ensures answers are 4-5 sentences long
- **Question Clarity**: Prioritizes single-concept questions over compound questions
- **Automatic Deduplication**: Removes semantically similar questions while keeping the highest quality version
- **Quality Scoring**: Evaluates Q&A pairs based on:
  - Question length (5-20 words preferred)
  - Answer length (20-200 words preferred)
  - Proper capitalization and punctuation
  - Answer diversity and uniqueness

## Output Format

The generated JSON follows this structure:
```json
[
  {
    "data": [
      {"instruction": "Question here"},
      {"output": "Answer here"}
    ],
    "source_document": "filename.md",
    "run_number": 1
  }
]
```

## How It Works

1. **Document Processing**: Reads Markdown files from the specified folder
2. **Question Generation**: Uses Gemini API to generate diverse, contextually grounded questions
3. **Answer Generation**: Creates detailed 4-5 sentence answers with supporting evidence
4. **Quality Filtering**: Validates answer length and content quality
5. **Duplicate Detection**: Uses sentence embeddings to identify semantically similar questions
6. **Duplicate Removal**: Keeps the highest quality version from each duplicate group
7. **Output**: Saves cleaned, high-quality Q&A pairs to JSON file

## Dependencies

- `google-genai`: Gemini API client
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `sentence-transformers`: Semantic similarity detection
- `scikit-learn`: Machine learning utilities for cosine similarity

## Contributing

If you find a bug or have suggestions for improvement, feel free to open an issue or submit a pull request on GitHub.

### How to Contribute

1. **Fork the Repository**: Start by forking the project on GitHub
2. **Clone the Repository**: Clone it to your local machine using:
   ```sh
   git clone https://github.com/your-username/sysgen.git
   ```
3. **Create a Branch**: Create a new branch for your changes:
   ```sh
   git checkout -b feature-branch-name
   ```
4. **Make Changes**: Implement your improvements or bug fixes
5. **Commit Your Changes**: Write a clear commit message:
   ```sh
   git commit -m "Added feature XYZ"
   ```
6. **Push to GitHub**: Push your changes:
   ```sh
   git push origin feature-branch-name
   ```
7. **Submit a Pull Request**: Open a PR describing your changes
8. **Review & Merge**: Wait for review and approval before merging

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Author**: [Adhishtanaka](https://github.com/Adhishtanaka)
- **Email**: kulasoooriyaa@gmail.com