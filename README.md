# Sysgen
Sysgen is a CLI tool that creates high-quality synthetic datasets using the Gemini API. It analyzes Markdown documents to generate realistic and diverse examples for machine learning, software testing, and data analysis.

## Features
- **Automated Q&A Generation**: Extracts questions and answers from Markdown files using AI.
- **Customizable Question Count**: Define the number of questions per document.
- **Multiple Runs**: Process each document multiple times to generate varied outputs.
- **JSON Output Format**: Saves results in a structured JSON file.

## Installation

### install it from pip
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
sysgen --md-folder path/to/md --output output.json --num-questions 50 --repeat 1
```

### Arguments
- `--md-folder`: Folder containing Markdown files (default: `md`)
- `--output`: Output JSON file (default: `output.json`)
- `--num-questions`: Number of questions per document (default: `100`)
- `--repeat`: Number of times to process each document (default: `1`)

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

## Contributing
If you find a bug or have suggestions for improvement, feel free to open an issue or submit a pull request on GitHub.

### How to Contribute
1. **Fork the Repository**: Start by forking the project on GitHub.
2. **Clone the Repository**: Clone it to your local machine using:
   ```sh
   git clone https://github.com//your-username/sysgen.git
   ```
3. **Create a Branch**: Create a new branch for your changes:
   ```sh
   git checkout -b feature-branch-name
   ```
4. **Make Changes**: Implement your improvements or bug fixes.
5. **Commit Your Changes**: Write a clear commit message:
   ```sh
   git commit -m "Added feature XYZ"
   ```
6. **Push to GitHub**: Push your changes:
   ```sh
   git push origin feature-branch-name
   ```
7. **Submit a Pull Request**: Open a PR describing your changes.
8. **Review & Merge**: Wait for review and approval before merging.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact
- **Author**: [Adhishtanaka](https://github.com/Adhishtanaka)
- **Email**: kulasoooriyaa@gmail.com
