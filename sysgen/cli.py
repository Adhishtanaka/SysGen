import os
import time
import json
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from google import genai


api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"

class DuplicateDetector:
    def __init__(self, similarity_threshold=0.8):
        self.similarity_threshold = similarity_threshold
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def semantic_duplicate_detection(self, questions: List[str]) -> List[List[int]]:
        """Find semantic duplicates using sentence embeddings"""
        print("Checking for semantic duplicates...")
        embeddings = self.sentence_model.encode(questions)
        similarity_matrix = cosine_similarity(embeddings)
        
        duplicate_groups = []
        processed = set()
        
        for i in range(len(questions)):
            if i in processed:
                continue
                
            similar_indices = []
            for j in range(i, len(questions)):
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    similar_indices.append(j)
                    processed.add(j)
            
            if len(similar_indices) > 1:
                duplicate_groups.append(similar_indices)
        
        return duplicate_groups

class DuplicateRemover:
    def __init__(self):
        pass
    
    def calculate_quality_score(self, question: str, answer: str) -> float:
        """Calculate quality score for a Q&A pair"""
        score = 0
        
        # Question length (prefer 5-20 words)
        q_len = len(question.split())
        if 5 <= q_len <= 20:
            score += 1
        
        # Answer length (prefer 20-200 words)
        a_len = len(answer.split())
        if 20 <= a_len <= 200:
            score += 1
        
        # Proper capitalization
        if question[0].isupper():
            score += 0.5
        
        # Question mark
        if question.strip().endswith('?'):
            score += 0.5
        
        # Answer diversity (avoid repetition)
        words = answer.lower().split()
        if len(words) > 0:
            unique_words = set(words)
            if len(unique_words) / len(words) > 0.7:
                score += 1
        
        return score
    
    def remove_duplicates(self, qa_pairs: List[Dict], duplicate_groups: List[List[int]]) -> List[Dict]:
        """Keep the highest quality example from each duplicate group"""
        indices_to_remove = set()
        
        for group in duplicate_groups:
            if len(group) <= 1:
                continue
            
            # Calculate quality scores for all items in group
            quality_scores = []
            for idx in group:
                qa_pair = qa_pairs[idx]
                question = qa_pair['data'][0]['instruction']
                answer = qa_pair['data'][1]['output']
                score = self.calculate_quality_score(question, answer)
                quality_scores.append((idx, score))
            
            # Sort by quality score (descending)
            quality_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Keep the best one, mark others for removal
            for idx, _ in quality_scores[1:]:
                indices_to_remove.add(idx)
        
        # Remove duplicates
        cleaned_pairs = [qa_pairs[i] for i in range(len(qa_pairs)) if i not in indices_to_remove]
        return cleaned_pairs

def merge_overlapping_groups(groups: List[List[int]]) -> List[List[int]]:
    """Merge groups that have overlapping indices"""
    if not groups:
        return []
    
    set_groups = [set(group) for group in groups]
    merged = []
    
    while set_groups:
        current = set_groups.pop(0)
        merged_with_current = True
        
        while merged_with_current:
            merged_with_current = False
            for i, other in enumerate(set_groups):
                if current & other:
                    current = current | other
                    set_groups.pop(i)
                    merged_with_current = True
                    break
        
        merged.append(list(current))
    
    return merged

def remove_duplicates_from_qa_pairs(qa_pairs: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
    """Remove duplicates from QA pairs"""
    if not qa_pairs:
        return qa_pairs
    
    print(f"Original QA pairs: {len(qa_pairs)}")
    
    # Extract questions for duplicate detection
    questions = [qa_pair['data'][0]['instruction'] for qa_pair in qa_pairs]
    
    # Initialize detector and remover
    detector = DuplicateDetector(similarity_threshold)
    remover = DuplicateRemover()
    
    # Detect semantic duplicates only
    semantic_groups = detector.semantic_duplicate_detection(questions)
    
    print(f"Found {len(semantic_groups)} duplicate groups")
    
    # Remove duplicates
    cleaned_pairs = remover.remove_duplicates(qa_pairs, semantic_groups)
    
    print(f"Cleaned QA pairs: {len(cleaned_pairs)}")
    print(f"Removed {len(qa_pairs) - len(cleaned_pairs)} duplicates")
    
    return cleaned_pairs

def generate_answers(chunk, questions):
    prompt = f"""
    Based on the following text from a document, provide thorough and well-explained answers to each of the following questions.
    
    TEXT:
    {chunk}
    
    QUESTIONS:
    {json.dumps(questions)}
    
    INSTRUCTIONS FOR ANSWERING:
    1. EVERY answer MUST be exactly 4-5 sentences long - this is crucial
    2. Provide detailed explanations with relevant context from the text
    3. Include supporting evidence or reasoning in each answer
    4. Structure each answer to have a clear beginning, explanation, and conclusion
    5. Be concise but comprehensive within the 4-5 sentence constraint
    6. Start each answer with "Answer: " followed by your explanation
    7. Generate only ONE single question (not compound questions like "What is X and why is it important?")
    8. Each generated question should ask about ONE concept only
    
    FORMAT: 
    * Keep each answer self-contained
    * Separate answers with "---" on its own line
    * Don't repeat the questions in your response
    """
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        answers_text = response.text.strip()
        answer_list = answers_text.split('---')
        processed_answers = []
        for answer in answer_list: 
            cleaned_answer = answer.strip()
            if cleaned_answer.startswith("Answer:"):
                cleaned_answer = cleaned_answer[7:].strip()
            if cleaned_answer: 
                processed_answers.append(cleaned_answer)
        
        return processed_answers
    except Exception as e:
        print(f"Error generating answers: {e}")
        return []

def generate_diverse_questions(chunk, num_questions):
    prompt = f"""
    Based on the following text from a document, generate {num_questions} diverse and specific questions that could be asked about this content.

    INSTRUCTIONS FOR QUESTION GENERATION:
    1. Make questions clear, direct, and focused on a SINGLE concept
    2. Each question should address only ONE specific aspect from the text
    3. Avoid compound questions or connecting multiple ideas with "and" or commas
    4. Ensure questions are contextually grounded in the document's content
    5. Questions should clearly indicate what specific information they're seeking
    6. Use proper grammar and punctuation
    7. Keep questions concise - aim for 15 words or fewer per question
    8. Each question should have clear relevance to the document subject

    TEXT:
    {chunk}

    FORMAT YOUR RESPONSE AS A JSON ARRAY OF QUESTION STRINGS ONLY.
    """

    try:
        response = client.models.generate_content(model=model, contents=prompt)
        questions_text = response.text
        json_start = questions_text.find('[')
        json_end = questions_text.rfind(']') + 1
        if json_start >= 0 and json_end > json_start:
            questions_json = questions_text[json_start:json_end]
            questions = json.loads(questions_json)
            return questions
        else:
            return []
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

def process_single_textfile(textfile_path, output_file, num_questions, run_number):
    all_conversations = []

    try:
        with open(textfile_path, "r", encoding="utf-8") as file:
            chunk = file.read()

        questions = generate_diverse_questions(chunk, num_questions)
        answers = generate_answers(chunk, questions)

        for question, answer in zip(questions, answers):
            if question and answer:
                sentence_count = len([s for s in answer.split(".") if s.strip()])
                if 3 <= sentence_count <= 6:
                    alpaca_conversation = [
                        {"instruction": question},
                        {"output": answer},
                    ]
                    all_conversations.append(
                        {
                            "data": alpaca_conversation,
                            "source_document": os.path.basename(textfile_path),
                            "run_number": run_number,
                        }
                    )

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend(all_conversations)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)

        print(
            f"Run {run_number}: Generated {len(all_conversations)} QA pairs for {textfile_path}"
        )
        return True

    except Exception as e:
        print(f"Run {run_number}: Error processing file {textfile_path}: {e}")
        return False

def clean_duplicates_from_file(output_file, similarity_threshold=0.8):
    """Clean duplicates from the output file"""
    if not os.path.exists(output_file):
        print(f"Output file {output_file} does not exist.")
        return
    
    print(f"\nCleaning duplicates from {output_file}...")
    
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Remove duplicates
    cleaned_data = remove_duplicates_from_qa_pairs(data, similarity_threshold)
    
    # Save cleaned data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f"Cleaned data saved to {output_file}")

def process_all_md_files(md_folder, output_file, num_questions, repeat, similarity_threshold):
    md_files = [f for f in os.listdir(md_folder) if f.endswith(".md")]

    for md_file in md_files:
        md_path = os.path.join(md_folder, md_file)
        for run in range(1, repeat + 1):
            print(f"Run {run}/{repeat}: Processing: {md_path}")
            success = process_single_textfile(md_path, output_file, num_questions, run)

            if success:
                print(
                    f"Run {run}/{repeat}: Completed: {md_path}. Waiting 60 seconds before next run..."
                )
                time.sleep(60)
            else:
                print(f"Run {run}/{repeat}: Skipping: {md_path} due to an error.")
    
    # Clean duplicates after all processing is done
    clean_duplicates_from_file(output_file, similarity_threshold)

def main():
    parser = argparse.ArgumentParser(description="AI-powered Q&A generator with duplicate removal")
    parser.add_argument(
        "--md-folder", type=str, default="md", help="Folder containing markdown files"
    )
    parser.add_argument(
        "--output", type=str, default="output.json", help="Output JSON file"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=100,
        help="Number of questions per document",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="How many times to process each document"
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.8, help="Similarity threshold for duplicate detection (0.0-1.0)"
    )
    
    args = parser.parse_args()

    if not api_key:
        print(
            "Error: API key is required. Set GEMINI_API_KEY environment variable."
        )
        return

    print(f"Processing Markdown files from {args.md_folder}")
    print(f"Number of questions per document: {args.num_questions}")
    print(f"Repetitions per document: {args.repeat}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    
    process_all_md_files(
        args.md_folder, 
        args.output, 
        args.num_questions, 
        args.repeat, 
        args.similarity_threshold
    )

if __name__ == "__main__":
    main()