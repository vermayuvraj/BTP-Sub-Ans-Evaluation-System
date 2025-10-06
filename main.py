
import cv2
import os
import pytesseract
from sentence_transformers import SentenceTransformer, util
import language_tool_python
import textstat

# --- 1. CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
IMAGE_PATH = os.path.join('sample_images', 'answer_sheet_2.jpg')

# --- 2. LOAD MODELS (This will be fast as they are cached) ---
print("Loading models...")
sbert_model = SentenceTransformer('all-mpnet-base-v2')
grammar_tool = language_tool_python.LanguageTool('en-US')
print("Models loaded successfully.")


def extract_text_blocks(image_path):
    """Performs DLA and OCR to extract text from an image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image at {image_path}")
            return None, None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilated = cv2.dilate(thresh, kernel, iterations=4)
    except Exception as e:
        print(f"An error occurred during image processing: {e}")
        return None, None

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    extracted_data = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50 and w < (image.shape[1] * 0.95):
            padded_y_start = max(0, y - 10)
            padded_y_end = min(image.shape[0], y + h + 10)
            padded_x_start = max(0, x - 10)
            padded_x_end = min(image.shape[1], x + w + 10)
            
            cropped_region = gray[padded_y_start:padded_y_end, padded_x_start:padded_x_end]
            
            try:
                text = pytesseract.image_to_string(cropped_region)
                if text.strip():
                    extracted_data.append((text, (x, y, w, h)))
            except Exception as e:
                print(f"An error occurred during OCR: {e}")
                
    return image, extracted_data


def calculate_semantic_similarity(model_answer, student_answer, model):
    """Calculates semantic similarity score."""
    embedding1 = model.encode(model_answer, convert_to_tensor=True)
    embedding2 = model.encode(student_answer, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_score.item()


def analyze_writing_style(text, tool):
    """
    Analyzes text for grammar errors and readability.
    Returns the number of errors and the Flesch-Kincaid grade level.
    """
    matches = tool.check(text)
    error_count = len(matches)
    grade_level = textstat.flesch_kincaid_grade(text)
    return error_count, grade_level

# --- NEW: Final Grading Function ---
def calculate_final_grade(semantic_score, grammar_errors):
    """Calculates a final grade based on semantics and errors."""
    
    # 1. Check if the answer is semantically correct enough to pass
    if semantic_score < 0.75:
        return "Failed (Content)", semantic_score * 100

    # 2. Start with the semantic score as the base grade
    final_score = semantic_score * 100

    # 3. Subtract points for grammar errors (e.g., 5 points per error)
    penalty = grammar_errors * 5
    final_score = final_score - penalty

    # 4. Ensure the score doesn't go below zero
    if final_score < 0:
        final_score = 0
        
    return "Passed", final_score


if __name__ == "__main__":
    # --- 3. DEFINE THE MODEL ANSWER ---
    model_answer = "A triangle is a polygon with three edges and three vertices. It is one of the basic shapes in geometry."

    # --- 4. PROCESS THE IMAGE ---
    original_image, student_answers_data = extract_text_blocks(IMAGE_PATH)

    if not student_answers_data:
        print("Could not find any text blocks in the image to grade.")
    else:
        print(f"\n--- Grading {len(student_answers_data)} Answer Block(s) ---")
        output_image = original_image.copy()

        # --- 5. GRADE EACH EXTRACTED ANSWER ---
        for i, (student_answer, coords) in enumerate(student_answers_data):
            cleaned_answer = student_answer.replace('\n', ' ').strip()
            
            # A. Calculate semantic score
            semantic_score = calculate_semantic_similarity(model_answer, cleaned_answer, sbert_model)
            
            # B. Analyze writing style
            grammar_errors, readability_grade = analyze_writing_style(cleaned_answer, grammar_tool)

            # --- MODIFIED: Call the new grading function ---
            final_verdict, final_grade = calculate_final_grade(semantic_score, grammar_errors)
            
            # --- 6. DISPLAY RESULTS ---
            print(f"\n--- Block {i+1} ---")
            print(f"Extracted Text: '{cleaned_answer}'")
            print(f"Semantic Score: {semantic_score:.4f}")
            print(f"Grammar/Spelling Errors: {grammar_errors}")
            print(f"Readability (Grade Level): {readability_grade}")
            # --- MODIFIED: Display the final calculated grade ---
            print(f"Verdict: {final_verdict}")
            print(f"Final Grade: {final_grade:.2f}%")
            
            # Determine grade and color for the bounding box based on the final verdict
            grade_text = f"{final_verdict}: {final_grade:.0f}%"
            box_color = (0, 0, 255) # Red for fail
            if final_verdict == "Passed":
                box_color = (0, 255, 0) # Green for pass

            # Draw results on the output image
            x, y, w, h = coords
            cv2.rectangle(output_image, (x, y), (x + w, y + h), box_color, 3)
            cv2.putText(output_image, grade_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        # Show the final graded image
        cv2.imshow('Graded Answer Sheet', output_image)
        print("\nPress any key in the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # Clean up the grammar tool server
    grammar_tool.close()