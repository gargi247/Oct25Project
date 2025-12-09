import os
import json
import time
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-pro")

# Load data
OUTPUT_PATH = "mcqs_gemini.json"
DATA_PATH = "data_clean/questions/US/4_options/phrases_no_exclude_train.jsonl"

with open(DATA_PATH, "r") as f:
    lines = f.readlines()

samples = [json.loads(line) for line in lines[:20]]

# Prepare output
if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

results = []
print(" Starting Gemini batch reasoning for first 20 questions...\n")

# Loop through each question
for idx, sample in enumerate(samples, start=1):
    question = sample["question"]
    options = sample["options"]

    print(f"\n Processing Question {idx}/20...")

    prompt = f"""
You are a medical reasoning system grounded in SNOMED CT ontology.

Your task:
1. Identify the correct answer purely based on clinical reasoning and SNOMED CT concept hierarchy.
2. Provide structured justification including relevant SNOMED concepts and relationships.
3. Include SNOMED CT concept IDs and official browser links.
4. Return only valid JSON (no Markdown, no explanations).

Expected JSON format:
{{
  "id": {idx},
  "question": "string",
  "options": {{
    "A": "string",
    "B": "string",
    "C": "string",
    "D": "string"
  }},
  "gemini_answer": "A/B/C/D",
  "justification": {{
    "reasoning_summary": "string",
    "snomed_concepts": [
      {{
        "name": "string",
        "snomed_id": "string",
        "link": "https://browser.ihtsdotools.org/?perspective=full&conceptId1=<id>"
      }}
    ]
  }}
}}

Question:
{question}

Options:
A. {options['A']}
B. {options['B']}
C. {options['C']}
D. {options['D']}
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip().replace("```json", "").replace("```", "").strip()

        # Parse Gemini response
        try:
            result_json = json.loads(text)
        except json.JSONDecodeError:
            print(f" Question {idx}: Invalid JSON. Saving fallback.")
            result_json = {
                "id": idx,
                "question": question,
                "options": options,
                "gemini_answer": None,
                "justification": {
                    "reasoning_summary": "Invalid JSON response",
                    "raw_response": text
                }
            }

        results.append(result_json)

        # Save incrementally (to avoid losing progress)
        with open(OUTPUT_PATH, "w") as outfile:
            json.dump(results, outfile, indent=2)

        print(f" Saved Question {idx} successfully.")
        time.sleep(2)

    except Exception as e:
        print(f" Error at Question {idx}: {e}")
        results.append({
            "id": idx,
            "question": question,
            "options": options,
            "gemini_answer": None,
            "justification": {
                "reasoning_summary": f"Error: {str(e)}"
            }
        })
        with open(OUTPUT_PATH, "w") as outfile:
            json.dump(results, outfile, indent=2)

print("\n All 20 questions processed and stored in mcqs_gemini.json")
