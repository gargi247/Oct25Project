import os
import json
import google.generativeai as genai
import time

INPUT_FILE = "mcqs_gemini.json"
OUTPUT_FILE = "mcqs_validated.json"

# Configure Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
validator = genai.GenerativeModel("gemini-2.5-pro")

# Load Gemini-generated results
with open(INPUT_FILE, "r") as f:
    gemini_data = json.load(f)

validated_output = []

print("\n Starting Gemini → Gemini validation (SNOMED enhancement)\n")

for idx, item in enumerate(gemini_data, start=1):
    print(f" Validating question {idx}/{len(gemini_data)}...")

    question = item["question"]
    options = item["options"]
    gemini_answer = item["gemini_answer"]
    justification = item["justification"]

    prompt = f"""
You are a medical expert with full SNOMED CT ontology awareness.

Your tasks:
1. Check if Gemini’s answer seems clinically correct (do NOT look at data labels).
2. Validate and fix the SNOMED concepts provided.
3. Add missing, highly relevant SNOMED concepts.
4. Correct any incorrect concept IDs or relationships.
5. For each concept, add the official SNOMED link:
   https://browser.ihtsdotools.org/?perspective=full&conceptId1=<ID>

6. Return ONLY valid JSON.
7. Do NOT reveal dataset answers.

JSON FORMAT:
{{
  "id": {item["id"]},
  "question": "{question}",
  "options": {{
    "A": "{options['A']}",
    "B": "{options['B']}",
    "C": "{options['C']}",
    "D": "{options['D']}"
  }},
  "gemini_answer": "{gemini_answer}",
  "validator_answer": "A/B/C/D",
  "is_consistent_with_reasoning": true/false,
  "enhanced_snomed_reasoning": {{
      "summary": "string",
      "concepts": [
        {{
          "name": "string",
          "snomed_id": "string",
          "link": "https://browser.ihtsdotools.org/?perspective=full&conceptId1=<ID>"
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

Gemini reasoning:
{json.dumps(justification, indent=2)}
"""

    try:
        response = validator.generate_content(prompt)
        text = response.text.strip()

        # Remove accidental backticks
        text = text.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(text)
        except:
            parsed = {
                "id": item["id"],
                "error": "Invalid JSON returned",
                "raw": text
            }

        validated_output.append(parsed)

        with open(OUTPUT_FILE, "w") as o:
            json.dump(validated_output, o, indent=2)

        print(f" Stored validation for question {idx}\n")
        time.sleep(1)

    except Exception as e:
        print(f" Error at question {idx}: {e}")
        validated_output.append({
            "id": item["id"],
            "error": str(e)
        })

        with open(OUTPUT_FILE, "w") as o:
            json.dump(validated_output, o, indent=2)


print("\n Validation done. Output saved → mcqs_validated.json\n")
