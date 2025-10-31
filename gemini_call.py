import os
import json
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.5-pro")

# Load a few MedQA samples
with open("data_clean/questions/US/4_options/phrases_no_exclude_train.jsonl", "r") as f:
    lines = f.readlines()

# Pick one sample for testing
sample = json.loads(lines[0])
question = sample["question"]
options = sample["options"]
answer = sample["answer"]

prompt = f"""
You are a clinical reasoning expert specializing in SNOMED CT ontology.
Analyze the following medical question using SNOMED CT concepts and relationships 
(such as 'is-a', 'associated finding', 'causative agent', 'morphologic abnormality', etc.)
to arrive at the most accurate answer.

Question: {question}

Options:
A. {options['A']}
B. {options['B']}
C. {options['C']}
D. {options['D']}

For each option:
1. Identify the relevant SNOMED CT concepts.
2. Describe their hierarchical relationships and clinical meaning.
3. Explain how these concepts relate to the question.
4. Conclude with the most likely correct answer and provide reasoning 
   grounded in SNOMED CT structure.

Format the output as:
- Concept mapping (for each option)
- SNOMED relationships used
- Step-by-step reasoning
- Final answer
"""

# Send to Gemini
response = model.generate_content(prompt)
print("Gemini Response:\n")
print(response.text)
print("\nCorrect answer from dataset:", answer)
