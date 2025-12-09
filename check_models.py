from google import generativeai as genai

genai.configure(api_key="AIzaSyD-uz8D5rYZxe2QXrSNCB_YHDvs9A0Rfy8")

# Convert generator to list to see available models
models = list(genai.list_models())
for m in models:
    print(m)
