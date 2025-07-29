import csv
import cohere
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import re

# Initialize Cohere client with your API key
try:
    co = cohere.Client("your api key ")
except Exception as e:
    print(f"Error initializing Cohere client: {e}")
    sys.exit(1)

# Load CSV file
def load_qa_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'prompt' not in df.columns or 'response' not in df.columns:
            raise ValueError("CSV file must contain 'prompt' and 'response' columns")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

# Find similar prompt using TF-IDF
def find_similar_prompt(user_input, prompts):
    try:
        vectorizer = TfidfVectorizer().fit_transform([user_input] + prompts)
        vectors = vectorizer.toarray()
        cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        most_similar_idx = np.argmax(cosine_similarities)
        return most_similar_idx, cosine_similarities[most_similar_idx]
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        return None, 0.0

# Check if response contains all required sections
def is_complete_response(response):
    required_sections = [r"1\)\s*Symptoms:", r"2\)\s*Treatment:", r"3\)\s*Prevention:", r"4\)\s*Care Measures:"]
    return all(re.search(section, response) for section in required_sections)

# Generate response using Cohere
def generate_cohere_response(user_input):
    try:
        # Refined prompt to enforce all four sections for disease-related queries
        structured_prompt = f"""
        You are a medical chatbot. If the user's input is related to a disease or medical condition, you MUST respond in the following format, including all four sections with detailed information:
        1) Symptoms: [List the symptoms of the disease in a clear, concise manner]
        2) Treatment: [Describe the treatment options, including medications or therapies]
        3) Prevention: [Explain preventive measures to avoid the disease]
        4) Care Measures: [Provide specific care measures for managing the condition]

        Each section must be non-empty and relevant. Do not skip any section. If the input is not related to a disease or medical condition, provide a concise, relevant response without using the above format.

        User Input: {user_input}
        Response:
        """
        response = co.generate(
            prompt=structured_prompt,
            max_tokens=300,  # Increased to ensure all sections are included
            temperature=0.7,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
        generated_text = response.generations[0].text.strip()

        # Check if response is complete; if not, append default care measures for disease-related queries
        if any(keyword in user_input.lower() for keyword in ['disease', 'condition', 'covid', 'diabetes', 'cancer', 'flu']) and not is_complete_response(generated_text):
            generated_text += "\n4) Care Measures: Monitor symptoms, consult a healthcare provider, stay hydrated, and rest adequately."
        
        return generated_text
    except Exception as e:
        return f"Error generating response: {e}"

# Main chatbot function
def chatbot(file_path):
    df = load_qa_data(file_path)
    if df is None:
        print("Cannot proceed without valid CSV data.")
        return

    prompts = df['prompt'].tolist()
    responses = df['response'].tolist()

    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            # Find similar prompt in CSV
            most_similar_idx, similarity_score = find_similar_prompt(user_input, prompts)
            if most_similar_idx is None:
                response = generate_cohere_response(user_input)
            else:
                # If similarity is high (>0.7), use CSV response; otherwise, use Cohere
                if similarity_score > 0.7:
                    response = responses[most_similar_idx]
                    # Ensure CSV response follows format for disease-related queries
                    if any(keyword in user_input.lower() for keyword in ['disease', 'condition', 'covid', 'diabetes', 'cancer', 'flu']) and not is_complete_response(response):
                        response = generate_cohere_response(user_input)
                else:
                    response = generate_cohere_response(user_input)
            
            print(f"Bot: {response}")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"Error during chat: {e}")

# Run the chatbot
if __name__ == "__main__":
    csv_file_path = "data.csv"
    chatbot(csv_file_path)