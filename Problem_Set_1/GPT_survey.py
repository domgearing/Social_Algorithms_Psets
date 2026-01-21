#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import random
from openai import OpenAI

# Initialize the OpenAI client
# The client automatically reads the OPENAI_API_KEY environment variable.
client = OpenAI()

csv_file = 'comma-survey.csv'

# Function to generate GPT prompts based on demographics
def generate_gpt_prompts(data):
    prompts = []
    for row in data:
        # Extract demographic variables safely
        age = row.get('Age', 'Unknown')
        gender = row.get('Gender', 'Unknown')
        income = row.get('Household Income', 'Unknown')
        education = row.get('Education', 'Unknown')
        location = row.get('Location (Census Region)', 'Unknown')
        
        # Define the survey prompt
        # We explicitly list the 7 substantive questions from the original dataset
        # and instruct GPT to output in a format easy to parse (a numbered list).
        prompt = f"""
        You are a survey participant with the following demographics:
        - Age: {age}
        - Gender: {gender}
        - Household Income: {income}
        - Education: {education}
        - Location: {location}

        You are invited to participate in a survey about grammar usage.
        Please answer the following 7 questions as if you are this person.
        
        Questions:
        1. In your opinion, which sentence is more gramatically correct?
           Options: ["It's important for a person to be honest, kind and loyal.", "It's important for a person to be honest, kind, and loyal."]
        2. Prior to reading about it above, had you heard of the serial (or Oxford) comma?
           Options: ["Yes", "No"]
        3. How much, if at all, do you care about the use (or lack thereof) of the serial (or Oxford) comma in grammar?
           Options: ["A lot", "Some", "Not much", "Not at all"]
        4. How would you write the following sentence?
           Options: ["Some experts say it's important to drink milk, but the data are inconclusive.", "Some experts say it's important to drink milk, but the data is inconclusive."]
        5. When faced with using the word "data", have you ever spent time considering if the word was a singular or plural noun?
           Options: ["Yes", "No"]
        6. How much, if at all, do you care about the debate over the use of the word "data" as a singluar or plural noun?
           Options: ["A lot", "Some", "Not much", "Not at all"]
        7. In your opinion, how important or unimportant is proper use of grammar?
           Options: ["Very important", "Somewhat important", "Somewhat unimportant", "Very unimportant"]

        **Output Instructions:** Provide ONLY the selected answer text for each question in a numbered list. 
        Do not add explanations. Example:
        1. Answer One
        2. Answer Two
        ...
        """
        prompts.append(prompt)
    return prompts

# Function to poll GPT
def poll_gpt(gpt_prompts, num_responses):
    responses = []
    print(f"Polling GPT for {num_responses} responses...")
    for i in range(num_responses):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective model for this assignment
                messages=[
                    {"role": "system", "content": "You are a helpful survey participant."},
                    {"role": "user", "content": gpt_prompts[i]}
                ],
                max_tokens=350,  # Adjusted to ensure full list of answers fits
                n=1,  # Number of responses to generate
                temperature=1.0,  # Adjust for response variability
            )
            # Strip whitespace to clean the result
            content = response.choices[0].message.content.strip()
            responses.append(content)
            print(f"Response {i+1} received.")
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            responses.append("") # Append empty string to keep index alignment
            
    return responses

# Load survey data from csv
survey_data = []
try:
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            survey_data.append(row)
except FileNotFoundError:
    print(f"Error: Could not find {csv_file}. Make sure it is in the same folder.")
    exit()

# Randomly select rows
num_responses = 300 
if len(survey_data) < num_responses:
    selected_data = survey_data
else:
    selected_data = random.sample(survey_data, num_responses)

# Generate GPT prompts based on the demographics
gpt_prompts = generate_gpt_prompts(selected_data)

# Poll GPT for survey responses
gpt_responses = poll_gpt(gpt_prompts, len(selected_data))

# Save the responses to a CSV file after processing them
output_filename = 'gpt_comma_survey.csv'

# Define the exact headers from the original dataset
headers = [
    'RespondentID',
    'In your opinion, which sentence is more gramatically correct?',
    'Prior to reading about it above, had you heard of the serial (or Oxford) comma?',
    'How much, if at all, do you care about the use (or lack thereof) of the serial (or Oxford) comma in grammar?',
    'How would you write the following sentence?',
    'When faced with using the word "data", have you ever spent time considering if the word was a singular or plural noun?',
    'How much, if at all, do you care about the debate over the use of the word "data" as a singluar or plural noun?',
    'In your opinion, how important or unimportant is proper use of grammar?',
    'Gender', 'Age', 'Household Income', 'Education',
    'Location (Census Region)'
]

print(f"Writing data to {output_filename}...")

with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers) 
    
    # Iterate through responses and the original selected_data in parallel
    for i, raw_response in enumerate(gpt_responses):
        if not raw_response:
            continue # Skip failed API calls

        # 1. Parse the GPT response (assumed numbered list format)
        # We split by newlines, clean up, and remove the "1. " numbering
        gpt_answers = []
        lines = raw_response.split('\n')
        for line in lines:
            line = line.strip()
            # If line starts with a digit and a dot (e.g., "1. "), strip it
            if len(line) > 0 and line[0].isdigit():
                parts = line.split(' ', 1)
                if len(parts) > 1:
                    gpt_answers.append(parts[1].strip())
        
        # Ensure we have 7 answers (padding with empty string if GPT hallucinated/failed)
        while len(gpt_answers) < 7:
            gpt_answers.append("")

        # 2. Get the demographics from the ORIGINAL selected row
        # This ensures the demographics match the person we simulated
        original_demo = selected_data[i]
        
        # 3. Construct the row
        # We generate a fake ID, add the 7 GPT answers, then the 5 demographics
        row_to_write = [
            str(1000000000 + i), # Fake RespondentID
            gpt_answers[0],
            gpt_answers[1],
            gpt_answers[2],
            gpt_answers[3],
            gpt_answers[4],
            gpt_answers[5],
            gpt_answers[6],
            original_demo.get('Gender', ''),
            original_demo.get('Age', ''),
            original_demo.get('Household Income', ''),
            original_demo.get('Education', ''),
            original_demo.get('Location (Census Region)', '')
        ]
        
        writer.writerow(row_to_write)

print("Done.")

