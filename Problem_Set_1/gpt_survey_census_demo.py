#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import random
import os
import sys
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

# Create logs folder
os.makedirs('logs', exist_ok=True)

# Setup logging to capture all output
log_filename = f"logs/gpt_survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_filename, 'w')

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize the OpenAI client
# The client automatically reads the OPENAI_API_KEY environment variable.
client = OpenAI()

csv_file = 'post_strat_long_full_cartesian_FIXED_v2.csv'

# Function to generate GPT prompts based on census demographics
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

        You are acting as this person who can make mistakes, not an AI agent. Before you answer the question please consider the demographcis of this person.

        How would this person's age influence their grammar rules and how much they think about grammar?
        How does this person's education level influence their view on grammar and their experience with grammar?
        How does this person's income level influence their view on grammar?
        How does regionaly influenced dialects and grammar rules impact this person's view on grammar?

        Some people don't use the Oxford comma, or haven't heard of it. Some people have.

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
                    {"role": "user", "content": gpt_prompts[i]}
                ],
                max_tokens=350,  # Adjusted to ensure full list of answers fits
                n=1,  # Number of responses to generate
                temperature=1.4,  # Adjust for response variability
            )
            # Strip whitespace to clean the result
            content = response.choices[0].message.content.strip()
            responses.append(content)
            print(f"Response {i+1} received.")
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            responses.append("") # Append empty string to keep index alignment
            
    return responses


# Load census data for sampling
census_data = []
try:
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            census_data.append(row)
except FileNotFoundError:
    print(f"Error: Could not find {csv_file}. Make sure it is in the same folder.")
    exit()

import numpy as np
# Sample rows according to pop_count weights
num_responses = 300
weights = [float(row.get('pop_count', 1)) for row in census_data]
total_weight = sum(weights)
probabilities = [w / total_weight for w in weights]
selected_indices = np.random.choice(len(census_data), size=num_responses, replace=True, p=probabilities)
selected_data = [census_data[i] for i in selected_indices]

# Generate GPT prompts based on the demographics
print(f"Generating GPT prompts...")
gpt_prompts = generate_gpt_prompts(selected_data)
print(f"Generated {len(gpt_prompts)} prompts.")

# Poll GPT for survey responses
print(f"Polling GPT for survey responses...")
gpt_responses = poll_gpt(gpt_prompts, len(selected_data))
print(f"Received {len(gpt_responses)} GPT responses.")

# Save the responses to a CSV file after processing them
output_filename = 'gpt_census_demo_comma_survey.csv'

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
    for i, raw_response in tqdm(enumerate(gpt_responses), total=len(gpt_responses), desc="Writing responses"):
        if not raw_response:
            print(f"Skipping row {i} due to failed API call.")
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
        original_demo = selected_data[i]
        # Map census columns to expected survey columns
        demo_map = {
            'Gender': original_demo.get('sex', ''),
            'Age': original_demo.get('age_group', ''),
            'Household Income': original_demo.get('income_bin', ''),
            'Education': original_demo.get('education_5', ''),
            'Location (Census Region)': original_demo.get('census_region', '')
        }

        # 3. Construct the row
        # We generate a fake ID, add the 7 GPT answers, then the 5 mapped demographics
        row_to_write = [
            str(1000000000 + i), # Fake RespondentID
            gpt_answers[0],
            gpt_answers[1],
            gpt_answers[2],
            gpt_answers[3],
            gpt_answers[4],
            gpt_answers[5],
            gpt_answers[6],
            demo_map['Gender'],
            demo_map['Age'],
            demo_map['Household Income'],
            demo_map['Education'],
            demo_map['Location (Census Region)']
        ]

        writer.writerow(row_to_write)

print("Done.")



