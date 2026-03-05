import asyncio
import aiohttp
import pandas as pd

# Using the same model you pulled earlier
MODEL = "llama3.2:3b"
# Testing low, medium, and the "very-high" range
TEMPS = [0.1, 1.0, 2.0] 
TRIALS = 500
CONCURRENT_REQUESTS = 20 # Increased for 500 trials

# Experimenting with a "Scattergories" specific prompt
PROMPT = "We are playing Scattergories. Name a fruit that starts with the letter 'B'. Output only the one word."

async def fetch_fruit(session, semaphore, temp):
    async with semaphore:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": MODEL,
            "prompt": PROMPT,
            "stream": False,
            "options": {"temperature": temp, "seed": -1} 
        }
        try:
            async with session.post(url, json=payload, timeout=10) as response:
                data = await response.json()
                # Cleaning the output to keep it to one word
                return data['response'].strip().lower().split()[0].replace(".", "")
        except:
            return "error"

async def main():
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    results = []
    async with aiohttp.ClientSession() as session:
        for t in TEMPS:
            print(f"Running Scattergories Task at T={t}...")
            tasks = [fetch_fruit(session, semaphore, t) for _ in range(TRIALS)]
            answers = await asyncio.gather(*tasks)
            for a in answers:
                if a != "error":
                    results.append({"temperature": t, "fruit": a})
    
    df = pd.DataFrame(results)
    df.to_csv("fruit_results.csv", index=False)
    print(f"Done! Data saved to fruit_results.csv")

if __name__ == "__main__":
    asyncio.run(main())