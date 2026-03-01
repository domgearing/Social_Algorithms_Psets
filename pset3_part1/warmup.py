import asyncio
import aiohttp
import pandas as pd

MODEL = "llama3.2:3b"
TEMPS = [0.1, 0.5, 1.0, 1.5, 2.0]
TRIALS = 500
CONCURRENT_REQUESTS = 5 # Slow and steady to avoid the T=1.5 crash
PROMPT = "Pick a random day of the week. Output only the name of the day."

async def fetch_answer(session, semaphore, temp):
    async with semaphore:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": MODEL,
            "prompt": PROMPT,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": 5, # Prevents server hang at high temps
                "seed": -1        # Forces new randomness
            }
        }
        try:
            async with session.post(url, json=payload, timeout=30) as response:
                data = await response.json()
                return data['response'].strip().lower().replace(".", "")
        except:
            return "error"

async def run_sweep():
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    all_data = []
    async with aiohttp.ClientSession() as session:
        for t in TEMPS:
            print(f"Running Temperature {t}...")
            tasks = [fetch_answer(session, semaphore, t) for _ in range(TRIALS)]
            answers = await asyncio.gather(*tasks)
            for ans in answers:
                if ans != "error":
                    all_data.append({"temperature": t, "answer": ans})
    
    pd.DataFrame(all_data).to_csv("day_of_week_results.csv", index=False)
    print("Finished! Now run analyze.py")

if __name__ == "__main__":
    asyncio.run(run_sweep())