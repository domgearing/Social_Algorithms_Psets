import asyncio
import aiohttp
import pandas as pd

MODEL = "llama3.2:3b"
TEMP = 1.0 # Keep temperature constant to isolate prompt impact
TRIALS = 100 

# Two different prompt strategies
PROMPTS = {
    "no_context": "Name a fruit that starts with the letter 'B'. Output only the one word.",
    "with_example": "We are playing Scattergories. Name a fruit that starts with the letter 'B' (e.g., Blueberry). Output only the one word."
}

async def fetch_variant(session, prompt_text):
    url = "http://localhost:11434/api/generate"
    payload = {"model": MODEL, "prompt": prompt_text, "stream": False, "options": {"temperature": TEMP, "seed": -1}}
    async with session.post(url, json=payload) as response:
        data = await response.json()
        return data['response'].strip().lower().split()[0].replace(".", "")

async def main():
    results = []
    async with aiohttp.ClientSession() as session:
        for label, text in PROMPTS.items():
            print(f"Testing prompt: {label}...")
            tasks = [fetch_variant(session, text) for _ in range(TRIALS)]
            answers = await asyncio.gather(*tasks)
            for a in answers:
                results.append({"prompt_type": label, "fruit": a})
    
    pd.DataFrame(results).to_csv("prompt_comparison.csv", index=False)
    print("Test complete! Check prompt_comparison.csv")

if __name__ == "__main__":
    asyncio.run(main())