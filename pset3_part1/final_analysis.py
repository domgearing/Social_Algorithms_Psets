import pandas as pd

df = pd.read_csv('fruit_results.csv')

# validation list based on common 'B' fruits
VALID_B_FRUITS = [
    'banana', 'blueberry', 'blackberry', 'blood orange', 'boysenberry', 
    'bilberry', 'barberry', 'breadfruit', 'black currant', 'babaço'
]

def get_metrics(temp_df):
    total = len(temp_df)
    # Check if the response is in our valid list
    valid_count = temp_df['fruit'].isin(VALID_B_FRUITS).sum()
    validity_rate = (valid_count / total) * 100
    unique_count = temp_df['fruit'].nunique()
    return validity_rate, unique_count

print("--- Scattergories Performance Report ---")
for t in sorted(df['temperature'].unique()):
    subset = df[df['temperature'] == t]
    v_rate, u_count = get_metrics(subset)
    print(f"Temp {t}: Validity = {v_rate:.1f}% | Unique Words = {u_count}")