import subprocess
from datasets import load_dataset

# Load dataset
dataset = load_dataset("HiTZ/This-is-not-a-dataset")

# Convert to pandas for easy iteration
df = dataset["train"].to_pandas()

#------------------------------------------------------------------------------------------------------

# Ollama run function
def run_ollama(model, prompt):
    """Run an Ollama model with a given prompt and return output."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        capture_output=True
    )
    return result.stdout.decode().strip()

#------------------------------------------------------------------------------------------------------

# Evaluation function
def evaluate_detector(df, model="detector", limit=20):
    counts = {"TT":0, "TF":0, "FT":0, "FF":0}
    
    for i, row in df.iterrows():
        if i >= limit:
            break
        
        sentence = row["sentence"]
        label = int(row["label"])  # 1 = True, 0 = False
        
        verdict = run_ollama(model, f"Classify as TRUE or FAKE and explain: {sentence}")
        
        if verdict.upper().startswith("TRUE"):
            pred = 1
        else:
            pred = 0
        
        # Update counts
        if label == 1 and pred == 1:
            counts["TT"] += 1
        elif label == 1 and pred == 0:
            counts["TF"] += 1
        elif label == 0 and pred == 1:
            counts["FT"] += 1
        elif label == 0 and pred == 0:
            counts["FF"] += 1
        
        # Show progress
        print(f"[{i+1}] {sentence[:60]}... → {verdict}")
    
    return counts
#------------------------------------------------------------------------------------------------------

# Run functions
results = evaluate_detector(df, limit=10)
print("Results:", results)
