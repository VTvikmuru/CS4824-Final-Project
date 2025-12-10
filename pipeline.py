import subprocess

# Run an Ollama model with a prompt and return its output
def run_ollama(model, prompt):
    
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        capture_output=True
    )
    return result.stdout.decode().strip()

def main():
    # Generate hallucinated statement
    statement = run_ollama("generator", "Write a fake news style sentence.")
    print("Generated:", statement)

    # Detect truthfulness with explanation
    verdict = run_ollama("detector", f"Classify as TRUE or FAKE and explain: {statement}")
    print("Detector verdict:", verdict)

    # output
    print(f"\nStatement: {statement}\nVerdict & Explanation: {verdict}")

if __name__ == "__main__":
    main()
    