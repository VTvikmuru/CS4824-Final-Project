# Code to pregenerate samples

numberOfSamples = 100

import subprocess

# Ollama generator wrapper
def run_ollama(model, prompt):
    """Run an Ollama model with a given prompt and return output."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        capture_output=True
    )
    return result.stdout.decode().strip()

# Generate fake samples
def generate_fake_samples(num_samples=100):
    fakes = []
    for i in range(num_samples):
        fake = run_ollama("generator", "Write false truth sentences. One example is 'An introduction is commonly the first section of a communication.'")
        fakes.append(fake)
        if (i+1) % 10 == 0 or i == num_samples-1:
            print(f"Generated {i+1}/{num_samples} samples")
    return fakes

# Save to text file
def save_samples_to_file(samples, filename="generated_fakes.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s + "\n")
    print(f"Saved {len(samples)} samples to {filename}")

# Run once
if __name__ == "__main__":
    samples = generate_fake_samples(num_samples=numberOfSamples)
    save_samples_to_file(samples, "generated_fakes.txt")
