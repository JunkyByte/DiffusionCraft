import subprocess
import os

# Function to run the generate.py script with given parameters
def run_experiment(prompt, prompt_idx, cond_images, cond_audio, cond_strength):
    outdir = f"./output/cond_strength/cond_strength_{cond_strength}/index_{prompt_idx}"
    command = [
        "python", "src/generate.py",
        "--steps", "75",
        "--outdir", outdir,
        "--prompt", prompt,
        "--n-iter", "8",
        "--cond-strength", str(cond_strength),
        "--alpha", "0.5"
    ]

    if len(cond_images):
        command.extend(["--cond-image", " ".join(cond_images)])
    if len(cond_audio):
        command.extend(["--cond-audio", " ".join(cond_audio)])

    subprocess.run(command)

# Read the input file
input_file = 'src/experiments/cond_strength.txt'  # change this to the path of your input file
with open(input_file, 'r') as file:
    lines = file.readlines()

# Iterate through each line in the input file
for ith, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue

    # Split the line into prompt and conditioning information
    parts = line.split('"')
    prompt = parts[1].strip()
    cond_files = parts[2].strip().split()

    # Separate image and audio conditioning files
    cond_images = [f for f in cond_files if not (f.endswith('.mp3') or f.endswith('.wav'))]
    cond_audio = [f for f in cond_files if f.endswith('.mp3') or f.endswith('.wav')]

    cond_images = [os.path.join('samples/imgs', f) for f in cond_images]
    cond_audio = [os.path.join('samples/audio', f) for f in cond_audio]

    # Run the experiments for different cond_strength values
    for cond_strength in [i * 0.25 for i in range(7)]:
        run_experiment(prompt, ith, cond_images, cond_audio, cond_strength)