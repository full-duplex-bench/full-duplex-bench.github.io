import os
import re
import shutil
import subprocess
from pathlib import Path
import wave
import numpy as np
import audioop

# Define main directories
source_dir = "audios"  # Your current root directory with the audio files
target_dir = "audio"   # Your new root directory for the organized structure

# Check if ffmpeg is available
def is_ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except FileNotFoundError:
        return False

# Create the new directory structure
def create_directory_structure():
    categories = ["pause", "backchannel", "turntaking", "interruption"]
    datasets = {"pause": ["candor", "synthetic"],
               "backchannel": ["icc"], 
               "turntaking": ["candor"],
               "interruption": ["synthetic"]}
    models = ["dGSLM", "moshi", "freezeomni"]
    
    # Create the root directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Create the structure
    for category in categories:
        category_dir = os.path.join(target_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for dataset in datasets[category]:
            dataset_dir = os.path.join(category_dir, dataset)
            os.makedirs(dataset_dir, exist_ok=True)
            
            for model in models:
                model_dir = os.path.join(dataset_dir, model)
                os.makedirs(model_dir, exist_ok=True)
                
    print("Directory structure created.")

# Extract sample ID from directory name
def extract_sample_id(directory_name):
    # Find all directories that match UUID pattern
    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    if re.match(uuid_pattern, directory_name):
        return directory_name[:8]  # Take first 8 chars of UUID as sample ID
    return None

# Combine two mono WAV files into one stereo WAV file using the wave module
def combine_wav_files(left_file, right_file, output_file):
    try:
        # Open the input files
        with wave.open(left_file, 'rb') as left_wav, wave.open(right_file, 'rb') as right_wav:
            # Check if the files are compatible
            if left_wav.getframerate() != right_wav.getframerate():
                print(f"Sample rates don't match: {left_file} ({left_wav.getframerate()} Hz) vs {right_file} ({right_wav.getframerate()} Hz)")
                return False
                
            # Create a new stereo WAV file
            with wave.open(output_file, 'wb') as output_wav:
                output_wav.setnchannels(2)
                output_wav.setsampwidth(left_wav.getsampwidth())
                output_wav.setframerate(left_wav.getframerate())
                
                # Calculate the number of frames to read
                n_frames = min(left_wav.getnframes(), right_wav.getnframes())
                
                # Read all frames from both files and interleave them
                for i in range(0, n_frames, 1024):
                    frames_to_read = min(1024, n_frames - i)
                    left_data = left_wav.readframes(frames_to_read)
                    right_data = right_wav.readframes(frames_to_read)
                    
                    # Ensure same length for both channels
                    if len(left_data) != len(right_data):
                        # Pad the shorter one
                        if len(left_data) < len(right_data):
                            left_data += b'\x00' * (len(right_data) - len(left_data))
                        else:
                            right_data += b'\x00' * (len(left_data) - len(right_data))
                    
                    # Interleave the channels
                    stereo_data = audioop.tomono(left_data, left_wav.getsampwidth(), 1, 0)
                    stereo_data = audioop.tostereo(stereo_data, left_wav.getsampwidth(), 1, 1)
                    output_wav.writeframes(stereo_data)
                
        print(f"Created {output_file}")
        return True
    except Exception as e:
        print(f"Error combining files: {e}")
        return False

# Process audio files for a specific model directory
def process_audio_files(model_name, source_path, target_base_path, category, dataset):
    # Map model names to the correct directory names
    model_dir_map = {
        "dGSLM": "dgslm",
        "moshi": "moshi",
        "freezeomni": "freeze_omni"
    }
    
    # Define which files to combine based on model
    file_combinations = {
        "dGSLM": ("input.wav", "dgslm_output_mono.wav"),
        "moshi": ("input.wav", "moshi_output_mono.wav"),
        "freezeomni": ("input.wav", "output.wav")
    }
    
    source_model_dir = os.path.join(source_path, model_dir_map[model_name])
    target_model_dir = os.path.join(target_base_path, category, dataset, model_name)
    
    if not os.path.exists(source_model_dir):
        print(f"Source directory not found: {source_model_dir}")
        return
    
    # Process each sample directory
    sample_count = 1
    for item in os.listdir(source_model_dir):
        item_path = os.path.join(source_model_dir, item)
        
        # If it's a directory that matches our UUID pattern
        if os.path.isdir(item_path) and extract_sample_id(item):
            sample_id = extract_sample_id(item)
            
            # Define input files
            left_channel = os.path.join(item_path, file_combinations[model_name][0])
            right_channel = os.path.join(item_path, file_combinations[model_name][1])
            
            # Check if both files exist
            if os.path.exists(left_channel) and os.path.exists(right_channel):
                output_file = os.path.join(target_model_dir, f"sample_{sample_count}.wav")
                
                # Try to combine files
                success = False
                
                # Use ffmpeg if available
                if is_ffmpeg_available():
                    try:
                        cmd = [
                            "ffmpeg", 
                            "-i", left_channel,
                            "-i", right_channel,
                            "-filter_complex", "[0:a][1:a]amerge=inputs=2[aout]",
                            "-map", "[aout]",
                            "-ac", "2",
                            output_file,
                            "-y"  # Overwrite output file if it exists
                        ]
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        success = True
                        print(f"Created {output_file} using ffmpeg")
                    except subprocess.CalledProcessError as e:
                        print(f"ffmpeg error: {e}")
                        success = False
                
                # Fall back to Python's wave module if ffmpeg fails or is not available
                if not success:
                    success = combine_wav_files(left_channel, right_channel, output_file)
                
                if success:
                    sample_count += 1
            else:
                missing_files = []
                if not os.path.exists(left_channel):
                    missing_files.append(file_combinations[model_name][0])
                if not os.path.exists(right_channel):
                    missing_files.append(file_combinations[model_name][1])
                print(f"Missing files for {item_path}: {', '.join(missing_files)}")

# Main function
def main():
    # Create the new directory structure
    create_directory_structure()
    
    # Process candor_pause directory for each model
    source_candor_pause = os.path.join(source_dir, "candor_pause")
    
    if os.path.exists(source_candor_pause):
        # Process each model's files
        process_audio_files("dGSLM", source_candor_pause, target_dir, "pause", "candor")
        process_audio_files("moshi", source_candor_pause, target_dir, "pause", "candor")
        process_audio_files("freezeomni", source_candor_pause, target_dir, "pause", "candor")
        
        print("Processing completed for candor_pause directory.")
    else:
        print(f"Source directory not found: {source_candor_pause}")
    
    # You can add processing for other categories and datasets here similarly
    
if __name__ == "__main__":
    main()