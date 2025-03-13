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
        # Since we can't easily resample with pure Python, we'll just use a subprocess
        # call to sox or ffmpeg if available
        if is_ffmpeg_available():
            cmd = [
                "ffmpeg", 
                "-i", left_file,
                "-i", right_file,
                "-filter_complex", "[0:a][1:a]amerge=inputs=2[aout]",
                "-map", "[aout]",
                "-ac", "2",
                output_file,
                "-y"  # Overwrite output file if it exists
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Created {output_file} using ffmpeg")
            return True
            
        # Check if sox is available as an alternative
        try:
            subprocess.run(["sox", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            cmd = [
                "sox", "-M", 
                left_file,
                right_file,
                output_file
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Created {output_file} using sox")
            return True
        except FileNotFoundError:
            print("Neither ffmpeg nor sox is available. Attempting to use Python's wave module.")
        
        # Open the input files
        with wave.open(left_file, 'rb') as left_wav, wave.open(right_file, 'rb') as right_wav:
            # Get parameters from both files
            left_rate = left_wav.getframerate()
            right_rate = right_wav.getframerate()
            left_width = left_wav.getsampwidth()
            right_width = right_wav.getsampwidth()
            left_channels = left_wav.getnchannels()
            right_channels = right_wav.getnchannels()
            
            # Check compatibility and warn about limitations
            if left_rate != right_rate:
                print(f"Warning: Sample rates don't match: {left_file} ({left_rate} Hz) vs {right_file} ({right_rate} Hz)")
                print("Using the sample rate of the first file. This may cause audio distortion.")
            
            if left_width != right_width:
                print(f"Warning: Sample widths don't match: {left_file} ({left_width} bytes) vs {right_file} ({right_width} bytes)")
                print("Using the sample width of the first file. This may cause audio distortion.")
            
            if left_channels != 1 or right_channels != 1:
                print(f"Warning: Input files should be mono: {left_file} ({left_channels} channels) vs {right_file} ({right_channels} channels)")
                print("Attempting to convert to mono before merging.")
                
            # Create a new stereo WAV file
            with wave.open(output_file, 'wb') as output_wav:
                output_wav.setnchannels(2)
                output_wav.setsampwidth(left_width)
                output_wav.setframerate(left_rate)
                
                # Calculate the number of frames to read
                n_frames = min(left_wav.getnframes(), right_wav.getnframes())
                
                # Read all frames from both files
                left_data = left_wav.readframes(left_wav.getnframes())
                right_data = right_wav.readframes(right_wav.getnframes())
                
                # Convert to mono if needed
                if left_channels != 1:
                    left_data = audioop.tomono(left_data, left_width, 1, 0)
                if right_channels != 1:
                    right_data = audioop.tomono(right_data, right_width, 1, 0)
                
                # Resample if needed (this is a very crude resampling and will affect quality)
                if left_rate != right_rate:
                    # Resample the right channel to match the left channel's rate
                    right_data = audioop.ratecv(right_data, right_width, 1, right_rate, left_rate, None)[0]
                
                # Ensure same length for both channels
                if len(left_data) != len(right_data):
                    # Pad the shorter one
                    if len(left_data) < len(right_data):
                        left_data += b'\x00' * (len(right_data) - len(left_data))
                    else:
                        right_data += b'\x00' * (len(left_data) - len(right_data))
                
                # Mix the two channels into a stereo file
                stereo_data = audioop.tostereo(left_data, left_width, 1, 1)
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
        "freezeomni": "freeze_omni",
        "candor_turn_dgslm_moshi": "candor_turn_dgslm_moshi",
        "candor_turn_freeze_omni": "candor_turn_freeze_omni"
    }
    
    # Define which files to combine based on model and category
    file_combinations = {
        # For pause category
        "pause": {
            "dGSLM": ("input.wav", "dgslm_output_mono.wav"),
            "moshi": ("input.wav", "moshi_output_mono.wav"),
            "freezeomni": ("input.wav", "output.wav")
        },
        # For turntaking category
        "turntaking": {
            "dGSLM": None,  # Special case, no concatenation needed
            "moshi": ("input.wav", "moshi_out_turn_taking.wav"),
            "freezeomni": ("input.wav", "output.wav")
        }
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
            output_file = os.path.join(target_model_dir, f"sample_{sample_count}.wav")
            
            # Special case for dGSLM in turntaking - just copy the stereo file
            if category == "turntaking" and model_name == "dGSLM":
                stereo_file = os.path.join(item_path, "dgslm_output_stereo.wav")
                if os.path.exists(stereo_file):
                    shutil.copy2(stereo_file, output_file)
                    print(f"Copied {stereo_file} to {output_file}")
                    sample_count += 1
                else:
                    print(f"Missing stereo file for {item_path}: dgslm_output_stereo.wav")
                continue
                
            # For other cases, get the file combination based on category and model
            file_combo = file_combinations.get(category, {}).get(model_name)
            if not file_combo:
                print(f"No file combination defined for {category}/{model_name}")
                continue
                
            left_channel = os.path.join(item_path, file_combo[0])
            right_channel = os.path.join(item_path, file_combo[1])
            
            # Check if both files exist
            if os.path.exists(left_channel) and os.path.exists(right_channel):
                # Combine the files
                success = combine_wav_files(left_channel, right_channel, output_file)
                
                if success:
                    sample_count += 1
            else:
                missing_files = []
                if not os.path.exists(left_channel):
                    missing_files.append(file_combo[0])
                if not os.path.exists(right_channel):
                    missing_files.append(file_combo[1])
                print(f"Missing files for {item_path}: {', '.join(missing_files)}")

# Process turntaking models with different directory structure
def process_turntaking_models(target_base_path):
    # Process candor_turn_taking/candor_turn_dgslm_moshi directory
    source_dir_path = os.path.join(source_dir, "candor_turn_taking") 
    source_dgslm_moshi = os.path.join(source_dir_path, "candor_turn_dgslm_moshi")
    
    if os.path.exists(source_dgslm_moshi):
        # For dGSLM and moshi models in the same directory
        sample_count = 1
        target_dgslm_dir = os.path.join(target_base_path, "turntaking", "candor", "dGSLM")
        target_moshi_dir = os.path.join(target_base_path, "turntaking", "candor", "moshi")
        
        # Create the target directories if they don't exist
        os.makedirs(target_dgslm_dir, exist_ok=True)
        os.makedirs(target_moshi_dir, exist_ok=True)
        
        for item in os.listdir(source_dgslm_moshi):
            item_path = os.path.join(source_dgslm_moshi, item)
            
            # If it's a directory that matches our UUID pattern
            if os.path.isdir(item_path) and extract_sample_id(item):
                # Process dGSLM - copy stereo file
                dgslm_stereo = os.path.join(item_path, "dgslm_output_stereo.wav")
                if os.path.exists(dgslm_stereo):
                    output_file = os.path.join(target_dgslm_dir, f"sample_{sample_count}.wav")
                    shutil.copy2(dgslm_stereo, output_file)
                    print(f"Copied {dgslm_stereo} to {output_file}")
                else:
                    print(f"Missing stereo file for {item_path}: dgslm_output_stereo.wav")
                
                # Process moshi - just copy moshi_out_turn_taking.wav directly
                moshi_wav = os.path.join(item_path, "moshi_out_turn_taking.wav")
                
                if os.path.exists(moshi_wav):
                    output_file = os.path.join(target_moshi_dir, f"sample_{sample_count}.wav")
                    shutil.copy2(moshi_wav, output_file)
                    print(f"Copied {moshi_wav} to {output_file}")
                else:
                    print(f"Missing file for moshi in {item_path}: moshi_out_turn_taking.wav")
                
                sample_count += 1
        
        print("Processing completed for candor_turn_dgslm_moshi directory.")
    else:
        print(f"Source directory not found: {source_dgslm_moshi}")
    
    # Process candor_turn_taking/candor_turn_freeze_omni directory
    source_freeze_omni = os.path.join(source_dir_path, "candor_turn_freeze_omni")
    if os.path.exists(source_freeze_omni):
        sample_count = 1
        target_freeze_omni_dir = os.path.join(target_base_path, "turntaking", "candor", "freezeomni")
        
        # Create the target directory if it doesn't exist
        os.makedirs(target_freeze_omni_dir, exist_ok=True)
        
        for item in os.listdir(source_freeze_omni):
            item_path = os.path.join(source_freeze_omni, item)
            
            # If it's a directory that matches our UUID pattern
            if os.path.isdir(item_path) and extract_sample_id(item):
                # Process freeze_omni - combine input.wav with output.wav
                input_wav = os.path.join(item_path, "input.wav")
                output_wav = os.path.join(item_path, "output.wav")
                
                if os.path.exists(input_wav) and os.path.exists(output_wav):
                    output_file = os.path.join(target_freeze_omni_dir, f"sample_{sample_count}.wav")
                    success = combine_wav_files(input_wav, output_wav, output_file)
                    if not success:
                        print(f"Failed to combine files for freezeomni: {item_path}")
                else:
                    missing = []
                    if not os.path.exists(input_wav): missing.append("input.wav")
                    if not os.path.exists(output_wav): missing.append("output.wav")
                    print(f"Missing files for freezeomni in {item_path}: {', '.join(missing)}")
                
                sample_count += 1
        
        print("Processing completed for candor_turn_freeze_omni directory.")
    else:
        print(f"Source directory not found: {source_freeze_omni}")

# Process synthetic_pause models
def process_synthetic_pause(target_base_path):
    source_synthetic_pause = os.path.join(source_dir, "synthetic_pause")
    
    if not os.path.exists(source_synthetic_pause):
        print(f"Source directory not found: {source_synthetic_pause}")
        return
    
    # Process dGSLM model
    source_dgslm = os.path.join(source_synthetic_pause, "dgslm")
    if os.path.exists(source_dgslm):
        target_dgslm_dir = os.path.join(target_base_path, "pause", "synthetic", "dGSLM")
        os.makedirs(target_dgslm_dir, exist_ok=True)
        
        process_numeric_directories(
            source_dgslm, 
            target_dgslm_dir, 
            None, 
            "dgslm_output_stereo.wav",
            combine=False
        )
        print("Processing completed for synthetic_pause/dgslm directory.")
    else:
        print(f"Source directory not found: {source_dgslm}")
    
    # Process freeze_omni model
    source_freezeomni = os.path.join(source_synthetic_pause, "freeze_omni")
    if os.path.exists(source_freezeomni):
        target_freezeomni_dir = os.path.join(target_base_path, "pause", "synthetic", "freezeomni")
        os.makedirs(target_freezeomni_dir, exist_ok=True)
        
        process_numeric_directories(
            source_freezeomni, 
            target_freezeomni_dir, 
            "input.wav", 
            "output.wav",
            combine=True
        )
        print("Processing completed for synthetic_pause/freeze_omni directory.")
    else:
        print(f"Source directory not found: {source_freezeomni}")
    
    # Process moshi model
    source_moshi = os.path.join(source_synthetic_pause, "moshi")
    if os.path.exists(source_moshi):
        target_moshi_dir = os.path.join(target_base_path, "pause", "synthetic", "moshi")
        os.makedirs(target_moshi_dir, exist_ok=True)
        
        process_numeric_directories(
            source_moshi, 
            target_moshi_dir, 
            None, 
            "moshi_out.wav",
            combine=False
        )
        print("Processing completed for synthetic_pause/moshi directory.")
    else:
        print(f"Source directory not found: {source_moshi}")

# Helper function to process directories with numeric names
def process_numeric_directories(source_dir, target_dir, left_file, right_file, combine=True):
    sample_count = 1
    
    # Get all numeric directories
    dir_items = [item for item in os.listdir(source_dir) if item.isdigit() and os.path.isdir(os.path.join(source_dir, item))]
    
    # Sort numerically
    dir_items.sort(key=int)
    
    for item in dir_items:
        item_path = os.path.join(source_dir, item)
        output_file = os.path.join(target_dir, f"sample_{sample_count}.wav")
        
        if combine:
            # Combine two files
            left_path = os.path.join(item_path, left_file)
            right_path = os.path.join(item_path, right_file)
            
            if os.path.exists(left_path) and os.path.exists(right_path):
                success = combine_wav_files(left_path, right_path, output_file)
                if success:
                    sample_count += 1
                else:
                    print(f"Failed to combine files for {item_path}")
            else:
                missing = []
                if not os.path.exists(left_path): missing.append(left_file)
                if not os.path.exists(right_path): missing.append(right_file)
                print(f"Missing files for {item_path}: {', '.join(missing)}")
        else:
            # Just copy a single file
            source_file = os.path.join(item_path, right_file)
            if os.path.exists(source_file):
                shutil.copy2(source_file, output_file)
                print(f"Copied {source_file} to {output_file}")
                sample_count += 1
            else:
                print(f"Missing file for {item_path}: {right_file}")

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
    
    # Process turntaking models with different directory structure
    process_turntaking_models(target_dir)
    
    # Process synthetic_pause models
    process_synthetic_pause(target_dir)
    
if __name__ == "__main__":
    main()