N=8  # Number of threads (adjust as needed)
GPUS=8  # Number of available GPUs
MAX_RETRIES=1  # Maximum number of retries for failed threads

input_list=$1  # Path to the fg_ratios_*.txt file
out_path=$2  # Output path for the variations
noise_level=$3  # Noise level input passed as the third argument

# Check if noise level is provided
if [ -z "$noise_level" ]; then
  echo "Error: Please provide a noise level (e.g., 0, 100, 200, etc.)."
  exit 1
fi

# Get the total number of lines in the input list
total_lines=$(wc -l < "$input_list")

# Calculate the number of lines per thread
lines_per_thread=$((total_lines / N))

# Split the input list into N parts
split -d -l "$lines_per_thread" "$input_list" "$input_list.part"

start=0

# Function to run each thread
function run_thread() {
  local thread_idx="$1"
  local retry=0
  local gpu_id=$((thread_idx % $GPUS + $start))  # Calculate the GPU ID for the current thread

  while [ $retry -le $MAX_RETRIES ]; do
    CUDA_VISIBLE_DEVICES="$gpu_id" python -u tools/toolbox_genview/generate_image_variations_noiselevels.py \
      --input-list "$input_list.part$(printf %02d "$thread_idx")" \
      --output-prefix $out_path \
      --noise-level "$noise_level"  # Pass the noise level as a parameter

    exit_status=$?

    if [ $exit_status -eq 0 ]; then
      echo "Thread $thread_idx completed successfully."
      break
    else
      echo "Thread $thread_idx failed (Retry $retry)."
      retry=$((retry + 1))
    fi
  done
}

# Run the Python script for each part in parallel
for ((i = 0; i < N; i++)); do
  run_thread "$i" &
done

# Wait for all background jobs to finish
wait

# Clean up the temporary files
rm "$input_list.part"*
