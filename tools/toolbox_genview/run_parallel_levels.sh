# Change the value of N to the desired number of threads
N=4
GPUS=4
 # Maximum number of retries for a failed thread
MAX_RETRIES=1

input_list=$1 #"/path/to/your/fg_ratios_0.txt"
out_path=#2 '/path/to/datasets/imagenet/train_variations/'

# Get the total number of lines in the input list
total_lines=$(wc -l < "$input_list")

# Calculate the number of lines per thread
lines_per_thread=$((total_lines / N))

# Split the input list into N parts
split -d -l "$lines_per_thread" "$input_list" "$input_list.part"
start=0
function run_thread() {
  local thread_idx="$1"
  local retry=0
  local gpu_id=$((thread_idx % $GPUS + $start))  # Calculate the GPU ID for the current thread

  while [ $retry -le $MAX_RETRIES ]; do
    CUDA_VISIBLE_DEVICES="$gpu_id" python -u tools/toolbox_genview/generate_image_variations_noiselevels.py --input_list "$input_list.part$(printf %02d "$i")" --output_prefix $out_path --input_prefix $input_prefix
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

# Run the Python script with each part as an argument using concurrent threads
for ((i = 0; i < N; i++)); do
  run_thread "$i" &
done

# Wait for all background jobs to finish
wait

# Clean up the temporary files
rm "$input_list.part"*