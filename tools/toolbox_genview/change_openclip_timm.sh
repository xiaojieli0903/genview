# Retrieve the Python version and the path to site-packages
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
site_packages_path="$CONDA_PREFIX/lib/python$python_version/site-packages"

# Define a function to replace files from the source directory to the destination directory
replace_files() {
  local src_dir=$1
  local dest_dir=$2
  # Loop through all files in the source directory
  for src_file in $(find $src_dir -type f); do
    # Calculate the relative path of the source file
    local relative_path="${src_file#$src_dir/}"
    local dest_file="$dest_dir/$relative_path"

    # Check if the destination file exists
    if [ -f "$dest_file" ]; then
      # Backup the original file by renaming it with a .back extension
      mv "$dest_file" "$dest_file.back"
      echo "Backed up $dest_file to $dest_file.back"
    else
      # Ensure the destination directory exists
      mkdir -p "$(dirname "$dest_file")"
    fi

    # Copy the new file to the destination directory
    cp "$src_file" "$dest_file"
    echo "Replaced $dest_file with $src_file"
  done
}

# Replace files in the open_clip and timm directories
echo "Updating open_clip..."
replace_files "./tools/clip_pca/open_clip" "$site_packages_path/open_clip"

echo "Updating timm..."
replace_files "./tools/clip_pca/timm" "$site_packages_path/timm"
