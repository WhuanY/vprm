#!/bin/bash
# Function to process each set of split files
process_files() {
    local part="$1"
    
    # Extract the base name of the file
    local base_name=$(basename "$part" .tar.gz.part_aa)
    
    # Merge the split files into a single archive
    cat "${base_name}".tar.gz.part_* > "${base_name}.tar.gz"
    
    # Extract the merged archive
    tar -xzf "${base_name}.tar.gz"
    
    # Remove the individual split files
    rm -rf "${base_name}".tar.gz.part_*

    rm -rf "${base_name}.tar.gz"
}

export -f process_files

# Find all .tar.gz.part_aa files and process them in parallel
find . -name '*.tar.gz.part_aa' | parallel process_files

# Wait for all background jobs to finish
wait