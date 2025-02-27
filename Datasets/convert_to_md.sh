#!/bin/bash

# Find all README.ipynb files within subdirectories of the current working directory
notebook_files=$(find "$(pwd)" -type f -name "README.ipynb")

# Check if any notebook files are found
if [ -z "$notebook_files" ]; then
  echo "No README.ipynb files found."
  exit 1
fi

# Convert each notebook to Markdown
for notebook in $notebook_files; do
  echo "Converting $notebook to Markdown..."
  jupyter nbconvert --to markdown "$notebook"
done

echo "Conversion completed."