#!/bin/bash

echo "Run handin template"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist, creating it!"
  mkdir plots
fi

# Run poisson.py script
echo "Run the Poisson script ..."
python3 poisson.py

# Run vandermonde.py script
echo "Run the Vandermonde script ..."
python3 vandermonde.py

# Run your other Python scripts as needed
# ...


echo "Generating the PDF"

pdflatex template.tex
bibtex template.aux
pdflatex template.tex
pdflatex template.tex

# Optional: Compile ex1_2.tex if needed
#pdflatex ex1_2.tex
#bibtex ex1_2.aux
#pdflatex ex1_2.tex
#pdflatex ex1_2.tex

echo "Script execution completed."
