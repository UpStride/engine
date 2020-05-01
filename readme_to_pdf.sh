#!/bin/bash
pandoc \
    --pdf-engine=xelatex \
    --template=doc/template.tex \
    --toc \
    --variable fontsize=12pt \
    --variable geometry:"bottom=3.5cm" \
    --variable version=2.0 README.md -o readme.pdf

# see https://pandoc.org/faqs.html
# --variable geometry:"top=3cm, bottom=1.5cm, left=2cm, right=2cm" \