#!/bin/bash

# GCC
# python src/ocs_ortools.py --system gcc --optimize_type both -t 12
# python src/ocs_ortools_cv.py --system gcc --optimize_type both -t 12
# # NodeJS
# python src/ocs_ortools.py --system nodejs --optimize_type both -t 12
# python src/ocs_ortools_cv.py --system nodejs --optimize_type both -t 12

# # Poppler
# python src/ocs_ortools.py --system poppler --optimize_type both -t 12
# python src/ocs_ortools_cv.py --system poppler --optimize_type both -t 12

# # SQLite
# python src/ocs_ortools.py --system sqlite --optimize_type both -t 12
# python src/ocs_ortools_cv.py --system sqlite --optimize_type both -t 12

# # x264
# python src/ocs_ortools.py --system x264 --optimize_type both -t 12
# python src/ocs_ortools_cv.py --system x264 --optimize_type both -t 12

# # XZ
# python src/ocs_ortools.py --system xz --optimize_type both -t 12
# python src/ocs_ortools_cv.py --system xz --optimize_type both -t 12

# # Lingeling
python src/ocs_ortools.py --system lingeling --optimize_type both -t 12
python src/ocs_ortools_cv.py --system lingeling --optimize_type both -t 12

# ImageMagick
# python src/ocs_ortools.py --system imagemagick --optimize_type both -t 12
# python src/ocs_ortools_cv.py --system imagemagick --optimize_type both -t 12
