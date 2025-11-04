#!/bin/bash
# Use the simple version that avoids local model imports
exec 2> >(grep -v "mutex.cc" >&2)
python3 scripts/test_retrieval_simple.py "$@"
