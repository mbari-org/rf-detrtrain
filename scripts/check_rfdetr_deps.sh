#!/bin/bash
# Check rfdetr package dependencies

echo "========================================="
echo "Checking rfdetr package dependencies"
echo "========================================="

# Check if rfdetr is available on PyPI and show its dependencies
pip index versions rfdetr 2>/dev/null || echo "rfdetr not found on PyPI"

echo ""
echo "Trying to get package info..."
pip show rfdetr 2>/dev/null || echo "rfdetr not installed locally"

echo ""
echo "========================================="
echo "Recommended: Use specific versions"
echo "========================================="
echo "The following versions are known to work together:"
echo "  - transformers>=4.35.0"
echo "  - accelerate>=0.25.0"
echo "  - torch>=2.0.0"
echo ""
echo "If rfdetr requires specific versions, they will be installed"
echo "when you run: pip install rfdetr"
