#!/bin/bash
# Build script for Render deployment

set -e

echo "=== Installing Backend Dependencies ==="
cd backend
pip install -r requirements.txt

echo "=== Installing Frontend Dependencies ==="
cd ../frontend
npm ci

echo "=== Building Frontend ==="
npm run build

echo "=== Copying Frontend to Backend ==="
mkdir -p ../backend/static
cp -r dist/* ../backend/static/

echo "=== Build Complete ==="

