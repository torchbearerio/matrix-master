#!/bin/bash

# Build zip
rm -f build.zip
zip -r build.zip environment.yml matrixmaster Dockerfile .ebextensions

# Deploy to EB
eb deploy
