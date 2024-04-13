build-NACA4AirfoilLayer:
  mkdir -p "$(ARTIFACTS_DIR)/python"
  cp *.py "$(ARTIFACTS_DIR)/python"
  python -m pip install -r requirements.txt -t "$(ARTIFACTS_DIR)/python"
  zip -r "$(ARTIFACTS_DIR)/python/naca_layer.zip" "$(ARTIFACTS_DIR)/python"
  rm -rf "$(ARTIFACTS_DIR)/python"