
import json
import sys

MINIMUM_R2_SCORE = 0.99

print("Reading sales model evaluation metrics...")

with open("metrics.json", "r") as file:
    metrics = json.load(file)

r2_score = metrics["r2_score"]

print("Model R2 Score:", round(r2_score, 4))
print("Required R2 Score:", MINIMUM_R2_SCORE)

if r2_score < MINIMUM_R2_SCORE:
    print("QUALITY GATE FAILED")
    print("Model performance is below the required threshold.")
    sys.exit(1)

print("QUALITY GATE PASSED")
print("Model performance satisfies the required threshold.")
sys.exit(0)
