from backend.run_pipeline import (
    run_full_pipeline
)

result = run_full_pipeline(
    "uploads/input.csv"
)

print(
    "\nPipeline Result:\n"
)

print(result)