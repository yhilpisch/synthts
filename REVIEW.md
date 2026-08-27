# Remaining Review Items

The package, notebooks, sweep utilities, documentation, and tests have
completed a source-level consistency review. This file tracks only unresolved
items.

## Open Items

### 1. Long-Run Model Validation

The automated suite validates model interfaces, short training runs, losses,
checkpointing, and plots. Publication claims comparing trained generators
still require fixed-budget long runs, held-out evaluation, and stored
experiment metadata.

### 2. Data Reproducibility

The examples download the canonical end-of-day dataset at runtime. A durable
release should pin a redistributable snapshot and its provenance metadata or
record a checksum and retrieval date for the remote input.
