# Preserved qualification failures

Attempt 1 preserved implementation full/patch/source before validation. The child interpreter path incorrectly resolved the venv symlink to the hosted base interpreter. Only venv path handling was repaired; accepted source and checkpoints were not rolled back.

Attempt 2 executed through the admitted venv and reached native tower qualification, but the child returned nonzero. The implementation checkpoint and runtime evidence remain preserved. Attempt 3 adds log propagation only.
