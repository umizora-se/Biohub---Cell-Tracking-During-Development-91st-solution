"""Execute a notebook-derived script cell by cell in one namespace (like Jupyter)."""
import re
import sys
import time

src = open(sys.argv[1]).read()
parts = re.split(r"^# %% \[cell ([\w.]+)\]\n", src, flags=re.M)
g = {"__name__": "__main__"}
for i in range(1, len(parts), 2):
    idx, code = parts[i], parts[i + 1]
    t0 = time.time()
    print(f"\n######## CELL {idx} start", flush=True)
    exec(compile(code, f"<cell {idx}>", "exec"), g)
    print(f"######## CELL {idx} done {time.time() - t0:.0f}s", flush=True)
