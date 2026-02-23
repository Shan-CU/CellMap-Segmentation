"""Patch EmptyImage.__init__ to accept **kwargs (fixes device= bug in cellmap_data)."""
import site
import os

# Find the installed empty_image.py
for sp in site.getsitepackages():
    target = os.path.join(sp, "cellmap_data", "empty_image.py")
    if os.path.exists(target):
        break
else:
    # Try user site
    import cellmap_data
    target = os.path.join(os.path.dirname(cellmap_data.__file__), "empty_image.py")

print(f"Patching: {target}")

with open(target, "r") as f:
    content = f.read()

# Find the exact pattern
old = "        empty_value: float | int = -100,\n    ):"
new = "        empty_value: float | int = -100,\n        **kwargs,\n    ):"

if old in content:
    content = content.replace(old, new, 1)
    with open(target, "w") as f:
        f.write(content)
    print("SUCCESS: Patched EmptyImage.__init__ to accept **kwargs")
else:
    # Check if already patched
    if "**kwargs" in content:
        print("ALREADY PATCHED: EmptyImage already has **kwargs")
    else:
        print("ERROR: Could not find pattern to patch")
        # Debug: show the __init__ signature
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "def __init__" in line or "empty_value" in line:
                print(f"  Line {i+1}: {line!r}")
