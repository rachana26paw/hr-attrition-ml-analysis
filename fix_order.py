import json

with open('IBM_HR_AdvancedML_Complete_3.ipynb', 'r', encoding='utf-8') as f:
    d = json.load(f)

cells = d['cells']
# Sections 7 and 8 might exist? Let's check.
# The issue: 6 is at the bottom, 9 is above 6.
# Let's sort the markdown and code cells based on the section number if possible, or just push 9 to the end.

s9_cells = []
s6_cells = []
other_cells = []

# To make it robust:
for c in cells:
    src = "".join(c.get('source', []))
    if '9.1' in src or '9.2' in src:
        s9_cells.append(c)
    # We should ensure everything goes in order, let's keep it simple: just move anything with '9.1' or '9.2' to the end.
    else:
        other_cells.append(c)

d['cells'] = other_cells + s9_cells

with open('IBM_HR_AdvancedML_Complete_3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1)
print("Notebook Fixed")
