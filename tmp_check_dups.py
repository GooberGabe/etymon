import json
from collections import Counter
from pathlib import Path

save_path = Path('saves/savegame.json')
if not save_path.exists():
    raise SystemExit(f'Missing {save_path}')
with save_path.open('r', encoding='utf-8') as fh:
    payload = json.load(fh)

names = [entry.get('name') for entry in payload.get('cultures', []) if entry.get('name')]
counts = Counter(names)
duplicates = [(name, count) for name, count in counts.items() if count > 1]
print(f'{len(duplicates)} duplicate culture names found')
for name, count in sorted(duplicates, key=lambda item: (-item[1], item[0])):
    print(f'{name}: {count}')
