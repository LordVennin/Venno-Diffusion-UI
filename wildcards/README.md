# Wildcards

Each `.txt` file here is a list of choices, one per line (blank lines and lines starting with `#` are skipped).
Use a file in any prompt as `__name__`, e.g. `__examples/hair_color__`; every image picks its own line.
Lines may use the same syntax again: `{a|b}`, `{2$$a|b|c}`, `{0.2::rare|common}`, `__other_file__`.

Anima tags are lowercase with spaces (no underscores, except `score_` tags); artists need an `@` prefix.
