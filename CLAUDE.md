# Tesseract weldpath — operating rules

How to work in this repository. Nothing here is a finding about the program; findings
live in `docs/findings.md` and are not binding — see **Reference material** below.

## Tooling

- Python is `./.venv/Scripts/python.exe`. Not `python`, not `py`.
- `gh` is `"C:/Program Files/GitHub CLI/gh.exe"`.
- **There is no scipy.** `numpy` and `tesseract_robotics` are there; assume nothing else.
- Repository files are **CRLF** and `core.autocrlf=true`. A file written with LF endings
  shows up as wholly rewritten.
- Entry point is `main.py`; `tools/cellsweep.py` is the only other script.
- Shell is PowerShell by default, with Bash available. They take different syntax.

## Tests

**No test suite is tracked in this repository.** About forty-five `*_test.py` files exist,
all of them inside session scratchpad directories under `%TEMP%\claude\...`, which are
temporary and per-session. Anything written there is gone as far as the next session is
concerned, and `docs/findings.md` refers to suites — `tree_test`, `recut_test` — that
cannot be run from a clean checkout.

So: a suite written to the scratchpad is a working tool, not a deliverable. If a test is
meant to hold a guarantee over time, it has to be committed.

## Studies and data

- `C:\Users\christopherf\Documents\AutoPathGenerator\Path 1`, `Path 2` and `Path 3` are
  **read only**. Work on a scratch copy, with `meshes` as a directory junction.
- **Study directories get reused.** A `Path N` folder is overwritten when another study is
  run into it, so a scratch copy's junction goes stale with nothing said and a folder's
  manifest need not match the failure views sitting beside it. Check that they agree
  before concluding anything from them.
- **Never write to the export directory** — `C:\Users\christopherf\Documents\TestFiles\Test
  Studies\Tesseract` — without saying so first.

## Running things

- Ask before any test expected to take longer than **three minutes**.
- Report what the run actually did. If it failed, say so with the output; if a step was
  skipped, say that.

## Git

- `.gitignore` does **not** cover `__pycache__` or `*.pyc`, so they have to be left out of
  commits by hand. Leave `weldpath/carve/`, `build/` and `dist/` untracked.
- Check the state of any open PR before doing git work. **Confirm before any push.**

## Reference material

Three documents, and they are not equivalent:

| | what it is |
|---|---|
| `MANUAL.md` | the internals manual, gitignored and local. Written for the author, not for Claude. Its section headings carry line numbers, which drift. |
| `README.md` | the user-facing manual. 59 of its 61 commits are Claude's, so it is not an independent source. |
| `docs/findings.md` | a dated log of measurements and decisions. These may become outdated quickly. **Not a specification.** |

`docs/findings.md` was this file until 2026-11-13, which meant it arrived as standing
instruction on every turn and got cited as settled fact. It is not that. Two of its
figures were wrong for weeks while being quoted — a 513 s `contactTest` profile that does
not reproduce, and a shell count of 11688 for a cell that builds 11020.

Consequences, which are the rules that matter here:

- **Do not cite any of the three as current.** Quote the code, or a run you just made.
- Be suspicious of explanations in code comments for the same reason.
- Check a flag, attribute or module name against the live object (`dir(type(x))`) or a
  grep before using it. Guessed names have been wrong repeatedly.
- Make a new test fixture reproduce a known-good result before reading anything off it.
- Before reporting a measured number, ask what else is inside it.
- Do not write conclusions back into this file. Findings go in `docs/findings.md`, dated,
  as findings.
