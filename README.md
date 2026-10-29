# rcipc

IPC in the joint coordinates of articulated rigid bodies (RC-IPC), rebuilt from scratch one stage at a time.
The plan and the conventions are in [notes/00_roadmap.md](notes/00_roadmap.md); every stage has its own note.

## Build and test

Needs CMake 3.24 or later and a C++17 compiler (tested with MSVC 2022). Eigen and GoogleTest are downloaded
at configure time.

```
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

## Layout

- `include/rcipc/`, `src/`: the library (what later stages reuse).
- `toys/`: small standalone models used to learn one idea (stage 1: `toys/ipc1d/`).
- `tests/`: GoogleTest suites, one per topic.
- `notes/`: derivations and exercises, in Chinese.
- `tools/`: Python scripts that plot outputs.
