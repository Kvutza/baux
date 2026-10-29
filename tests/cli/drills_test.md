# Workflow Drills Verification Suite

This test suite executes and validates the mechanical commands of the BAUX Workflow Drills using Meta Scrut.

## Drill 01: Inner-Loop Target Determination (btd)

Test that `btd` help operates as expected:

```scrut
$ ./tools/btdw --help | head -n 1
Buck-based target determinator
```

Test that the one-shot target determination script isolates target levels:

```scrut
$ ./tools/btd-targetsw 'root//rust/crates/cli:' | grep -E "^Level [0-9]" | head -n 1
Level 0
```

## Drill 02: Beads Task Coordination

Test that task context binding is active:

```scrut
$ ./baux context | grep -E "\[in_progress\]|\[open\]" | head -n 1 | awk '{print $2}'
[in_progress]
```

## Drill 03: Fastmod Codebase Codemodding

Test that fastmod is installed and executable:

```scrut
$ ./tools/fastmodw --version
fastmod 0.4.4
```

## Drill 04: Scrut Literate CLI Testing

Test that Scrut framework is functional:

```scrut
$ ./tools/scrutw --version
scrut 0.5.0
```

## Drill 07: Literature Database & Theorem Crawler

Test that embedded PostgreSQL daemon is active:

```scrut
$ ./tools/pgrustw status | grep "Running" | cut -d' ' -f1
[pgrustw]
```

Test that arXiv paper crawler CLI is available:

```scrut
$ ./tools/paper-crawlw 2>&1 | head -n 1 | cut -d' ' -f1,2
Usage: paper-crawl
```

## Drill 08: Hermetic Build & Packaging Toolchain

Test that Buck2 wrapper operates hermetically:

```scrut
$ ./tools/buck2w --version | cut -d' ' -f1
buck2
```

Test that Reindeer Cargo buckifier operates hermetically:

```scrut
$ ./tools/reindeerw --help | head -n 1
Usage: reindeer [OPTIONS] <COMMAND>
```

Test that Prek pre-commit runner is functional:

```scrut
$ ./tools/prekw --version | cut -d' ' -f1,2
prek 0.5.7
```
