# BAUX Toolchain Verification Tests

This test suite verifies the hermetic DotSlash and host toolchains using Meta Scrut.

## Meta DotSlash Toolchain Commands

Test that Buck2 change detector (btd) is functional:

```scrut
$ ./tools/btdw --help | head -n 1
Buck-based target determinator
```

Test that automated target determination wrapper works:

```scrut
$ ./tools/btd-targetsw 'root//rust/crates/cli:' | grep -E "^Level [0-9]" | head -n 1
Level 0
```

Test that Scrut CLI testing framework is functional:

```scrut
$ ./tools/scrutw --version
scrut 0.5.0
```

Test that Fastmod is functional:

```scrut
$ ./tools/fastmodw --version
fastmod 0.4.4
```

Test that Buck2 wrapper is functional:

```scrut
$ ./tools/buck2w --version | cut -d' ' -f1
buck2
```
