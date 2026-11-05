"""Workspace test dependencies that Reindeer does not export from dev-deps."""

def test_aliases():
    # Cargo pins baux/fbt tests to rand 0.8 and rand_chacha 0.3.1.
    native.alias(
        name = "rand_chacha",
        actual = ":rand_chacha-0.3",
        visibility = ["PUBLIC"],
    )
