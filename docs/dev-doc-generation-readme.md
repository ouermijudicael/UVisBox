# Documentation Generation Guide

UVisBox uses Sphinx for its hand-written documentation and Sphinx AutoAPI for
the API reference. AutoAPI parses Python source without importing or executing
UVisBox. The two kinds of source are deliberately separated:

- Edit narrative pages directly in `docs/source/`, including `examples.rst`.
- Never edit `docs/source/_generated/`; it is recreated from `uvisbox/` on
  every build and is ignored by Git.

## Environment

The repository's `environment.yml` includes UVisBox and the documentation
dependencies. Create the environment only if it does not already exist:

```bash
conda env create -f environment.yml
```

Update an existing environment after dependency changes:

```bash
conda env update -n uvisbox -f environment.yml --prune
```

## Build the documentation

From the repository root, run:

```bash
conda run -n uvisbox python docs/build_docs.py
```

The build helper performs two operations:

1. Removes stale API pages; Sphinx AutoAPI then recreates canonical
   `uvisbox.*` pages in `docs/source/_generated/api/` without importing UVisBox.
2. Builds HTML with warnings treated as errors and writes it to
   `docs/build/html/`.

Warnings are also recorded in `docs/build/html-warnings.log` to make strict
build failures easier to review.

The resulting site starts at `docs/build/html/index.html`.

To inspect incomplete documentation while resolving warnings, strict mode may
be disabled locally:

```bash
conda run -n uvisbox python docs/build_docs.py --no-strict
```

Do not use `--no-strict` in CI or for deployment.

## Editing examples

`docs/source/examples.rst` is a hand-written gallery. It uses a Sphinx
`include` directive to render the reStructuredText narrative inside each
script's leading module docstring, plus download and image directives. This
keeps the prose and sectioned code blocks synchronized without executing the
example scripts. When adding an example:

1. Add the runnable script under `examples/`.
2. Put the narrative and its `code-block` sections in a leading triple-quoted
   module docstring.
3. Add or update its section in `docs/source/examples.rst`.
4. Add an `include` bounded by the triple quotes and a download directive
   pointing to the same script.
5. Add a representative result image under `docs/source/_static/` when useful.
6. Run the strict documentation build.

Keep expensive computation and plotting in a `main()` function protected by
`if __name__ == "__main__":` so examples can also be inspected safely by other
tools.

## Adding or removing Python modules

No API stub needs to be created or deleted manually. Update the source package
and run the build helper. If a public module is missing from the resulting API,
check its package structure and Sphinx warnings rather than editing generated
files.

## CI behavior

Pull requests and pushes to `main` run the same strict build. GitHub Pages is
deployed only for successful builds from `main`. An import failure, malformed
reference, or other Sphinx warning therefore prevents publication of a partial
site.
