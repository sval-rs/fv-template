# `fv-template`

[![Rust](https://github.com/sval-rs/fv-template/workflows/Rust/badge.svg)](https://github.com/sval-rs/fv-template/actions)
[![Latest version](https://img.shields.io/crates/v/fv-template.svg)](https://crates.io/crates/fv-template)
[![Documentation Latest](https://docs.rs/fv-template/badge.svg)](https://docs.rs/fv-template)

## Getting started

In your proc-macro crate, you can add `fv-template` as a dependency. Consumers of your proc-macros don't need to depend on `fv-template` themselves.

For details on what field-value templates are and why you might want to use them, see [the docs](https://docs.rs/fv-template).

## How do I use it?

This library is intended to be used by proc-macro authors, like [`emit`](https://github.com/KodrAus/emit). It doesn't define any macros of its own.

See the `examples` directory for a sample that uses `fv-template` in a proc-macro and a consuming application.
