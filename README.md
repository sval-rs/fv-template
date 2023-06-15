# `fv-template`

[![Rust](https://github.com/sval-rs/fv-template/workflows/Rust/badge.svg)](https://github.com/sval-rs/fv-template/actions)
[![Latest version](https://img.shields.io/crates/v/fv-template.svg)](https://crates.io/crates/fv-template)
[![Documentation Latest](https://docs.rs/fv-template/badge.svg)](https://docs.rs/fv-template)

## What is this?

This library defines _field-value templates_: a string interpolation syntax for Rust. It's an alternative to `format_args!` that uses field-value expressions like `field: "value"` for arguments and string interpolation.

The following are all examples of field-value templates:

- `"a regular string"`
- `"a string with an {ident} in it"`
- `"a string with a {named: \"value\"} in it"`
- `before: 1, "any string like we've already seen", after: 2`

Templates take the following form:

```
pre_template_arg_1: FieldValue, pre_template_arg_n: FieldValue, template: Lit, post_template_arg_1: FieldValue, post_template_arg_n: FieldValue
```

where the `template` supports interpolating values with `{FieldValue}`.

### How are templates different from `format_args!`?

Templates differ from `format_args!` by using an explicit API for parsing at compile time. That way consumers can work with a parsed template directly instead of having to make assumptions about how it will be interpreted downstream just based on its input tokens.

Templates also use different syntax for interpolated values. Because they're just field values, templates support any arbitrary Rust expression between `{}` and have consistent syntax for associating an identifier with an expression. Templates don't need to invent flags like `:?` for communicating how to interpret expressions out-of-band, that's left up to the user to decide.

### How are templates different from `log!`?

`log!` is based on `format_args!` but supports additional parameters besides the string to interpolate. Templates differ from `log!` by using field values for those parameters surrounding the template. This works like named function arguments where arguments can be provided or omitted in any order. Named arguments can be more future-proof than positional arguments, without needing the boilerplate of arguments structs.

## How do I use it?

This library is intended to be used by proc-macro authors, like [`emit`](https://github.com/KodrAus/emit). It doesn't define any macros of its own.

Parse a `Template` from some 
