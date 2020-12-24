# `fv-template`

An alternative to `format_args!` (and `log!`) for string templating. Templates take the following form:

```
pre_template_arg_1: FieldValue, pre_template_arg_n: FieldValue, template: Lit, post_template_arg_1: FieldValue, post_template_arg_n: FieldValue
```

where the `template` supports interpolating values with `{interpolated: FieldValue}`.

As an example, the following is a template:

```
log: some.log, debug, "This is the literal {interpolated: 42} of the template", extra
```

## How are templates different from `format_args!`?

Templates differ from `format_args!` by using an explicit API for parsing at compile time. That way consumers can interrogate a parsed template directly instead of having to make assumptions about how it will be interpreted downstream just based on its input tokens.

Templates also use different syntax for interpolated values. Because they're just field values, templates support any arbitrary Rust expression between `{}` and have consistent syntax for associating an identifier with an expression. Templates don't invent flags for determining how to interpret expressions, that's left up to the caller to decide. In `emit`, we use attributes like `#[debug]` instead of flags like `?`.

## How are templates different from `log!`?

Templates differ from `log!` by using field values for the inputs before and after the template. This works like named function arguments where arguments can be provided or omitted in any order. It's more future-proof than positional arguments without needing a lot more syntax for full structs.
