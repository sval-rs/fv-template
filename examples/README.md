# An example field-value template macro

This directory contains an example proc macro that uses `fv_template`.  There are two crates in here:

- `macros`: Defines a macro called `template_args!` that works like `format_args!`, but supports arbitrary
expressions in template strings.
- `app`: A simple binary that depends on `macros` and writes to the console.
