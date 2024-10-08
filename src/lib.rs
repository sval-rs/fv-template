/*!
Compile-time support for interpolated string templates using field-value expressions.

# Field-value templates

A field-value template is a string literal surrounded by field-value expressions:

```text
   a, b: 42, "Some text {c} and {d: true}", e, f: "text"
   ───┬────  ───────────────┬─────────────  ──────┬─────
before literal           literal             after literal
```

The template string literal consists of blocks of text with holes between braces, where
the value in a hole is a field-value expression:

```text
"Some text {c} and {d: true}"
           ─┬─     ────┬────
            └────┬─────┘
                hole

"Some text {c} and {d: true}"
 ─────┬────   ──┬──
      └────┬────┘
         text
```

The syntax is similar to Rust's `format_args!` macro, but leans entirely on standard field-value
expressions for specifying values to interpolate.

# Why not `format_args!`?

Rust's `format_args!` macro already defines a syntax for string interpolation, but isn't suitable
for all situations:

- It's core purpose is to build strings. `format_args!` is based on machinery that throws away
type-specific information eagerly. It also performs optimizations at compile time that inline
certain values into the builder.
- It doesn't have a programmatic API. You can only make assumptions about how a `format_args!`
invocation will behave by observing the syntactic tokens passed to it at compile-time. You don't get any
visibility into the format literal itself.
- Flags are compact for formatting, but don't scale. The `:?#<>` tokens used for customizing formatting
are compact, but opaque, and don't naturally allow for arbitrarily complex annotation like attributes do.

When any of those trade-offs in `format_args!` becomes a problem, field-value templates may be a solution.
*/

#[cfg(test)]
#[macro_use]
extern crate quote;

use std::fmt::Formatter;
use std::{
    borrow::Cow,
    fmt,
    iter::Peekable,
    ops::Range,
    str::{self, CharIndices},
};

use proc_macro2::{token_stream, Literal, Span, TokenStream, TokenTree};
use quote::ToTokens;
use syn::{spanned::Spanned, FieldValue};

/**
A field-value template.
 */
pub struct Template {
    before_template: Vec<FieldValue>,
    literal: Vec<LiteralPart>,
    after_template: Vec<FieldValue>,
}

/**
A visitor for the parts of a template string.
 */
pub trait LiteralVisitor {
    /**
    Visit a text part in a template literal.
     */
    fn visit_text(&mut self, text: &str);

    /**
    Visit a hole part in a template literal.
     */
    fn visit_hole(&mut self, hole: &FieldValue);
}

impl<'a, V: ?Sized> LiteralVisitor for &'a mut V
where
    V: LiteralVisitor,
{
    fn visit_text(&mut self, text: &str) {
        (**self).visit_text(text)
    }

    fn visit_hole(&mut self, hole: &FieldValue) {
        (**self).visit_hole(hole)
    }
}

impl Template {
    /**
    Parse a template from a `TokenStream`.

    The `TokenStream` is typically all the tokens given to a macro.
     */
    pub fn parse2(input: TokenStream) -> Result<Self, Error> {
        let mut scan = ScanTemplate::new(input);

        // Take any arguments up to the string template
        // These are control arguments for the log statement that aren't key-value pairs
        let mut parsing_value = false;
        let (before_template, template) = scan.take_until(|tt| {
            // If we're parsing a value then skip over this token
            // It won't be interpreted as the template because it belongs to an arg
            if parsing_value {
                parsing_value = false;
                return false;
            }

            match tt {
                // A literal is interpreted as the template
                TokenTree::Literal(_) => true,
                // A `:` token marks the start of a value in a field-value
                // The following token is the value, which isn't considered the template
                TokenTree::Punct(p) if p.as_char() == ':' => {
                    parsing_value = true;
                    false
                }
                // Any other token isn't the template
                _ => false,
            }
        });

        // If there's more tokens, they should be a comma followed by comma-separated field-values
        let after_template = if scan.has_input() {
            scan.expect_punct(',')?;
            scan.rest.collect()
        } else {
            TokenStream::new()
        };

        // Parse the template literal into its text fragments and field-value holes
        let literal = if let Some(template) = template {
            LiteralPart::parse_lit2(ScanTemplate::take_literal(template)?)?
        } else {
            Vec::new()
        };

        let before_template = ScanTemplate::new(before_template).collect_field_values()?;
        let after_template = ScanTemplate::new(after_template).collect_field_values()?;

        Ok(Template {
            before_template,
            literal,
            after_template,
        })
    }

    /**
    Field-values that appear before the template string literal.
     */
    pub fn before_literal_field_values<'a>(&'a self) -> impl Iterator<Item = &'a FieldValue> {
        self.before_template.iter()
    }

    /**
    Field-values that appear within the template string literal.

    This is a simple alternative to [`Template::visit_literal`] that iterates over the field-value holes.
     */
    pub fn literal_field_values<'a>(&'a self) -> impl Iterator<Item = &'a FieldValue> {
        self.literal.iter().filter_map(|part| {
            if let LiteralPart::Hole { expr, .. } = part {
                Some(expr)
            } else {
                None
            }
        })
    }

    /**
    Whether the template contains a literal.
    */
    pub fn has_literal(&self) -> bool {
        !self.literal.is_empty()
    }

    /**
    Field-values that appear after the template string literal.
     */
    pub fn after_literal_field_values<'a>(&'a self) -> impl Iterator<Item = &'a FieldValue> {
        self.after_template.iter()
    }

    /**
    Visit the parts of the string literal part of the template.

    Each fragment of text and field-value hole will be visited in sequence.

    Given a template string like:

    ```text
    Some text and a {hole} and some {more}.
    ```

    the visitor will be called with the following inputs:

    1. `visit_text("Some text and a ")`
    2. `visit_hole("hole")`
    3. `visit_text(" and some ")`
    4. `visit_hole("more")`
    5. `visit_text(".")`

    If the template doesn't contain a literal then the visitor won't be called.
     */
    pub fn visit_literal(&self, mut visitor: impl LiteralVisitor) {
        for part in &self.literal {
            match part {
                LiteralPart::Text { text, .. } => visitor.visit_text(text),
                LiteralPart::Hole { expr, .. } => visitor.visit_hole(&expr),
            }
        }
    }
}

/**
A part of a parsed template string literal.
 */
enum LiteralPart {
    /**
    A fragment of text.
     */
    Text {
        /**
        The literal text content.
        */
        text: String,
        /**
        The range within the template string that covers this part.
        */
        range: Range<usize>,
    },
    /**
    A replacement expression.
     */
    Hole {
        /**
        The expression within the hole.
        */
        expr: FieldValue,
        /**
        The range within the template string that covers this part.
        */
        range: Range<usize>,
    },
}

impl fmt::Debug for LiteralPart {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LiteralPart::Text { text, range } => f
                .debug_struct("Text")
                .field("text", text)
                .field("range", range)
                .finish(),
            LiteralPart::Hole { expr, range } => f
                .debug_struct("Hole")
                .field("expr", &format_args!("`{}`", expr.to_token_stream()))
                .field("range", range)
                .finish(),
        }
    }
}

impl LiteralPart {
    fn parse_lit2(lit: Literal) -> Result<Vec<Self>, Error> {
        enum Expecting {
            TextOrEOF,
            Hole,
        }

        let input = lit.to_string();

        let mut parts = Vec::new();
        let mut expecting = Expecting::TextOrEOF;

        let mut scan = ScanPart::new(lit, &input)?;

        // If the template is empty then return a single text part
        // This distinguishes an empty template from a missing template
        if !scan.has_input() {
            return Ok(vec![LiteralPart::Text {
                text: String::new(),
                range: 0..0,
            }]);
        }

        while scan.has_input() {
            match expecting {
                Expecting::TextOrEOF => {
                    if let Some((text, range)) = scan.take_until_eof_or_hole_start()? {
                        parts.push(LiteralPart::Text {
                            text: text.into_owned(),
                            range,
                        });
                    }

                    expecting = Expecting::Hole;
                    continue;
                }
                Expecting::Hole => {
                    let (expr, range) = scan.take_until_hole_end()?;

                    let expr_span = scan.lit.subspan(range.start..range.end);

                    let tokens = {
                        let tokens: TokenStream = str::parse(&*expr).map_err(|e| {
                            Error::lex_fv_expr(expr_span.unwrap_or(scan.lit.span()), &*expr, e)
                        })?;

                        // Attempt to shrink the span of the parsed expression to just the
                        // fragment of the literal it was parsed from
                        if let Some(span) = scan.lit.subspan(range.start..range.end) {
                            tokens
                                .into_iter()
                                .map(|mut tt| {
                                    tt.set_span(span);
                                    tt
                                })
                                .collect()
                        } else {
                            tokens
                        }
                    };

                    let expr = syn::parse2(tokens).map_err(|e| {
                        Error::parse_fv_expr(expr_span.unwrap_or(scan.lit.span()), &*expr, e)
                    })?;

                    parts.push(LiteralPart::Hole { expr, range });

                    expecting = Expecting::TextOrEOF;
                    continue;
                }
            }
        }

        Ok(parts)
    }
}

struct ScanTemplate {
    span: Span,
    rest: Peekable<token_stream::IntoIter>,
}

impl ScanTemplate {
    fn new(input: TokenStream) -> Self {
        ScanTemplate {
            span: input.span(),
            rest: input.into_iter().peekable(),
        }
    }

    fn has_input(&mut self) -> bool {
        self.rest.peek().is_some()
    }

    fn take_until(
        &mut self,
        mut until_true: impl FnMut(&TokenTree) -> bool,
    ) -> (TokenStream, Option<TokenTree>) {
        let mut taken = TokenStream::new();

        while let Some(tt) = self.rest.next() {
            if until_true(&tt) {
                return (taken, Some(tt));
            }

            taken.extend(Some(tt));
        }

        (taken, None)
    }

    fn is_punct(input: &TokenTree, c: char) -> bool {
        match input {
            TokenTree::Punct(p) if p.as_char() == c => true,
            _ => false,
        }
    }

    fn expect_punct(&mut self, c: char) -> Result<TokenTree, Error> {
        match self.rest.next() {
            Some(tt) => {
                if Self::is_punct(&tt, c) {
                    Ok(tt)
                } else {
                    Err(Error::invalid_char(tt.span(), &[c]))
                }
            }
            None => Err(Error::invalid_char_eof(self.span, &[c])),
        }
    }

    fn take_literal(tt: TokenTree) -> Result<Literal, Error> {
        match tt {
            TokenTree::Literal(l) => Ok(l),
            _ => Err(Error::invalid_literal(tt.span())),
        }
    }

    fn collect_field_values(mut self) -> Result<Vec<FieldValue>, Error> {
        let mut result = Vec::new();

        while self.has_input() {
            let (arg, _) = self.take_until(|tt| Self::is_punct(&tt, ','));

            if !arg.is_empty() {
                let expr_span = arg.span();

                result.push(syn::parse2::<FieldValue>(arg).map_err(|e| {
                    Error::parse_fv_expr(expr_span, expr_span.source_text().as_deref(), e)
                })?);
            }
        }

        Ok(result)
    }
}

struct ScanPart<'input> {
    lit: Literal,
    input: &'input str,
    start: usize,
    end: usize,
    rest: Peekable<CharIndices<'input>>,
}

struct TakeUntil<'a, 'input> {
    current: char,
    current_idx: usize,
    rest: &'a mut Peekable<CharIndices<'input>>,
    lit: &'a Literal,
}

impl<'input> ScanPart<'input> {
    fn new(lit: Literal, input: &'input str) -> Result<Self, Error> {
        let mut iter = input.char_indices();
        let start = iter.next();
        let end = iter.next_back();

        // This just checks that we're looking at a string
        // It doesn't bother with ensuring that last quote is unescaped
        // because the input to this is expected to be a proc-macro literal
        if start.map(|(_, c)| c) != Some('"') || end.map(|(_, c)| c) != Some('"') {
            return Err(Error::invalid_literal(lit.span()));
        }

        Ok(ScanPart {
            lit,
            input: &input,
            start: 1,
            end: input.len() - 1,
            rest: iter.peekable(),
        })
    }

    fn has_input(&mut self) -> bool {
        self.rest.peek().is_some()
    }

    fn take_until(
        &mut self,
        mut until_true: impl FnMut(TakeUntil<'_, 'input>) -> Result<bool, Error>,
    ) -> Result<Option<(Cow<'input, str>, Range<usize>)>, Error> {
        let mut scan = || {
            while let Some((i, c)) = self.rest.next() {
                if until_true(TakeUntil {
                    current: c,
                    current_idx: i,
                    rest: &mut self.rest,
                    lit: &self.lit,
                })? {
                    let start = self.start;
                    let end = i;

                    self.start = end + 1;

                    let range = start..end;

                    return Ok((Cow::Borrowed(&self.input[range.clone()]), range));
                }
            }

            let range = self.start..self.end;

            Ok((Cow::Borrowed(&self.input[range.clone()]), range))
        };

        match scan()? {
            (s, r) if s.len() > 0 => Ok(Some((s, r))),
            _ => Ok(None),
        }
    }

    fn take_until_eof_or_hole_start(
        &mut self,
    ) -> Result<Option<(Cow<'input, str>, Range<usize>)>, Error> {
        let mut escaped = false;
        let scanned = self.take_until(|state| match state.current {
            // A `{` that's followed by another `{` is escaped
            // If it's followed by a different character then it's
            // the start of an interpolated expression
            '{' => {
                let start = state.current_idx;

                match state.rest.peek().map(|(_, peeked)| *peeked) {
                    Some('{') => {
                        escaped = true;
                        let _ = state.rest.next();
                        Ok(false)
                    }
                    Some(_) => Ok(true),
                    None => Err(Error::incomplete_hole(
                        state
                            .lit
                            .subspan(start..start + 1)
                            .unwrap_or(state.lit.span()),
                    )),
                }
            }
            // A `}` that's followed by another `}` is escaped
            // We should never see these in this parser unless they're escaped
            // If we do it means an interpolated expression is missing its start
            // or it's been improperly escaped
            '}' => match state.rest.peek().map(|(_, peeked)| *peeked) {
                Some('}') => {
                    escaped = true;
                    let _ = state.rest.next();
                    Ok(false)
                }
                Some(_) => Err(Error::unescaped_hole(
                    state
                        .lit
                        .subspan(state.current_idx..state.current_idx + 1)
                        .unwrap_or(state.lit.span()),
                )),
                None => Err(Error::unescaped_hole(
                    state
                        .lit
                        .subspan(state.current_idx..state.current_idx + 1)
                        .unwrap_or(state.lit.span()),
                )),
            },
            _ => Ok(false),
        })?;

        match scanned {
            Some((input, range)) if escaped => {
                // If the input is escaped, then replace `{{` and `}}` chars
                let input = (&*input).replace("{{", "{").replace("}}", "}");
                Ok(Some((Cow::Owned(input), range)))
            }
            scanned => Ok(scanned),
        }
    }

    fn take_until_hole_end(&mut self) -> Result<(Cow<'input, str>, Range<usize>), Error> {
        let mut depth = 1;
        let mut matched_hole_end = false;
        let mut escaped = false;
        let mut next_terminator_escaped = false;
        let mut terminator = None;

        // NOTE: The starting point is the first char _after_ the opening `{`
        // so to get a correct span here we subtract 1 from it to cover that character
        let start = self.start - 1;

        let scanned = self.take_until(|state| {
            match state.current {
                // If the depth would return to its start then we've got a full expression
                '}' if terminator.is_none() && depth == 1 => {
                    matched_hole_end = true;
                    Ok(true)
                }
                // A block end will reduce the depth
                '}' if terminator.is_none() => {
                    depth -= 1;
                    Ok(false)
                }
                // A block start will increase the depth
                '{' if terminator.is_none() => {
                    depth += 1;
                    Ok(false)
                }
                // A double quote may be the start or end of a string
                // It may also be escaped
                '"' if terminator.is_none() => {
                    terminator = Some('"');
                    Ok(false)
                }
                // A single quote may be the start or end of a character
                // It may also be escaped
                '\'' if terminator.is_none() => {
                    terminator = Some('\'');
                    Ok(false)
                }
                // A `\` means there's embedded escaped characters
                // These may be escapes the user needs to represent a `"`
                // or they may be intended to appear in the final string
                '\\' if state
                    .rest
                    .peek()
                    .map(|(_, peeked)| *peeked == '\\')
                    .unwrap_or(false) =>
                {
                    next_terminator_escaped = !next_terminator_escaped;
                    escaped = true;
                    Ok(false)
                }
                '\\' => {
                    escaped = true;
                    Ok(false)
                }
                // The sequence `//` or `/*` means the expression contains a comment
                // These aren't supported so bail with an error
                '/' if state
                    .rest
                    .peek()
                    .map(|(_, peeked)| *peeked == '/' || *peeked == '*')
                    .unwrap_or(false) =>
                {
                    Err(Error::unsupported_comment(
                        state
                            .lit
                            .subspan(state.current_idx..state.current_idx + 1)
                            .unwrap_or(state.lit.span()),
                    ))
                }
                // If the current character is a terminator and it's not escaped
                // then break out of the current string or character
                c if Some(c) == terminator && !next_terminator_escaped => {
                    terminator = None;
                    Ok(false)
                }
                // If the current character is anything else then discard escaping
                // for the next character
                _ => {
                    next_terminator_escaped = false;
                    Ok(false)
                }
            }
        })?;

        if !matched_hole_end {
            Err(Error::incomplete_hole(
                self.lit
                    .subspan(start..self.start)
                    .unwrap_or(self.lit.span()),
            ))?;
        }

        match scanned {
            Some((input, range)) if escaped => {
                // If the input is escaped then replace `\"` with `"`
                let input = (&*input).replace("\\\"", "\"");
                Ok((Cow::Owned(input), range))
            }
            Some((input, range)) => Ok((input, range)),
            None => Err(Error::missing_expr(
                self.lit
                    .subspan(start..self.start)
                    .unwrap_or(self.lit.span()),
            ))?,
        }
    }
}

/**
An error encountered while parsing a template.
 */
#[derive(Debug)]
pub struct Error {
    reason: String,
    source: Option<Box<dyn std::error::Error>>,
    span: Span,
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "parsing failed: {}", self.reason)
    }
}

impl Error {
    pub fn span(&self) -> Span {
        self.span
    }

    fn incomplete_hole(span: Span) -> Self {
        Error {
            reason: format!("unexpected end of input, expected `}}`"),
            source: None,
            span,
        }
    }

    fn unescaped_hole(span: Span) -> Self {
        Error {
            reason: format!("`{{` and `}}` characters must be escaped as `{{{{` and `}}}}`"),
            source: None,
            span,
        }
    }

    fn missing_expr(span: Span) -> Self {
        Error {
            reason: format!("empty replacements (`{{}}`) aren't supported, put the replacement inside like `{{some_value}}`"),
            source: None,
            span,
        }
    }

    fn lex_fv_expr(span: Span, expr: &str, err: proc_macro2::LexError) -> Self {
        Error {
            reason: format!("failed to parse `{}` as a field-value expression", expr),
            span,
            source: Some(format!("{:?}", err).into()),
        }
    }

    fn parse_fv_expr<'a>(span: Span, expr: impl Into<Option<&'a str>>, err: syn::Error) -> Self {
        Error {
            reason: if let Some(expr) = expr.into() {
                format!("failed to parse `{}` as a field-value expression", expr)
            } else {
                format!("failed to parse field-value expression")
            },
            span,
            source: Some(err.into()),
        }
    }

    fn invalid_literal(span: Span) -> Self {
        Error {
            reason: format!("templates must be parsed from string literals"),
            source: None,
            span,
        }
    }

    fn invalid_char(span: Span, expected: &[char]) -> Self {
        Error {
            reason: format!(
                "invalid character, expected: {}",
                Error::display_list(expected)
            ),
            source: None,
            span,
        }
    }

    fn invalid_char_eof(span: Span, expected: &[char]) -> Self {
        Error {
            reason: format!(
                "unexpected end-of-input, expected: {}",
                Error::display_list(expected)
            ),
            source: None,
            span,
        }
    }

    fn unsupported_comment(span: Span) -> Self {
        Error {
            reason: format!("comments within expressions are not supported"),
            source: None,
            span,
        }
    }

    fn display_list<'a>(l: &'a [impl fmt::Display]) -> impl fmt::Display + 'a {
        struct DisplayList<'a, T>(&'a [T]);

        impl<'a, T: fmt::Display> fmt::Display for DisplayList<'a, T> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                match self.0.len() {
                    1 => write!(f, "`{}`", self.0[0]),
                    _ => {
                        let mut first = true;

                        for item in self.0 {
                            if !first {
                                write!(f, ", ")?;
                            }
                            first = false;

                            write!(f, "`{}`", item)?;
                        }

                        Ok(())
                    }
                }
            }
        }

        DisplayList(l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::Member;

    #[test]
    fn parse_ok() {
        let cases = vec![
            (quote!(), None::<&str>),
            (quote!(""), Some("")),
            (quote!("template"), Some("template")),
            (quote!(a: 42, "template"), Some("template")),
            (quote!("template", a: 42), Some("template")),
            (quote!(a: 42, "template", b: 42), Some("template")),
        ];

        for (case, expected) in cases {
            let tpl = Template::parse2(case).unwrap();

            if let Some(expected) = expected {
                let Some(LiteralPart::Text { ref text, .. }) = tpl.literal.get(0) else {
                    panic!(
                        "unexpected template {:?} (expected {:?})",
                        tpl.literal, expected
                    );
                };

                assert_eq!(expected, text);
            } else {
                assert_eq!(0, tpl.literal.len(), "expected an empty template");
            }
        }
    }

    #[test]
    fn parse_err() {
        let cases = vec![
            (
                quote!(42),
                "parsing failed: templates must be parsed from string literals",
            ),
            (
                quote!(a: 42, true),
                "parsing failed: failed to parse field-value expression",
            ),
            (
                quote!(fn x() {}, "template"),
                "parsing failed: failed to parse field-value expression",
            ),
            (
                quote!("template", fn x() {}),
                "parsing failed: failed to parse field-value expression",
            ),
        ];

        for (input, expected) in cases {
            let actual = match Template::parse2(input.clone()) {
                Err(e) => e,
                Ok(_) => panic!("parsing {} should've failed but produced a value", input),
            };

            assert_eq!(expected, actual.to_string(),);
        }
    }

    #[test]
    fn template_parse_ok() {
        let cases = vec![
            ("", vec![text("", 0..0)]),
            ("Hello world 🎈📌", vec![text("Hello world 🎈📌", 1..21)]),
            (
                "Hello {world} 🎈📌",
                vec![
                    text("Hello ", 1..7),
                    hole("world", 8..13),
                    text(" 🎈📌", 14..23),
                ],
            ),
            ("{world}", vec![hole("world", 2..7)]),
            (
                "Hello {#[log::debug] world} 🎈📌",
                vec![
                    text("Hello ", 1..7),
                    hole("#[log::debug] world", 8..27),
                    text(" 🎈📌", 28..37),
                ],
            ),
            (
                "Hello {#[log::debug] world: 42} 🎈📌",
                vec![
                    text("Hello ", 1..7),
                    hole("#[log::debug] world: 42", 8..31),
                    text(" 🎈📌", 32..41),
                ],
            ),
            (
                "Hello {#[log::debug] world: \"is text\"} 🎈📌",
                vec![
                    text("Hello ", 1..7),
                    hole("#[log::debug] world: \"is text\"", 8..40),
                    text(" 🎈📌", 41..50),
                ],
            ),
            (
                "{Hello} {world}",
                vec![hole("Hello", 2..7), text(" ", 8..9), hole("world", 10..15)],
            ),
            (
                "{a}{b}{c}",
                vec![hole("a", 2..3), hole("b", 5..6), hole("c", 8..9)],
            ),
            (
                "🎈📌{a}🎈📌{b}🎈📌{c}🎈📌",
                vec![
                    text("🎈📌", 1..9),
                    hole("a", 10..11),
                    text("🎈📌", 12..20),
                    hole("b", 21..22),
                    text("🎈📌", 23..31),
                    hole("c", 32..33),
                    text("🎈📌", 34..42),
                ],
            ),
            (
                "Hello 🎈📌 {{world}}",
                vec![text("Hello 🎈📌 {world}", 1..25)],
            ),
            (
                "🎈📌 Hello world {{}}",
                vec![text("🎈📌 Hello world {}", 1..26)],
            ),
            (
                "Hello {#[log::debug] world: \"{\"} 🎈📌",
                vec![
                    text("Hello ", 1..7),
                    hole("#[log::debug] world: \"{\"", 8..34),
                    text(" 🎈📌", 35..44),
                ],
            ),
            (
                "Hello {#[log::debug] world: '{'} 🎈📌",
                vec![
                    text("Hello ", 1..7),
                    hole("#[log::debug] world: '{'", 8..32),
                    text(" 🎈📌", 33..42),
                ],
            ),
            (
                "Hello {#[log::debug] world: \"is text with 'embedded' stuff\"} 🎈📌",
                vec![
                    text("Hello ", 1..7),
                    hole(
                        "#[log::debug] world: \"is text with 'embedded' stuff\"",
                        8..62,
                    ),
                    text(" 🎈📌", 63..72),
                ],
            ),
            ("{{", vec![text("{", 1..3)]),
            ("}}", vec![text("}", 1..3)]),
        ];

        for (template, expected) in cases {
            let actual = match LiteralPart::parse_lit2(Literal::string(template)) {
                Ok(template) => template,
                Err(e) => panic!("failed to parse {:?}: {}", template, e),
            };

            assert_eq!(
                format!("{:?}", expected),
                format!("{:?}", actual),
                "parsing template: {:?}",
                template
            );
        }
    }

    #[test]
    fn template_parse_err() {
        let cases = vec![
            ("{", "parsing failed: unexpected end of input, expected `}`"),
            ("a {", "parsing failed: unexpected end of input, expected `}`"),
            ("a { a", "parsing failed: unexpected end of input, expected `}`"),
            ("{ a", "parsing failed: unexpected end of input, expected `}`"),
            ("}", "parsing failed: `{` and `}` characters must be escaped as `{{` and `}}`"),
            ("} a", "parsing failed: `{` and `}` characters must be escaped as `{{` and `}}`"),
            ("a } a", "parsing failed: `{` and `}` characters must be escaped as `{{` and `}}`"),
            ("a }", "parsing failed: `{` and `}` characters must be escaped as `{{` and `}}`"),
            ("{}", "parsing failed: empty replacements (`{}`) aren\'t supported, put the replacement inside like `{some_value}`"),
            ("{not real rust}", "parsing failed: failed to parse `not real rust` as a field-value expression"),
            ("{// a comment!}", "parsing failed: comments within expressions are not supported"),
            ("{/* a comment! */}", "parsing failed: comments within expressions are not supported"),
        ];

        for (template, expected) in cases {
            let actual = match LiteralPart::parse_lit2(Literal::string(template)) {
                Err(e) => e,
                Ok(actual) => panic!(
                    "parsing {:?} should've failed but produced {:?}",
                    template, actual
                ),
            };

            assert_eq!(
                expected,
                actual.to_string(),
                "parsing template: {:?}",
                template
            );
        }
    }

    fn text(text: &str, range: Range<usize>) -> LiteralPart {
        LiteralPart::Text {
            text: text.to_owned(),
            range,
        }
    }

    fn hole(expr: &str, range: Range<usize>) -> LiteralPart {
        LiteralPart::Hole {
            expr: syn::parse_str(expr)
                .unwrap_or_else(|e| panic!("failed to parse {:?} ({})", expr, e)),
            range,
        }
    }

    #[test]
    fn visit_literal() {
        fn to_rt_tokens(template: &Template, base: TokenStream) -> TokenStream {
            struct DefaultVisitor {
                base: TokenStream,
                parts: Vec<TokenStream>,
            }

            impl LiteralVisitor for DefaultVisitor {
                fn visit_text(&mut self, text: &str) {
                    let base = &self.base;

                    self.parts.push(quote!(#base::Part::Text(#text)));
                }

                fn visit_hole(&mut self, hole: &FieldValue) {
                    let hole = match hole.member {
                        Member::Named(ref member) => member.to_string(),
                        Member::Unnamed(ref member) => member.index.to_string(),
                    };

                    let base = &self.base;

                    self.parts.push(quote!(#base::Part::Hole(#hole)));
                }
            }

            let mut visitor = DefaultVisitor {
                base,
                parts: Vec::new(),
            };
            template.visit_literal(&mut visitor);

            let base = &visitor.base;
            let parts = &visitor.parts;

            quote!(
                #base::Template(&[#(#parts),*])
            )
        }

        let cases = vec![(
            quote!("text and {label} and {more: 42}"),
            quote!(crate::rt::Template(&[
                crate::rt::Part::Text("text and "),
                crate::rt::Part::Hole("label"),
                crate::rt::Part::Text(" and "),
                crate::rt::Part::Hole("more")
            ])),
        )];

        for (template, expected) in cases {
            let template = Template::parse2(template).unwrap();

            assert!(template.has_literal());

            assert_eq!(
                expected.to_string(),
                to_rt_tokens(&template, quote!(crate::rt)).to_string()
            );
        }
    }

    #[test]
    fn visit_literal_empty() {
        struct DefaultVisitor {
            called: bool,
        }

        impl LiteralVisitor for DefaultVisitor {
            fn visit_text(&mut self, _: &str) {
                self.called = true;
            }

            fn visit_hole(&mut self, _: &FieldValue) {
                unreachable!()
            }
        }

        let mut visitor = DefaultVisitor { called: false };

        let template = Template::parse2(quote!("")).unwrap();

        template.visit_literal(&mut visitor);

        assert!(template.has_literal());
        assert!(visitor.called);
    }

    #[test]
    fn visit_literal_none() {
        struct DefaultVisitor {
            called: bool,
        }

        impl LiteralVisitor for DefaultVisitor {
            fn visit_text(&mut self, _: &str) {
                unreachable!()
            }

            fn visit_hole(&mut self, _: &FieldValue) {
                unreachable!()
            }
        }

        let mut visitor = DefaultVisitor { called: false };

        let template = Template::parse2(quote!()).unwrap();

        template.visit_literal(&mut visitor);

        assert!(!template.has_literal());
        assert!(!visitor.called);
    }
}
