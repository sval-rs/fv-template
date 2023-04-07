/*!
Compile-time string template formatting.
*/

use std::{
    borrow::Cow,
    fmt,
    iter::Peekable,
    ops::Range,
    str::{self, CharIndices},
};

use proc_macro2::{token_stream, Literal, Span, TokenStream, TokenTree};
use quote::ToTokens;
use syn::{spanned::Spanned, ExprLit, FieldValue, Lit, LitStr, Member};
use thiserror::Error;

/**
An error encountered while parsing a template.
*/
#[derive(Error, Debug)]
#[error("parsing failed: {reason}")]
pub struct Error {
    reason: String,
    source: Option<Box<dyn std::error::Error>>,
    span: Span,
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

    fn lex_expr(span: Span, expr: &str, err: proc_macro2::LexError) -> Self {
        Error {
            reason: format!("failed to parse `{}` as an expression", expr),
            span,
            source: Some(format!("{:?}", err).into()),
        }
    }

    fn parse_expr(span: Span, expr: &str, err: syn::Error) -> Self {
        Error {
            reason: format!("failed to parse `{}` as an expression", expr),
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
            reason: format!("invalid character, expected: {}", display_list(expected)),
            source: None,
            span,
        }
    }

    fn invalid_char_eof(span: Span, expected: &[char]) -> Self {
        Error {
            reason: format!(
                "unexpected end-of-input, expected: {}",
                display_list(expected)
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
}

/**
A compile-time field value template.
*/
pub struct Template {
    before_template: Vec<FieldValue>,
    template: Vec<Part>,
    after_template: Vec<FieldValue>,
}

impl Template {
    /**
    Parse a template from a `TokenStream`.

    The `TokenStream` is typically all the tokens given to a macro.
    */
    pub fn parse2(input: TokenStream) -> Result<Self, Error> {
        struct Scan {
            span: Span,
            iter: Peekable<token_stream::IntoIter>,
        }

        impl Scan {
            fn new(input: TokenStream) -> Self {
                Scan {
                    span: input.span(),
                    iter: input.into_iter().peekable(),
                }
            }

            fn has_input(&mut self) -> bool {
                self.iter.peek().is_some()
            }

            fn take_until(
                &mut self,
                mut until_true: impl FnMut(&TokenTree) -> bool,
            ) -> (TokenStream, Option<TokenTree>) {
                let mut taken = TokenStream::new();

                while let Some(tt) = self.iter.next() {
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
                match self.iter.next() {
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

            fn collect_field_values(mut self) -> Vec<FieldValue> {
                let mut result = Vec::new();

                while self.has_input() {
                    let (arg, _) = self.take_until(|tt| Self::is_punct(&tt, ','));

                    if !arg.is_empty() {
                        result.push(syn::parse2::<FieldValue>(arg).unwrap());
                    }
                }

                result
            }
        }

        let mut scan = Scan::new(input);

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
            scan.iter.collect()
        } else {
            TokenStream::new()
        };

        let before_template = Scan::new(before_template).collect_field_values();
        let after_template = Scan::new(after_template).collect_field_values();

        let template = Part::parse_lit2(Scan::take_literal(
            template.expect("missing string template"),
        )?)?;

        Ok(Template {
            before_template,
            template,
            after_template,
        })
    }

    /**
    Field values that appear before the template string literal.
    */
    pub fn before_template_field_values<'a>(&'a self) -> impl Iterator<Item = &'a FieldValue> {
        self.before_template.iter()
    }

    /**
    Field values that appear within the template string literal.
    */
    pub fn template_field_values<'a>(&'a self) -> impl Iterator<Item = &'a FieldValue> {
        self.template.iter().filter_map(|part| {
            if let Part::Hole { expr, .. } = part {
                Some(expr)
            } else {
                None
            }
        })
    }

    /**
    Field values that appear after the template string literal.
    */
    pub fn after_template_field_values<'a>(&'a self) -> impl Iterator<Item = &'a FieldValue> {
        self.after_template.iter()
    }

    /**
    Generate a `TokenStream` that constructs a runtime representation of this template.
    */
    pub fn to_rt_tokens(&self, base: TokenStream) -> TokenStream {
        struct DefaultVisitor;

        impl Visitor for DefaultVisitor {}

        self.to_rt_tokens_with_visitor(base, DefaultVisitor)
    }

    /**
    Generate a `TokenStream` the constructs a runtime representation of this template.

    The `Visitor` has a chance to modify fragments of the template during code generation.
    */
    pub fn to_rt_tokens_with_visitor(
        &self,
        base: TokenStream,
        mut visitor: impl Visitor,
    ) -> TokenStream {
        let parts = self.template.iter().map(|part| match part {
            Part::Text { text, .. } => quote!(#base::Part::Text(#text)),
            Part::Hole { expr, .. } => {
                let (label, hole) = match expr.member {
                    Member::Named(ref member) => (
                        member.to_string(),
                        ExprLit {
                            attrs: vec![],
                            lit: Lit::Str(LitStr::new(&member.to_string(), member.span())),
                        },
                    ),
                    Member::Unnamed(ref member) => (
                        member.index.to_string(),
                        ExprLit {
                            attrs: vec![],
                            lit: Lit::Str(LitStr::new(&member.index.to_string(), member.span)),
                        },
                    ),
                };

                visitor.visit_hole(&label, quote!(#base::Part::Hole(#hole)))
            }
        });

        quote!(
            #base::template(&[#(#parts),*])
        )
    }
}

/**
A visitor for the construction of a runtime template.
*/
pub trait Visitor {
    /**
    Visit a hole.
    */
    fn visit_hole(&mut self, label: &str, hole: TokenStream) -> TokenStream {
        let _ = label;
        hole
    }
}

impl<'a, V: ?Sized> Visitor for &'a mut V
where
    V: Visitor,
{
    fn visit_hole(&mut self, label: &str, hole: TokenStream) -> TokenStream {
        (**self).visit_hole(label, hole)
    }
}

/**
A part of a parsed template.
*/
pub(super) enum Part {
    /**
    A fragment of text.
    */
    Text { text: String, range: Range<usize> },
    /**
    A replacement expression.
    */
    Hole {
        // TODO: Set the span on this properly
        expr: FieldValue,
        range: Range<usize>,
    },
}

impl fmt::Debug for Part {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Part::Text { text, range } => f
                .debug_struct("Text")
                .field("text", text)
                .field("range", range)
                .finish(),
            Part::Hole { expr, range } => f
                .debug_struct("Hole")
                .field("expr", &format_args!("`{}`", expr.to_token_stream()))
                .field("range", range)
                .finish(),
        }
    }
}

impl Part {
    fn parse_lit2(lit: Literal) -> Result<Vec<Self>, Error> {
        struct Scan<'input> {
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

        impl<'input> Scan<'input> {
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

        enum Expecting {
            TextOrEOF,
            Hole,
        }

        let input = lit.to_string();

        let mut parts = Vec::new();
        let mut expecting = Expecting::TextOrEOF;

        if input.len() == 0 {
            return Ok(parts);
        }

        let mut iter = input.char_indices();
        let start = iter.next();
        let end = iter.next_back();

        // This just checks that we're looking at a string
        // It doesn't bother with ensuring that last quote is unescaped
        // because the input to this is expected to be a proc-macro literal
        if start.map(|(_, c)| c) != Some('"') || end.map(|(_, c)| c) != Some('"') {
            return Err(Error::invalid_literal(lit.span()));
        }

        // NOTE: The input is wrapped in quotes, so the actual content
        // is from `input[1..input.len() - 1]`
        let mut scan = Scan {
            lit,
            input: &input,
            start: 1,
            end: input.len() - 1,
            rest: iter.peekable(),
        };

        while scan.has_input() {
            match expecting {
                Expecting::TextOrEOF => {
                    if let Some((text, range)) = scan.take_until_eof_or_hole_start()? {
                        parts.push(Part::Text {
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
                            Error::lex_expr(expr_span.unwrap_or(scan.lit.span()), &*expr, e)
                        })?;

                        // Set the span to the correct place within the literal
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
                        Error::parse_expr(expr_span.unwrap_or(scan.lit.span()), &*expr, e)
                    })?;

                    parts.push(Part::Hole { expr, range });

                    expecting = Expecting::TextOrEOF;
                    continue;
                }
            }
        }

        Ok(parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ok() {
        let cases = vec![
            ("", vec![]),
            ("", vec![]),
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
            let actual = match Part::parse_lit2(Literal::string(template)) {
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
    fn parse_err() {
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
            ("{not real rust}", "parsing failed: failed to parse `not real rust` as an expression"),
            ("{// a comment!}", "parsing failed: comments within expressions are not supported"),
            ("{/* a comment! */}", "parsing failed: comments within expressions are not supported"),
        ];

        for (template, expected) in cases {
            let actual = match Part::parse_lit2(Literal::string(template)) {
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

    fn text(text: &str, range: Range<usize>) -> Part {
        Part::Text {
            text: text.to_owned(),
            range,
        }
    }

    fn hole(expr: &str, range: Range<usize>) -> Part {
        Part::Hole {
            expr: syn::parse_str(expr)
                .unwrap_or_else(|e| panic!("failed to parse {:?} ({})", expr, e)),
            range,
        }
    }
}
