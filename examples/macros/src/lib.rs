#[macro_use]
extern crate quote;

use proc_macro2::TokenStream;
use syn::{FieldValue, spanned::Spanned};

use fv_template::{Template, TemplateVisitor};

#[proc_macro]
pub fn template_args(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_template_args(TokenStream::from(input))
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand_template_args(input: TokenStream) -> syn::Result<TokenStream> {
    let span = input.span();

    // First, parse the template
    let template = Template::parse2(input)
        .map_err(|e| syn::Error::new(span, e))?;

    // In this example, we only want to support expressions in the format string itself
    if template.before_template_field_values().count() > 0 || template.after_template_field_values().count() > 0 {
        return Err(syn::Error::new(span, "arguments outside the template string are not supported"));
    }

    // Create a visitor that will get each fragment of the template string
    // Using this visitor, we build up an equivalent `format_args!` call
    let mut visitor = FormatVisitor {
        fmt: String::new(),
        args: Vec::new(),
    };

    template.visit_template(&mut visitor);

    let fmt = visitor.fmt;
    let args = visitor.args;

    Ok(quote!(format_args!(#fmt, #(#args),*)))
}

struct FormatVisitor {
    fmt: String,
    args: Vec<TokenStream>,
}

impl TemplateVisitor for FormatVisitor {
    fn visit_hole(&mut self, hole: &FieldValue) {
        let hole = &hole.expr;

        self.fmt.push_str("{}");
        self.args.push(quote!(#hole));
    }

    fn visit_text(&mut self, text: &str) {
        self.fmt.push_str(text);
    }
}
