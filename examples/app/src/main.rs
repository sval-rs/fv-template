#[macro_use]
extern crate fv_template_example_macros;

fn main() {
    println!("{}", template_args!("Hello, {name: \"World\"}"));
}
