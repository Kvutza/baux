#[path = "link_rpath.rs"]
mod link_rpath;

fn main() {
    link_rpath::emit_args();
}
