use std::env;
use std::path::PathBuf;

fn generate_bindings(include_path: &std::path::Path) {
    let c_bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}", include_path.to_string_lossy()))
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    c_bindings
        .write_to_file(out_path.join("c_bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn main() {
    use cmake::Config;
    let mut dst = Config::new("HiGHS");

    if cfg!(feature = "ninja") {
        dst.generator("Ninja");
    }

    if cfg!(feature = "highs_release") {
        dst.profile("Release");
    }

    let dst = dst
        .define("FAST_BUILD", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL")
        .define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "FALSE")
        .define("ZLIB", if cfg!(feature = "libz") { "ON" } else { "OFF" })
        .build();

    let include_path = dst.join("include").join("highs");
    generate_bindings(&include_path);

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-lib=static=highs");

    if cfg!(feature = "libz") {
        println!("cargo:rustc-link-lib=z");
    }

    let target = env::var("TARGET").unwrap();
    let apple = target.contains("apple");
    let linux = target.contains("linux");
    let mingw = target.contains("pc-windows-gnu");
    if apple {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if linux || mingw {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    println!("cargo:rerun-if-changed=HiGHS/src/interfaces/highs_c_api.h");
}
