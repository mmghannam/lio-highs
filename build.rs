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

    if cfg!(feature = "inst-count") {
        // Load the LLVM pass plugin for instruction counting.
        // Looks in ~/.inst-count/ (installed by `cargo inst-count setup`)
        // or the INST_COUNT_PASS_PLUGIN env var.
        let plugin = env::var("INST_COUNT_PASS_PLUGIN").unwrap_or_else(|_| {
            let home = env::var("HOME")
                .or_else(|_| env::var("USERPROFILE"))
                .expect("cannot determine home directory");
            let ext = if cfg!(target_os = "macos") { "dylib" } else { "so" };
            format!("{home}/.inst-count/libInstructionCountPass.{ext}")
        });
        // Use CMAKE_*_FLAGS_RELEASE so the pass isn't loaded during CMake's
        // compiler detection (try_compile), which would fail to link __inst_count_add.
        let flag = format!("-fpass-plugin={plugin}");
        dst.define("CMAKE_C_FLAGS_RELEASE",
                   format!("-O3 -DNDEBUG {flag}"));
        dst.define("CMAKE_CXX_FLAGS_RELEASE",
                   format!("-O3 -DNDEBUG {flag}"));
        // Allow undefined __inst_count_add during HiGHS linking — it will be
        // resolved when the final Rust binary links the inst-count runtime.
        if cfg!(target_os = "macos") {
            dst.define("CMAKE_EXE_LINKER_FLAGS", "-Wl,-undefined,dynamic_lookup");
            dst.define("CMAKE_SHARED_LINKER_FLAGS", "-Wl,-undefined,dynamic_lookup");
        } else {
            dst.define("CMAKE_EXE_LINKER_FLAGS", "-Wl,--allow-shlib-undefined");
            dst.define("CMAKE_SHARED_LINKER_FLAGS", "-Wl,--allow-shlib-undefined");
        }

        // The pass plugin must be loaded by the same LLVM version it was built against.
        // On macOS, use Homebrew's clang instead of Apple Clang.
        let llvm_bin = env::var("INST_COUNT_LLVM_BIN")
            .unwrap_or_else(|_| "/opt/homebrew/opt/llvm/bin".to_string());
        dst.define("CMAKE_C_COMPILER", format!("{llvm_bin}/clang"));
        dst.define("CMAKE_CXX_COMPILER", format!("{llvm_bin}/clang++"));
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
        if cfg!(feature = "inst-count") {
            // When inst-count is active, HiGHS was compiled with Homebrew's clang
            // which uses a newer libc++. Link against Homebrew's libc++ to match.
            let llvm_lib = env::var("INST_COUNT_LLVM_LIB")
                .unwrap_or_else(|_| "/opt/homebrew/opt/llvm/lib/c++".to_string());
            println!("cargo:rustc-link-search=native={llvm_lib}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{llvm_lib}");
            println!("cargo:rustc-link-lib=dylib=c++");
        } else {
            println!("cargo:rustc-link-lib=dylib=c++");
        }
    } else if linux || mingw {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    println!("cargo:rerun-if-changed=HiGHS/highs/interfaces/highs_c_api.h");
    // Re-run when RUSTC_WRAPPER changes so OUT_DIR/c_bindings.rs is regenerated
    // (cargo inst-count sets RUSTC_WRAPPER which changes cargo's fingerprint hash)
    println!("cargo:rerun-if-env-changed=RUSTC_WRAPPER");
}
