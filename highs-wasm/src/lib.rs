use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

use lio_highs::ffi::*;

static OBJ_VALUE: std::sync::Mutex<f64> = std::sync::Mutex::new(0.0);

extern "C" {
    fn js_on_log(ptr: *const u8, len: usize);
}

unsafe extern "C" fn log_callback(
    _callback_type: c_int,
    message: *const c_char,
    _data_out: *const HighsCallbackDataOut,
    _data_in: *mut HighsCallbackDataIn,
    _user_data: *mut c_void,
) {
    if !message.is_null() {
        let msg = unsafe { std::ffi::CStr::from_ptr(message) }.to_bytes();
        unsafe { js_on_log(msg.as_ptr(), msg.len()) };
    }
}

#[no_mangle]
pub extern "C" fn highs_wasm_solve(filename_ptr: *const u8, filename_len: usize) -> i32 {
    *OBJ_VALUE.lock().unwrap() = 0.0;

    let filename = unsafe {
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(filename_ptr, filename_len))
    };
    let c_filename = CString::new(filename).unwrap();

    unsafe {
        let highs = Highs_create();
        Highs_setCallback(highs, Some(log_callback), std::ptr::null_mut());
        Highs_startCallback(highs, kHighsCallbackLogging);

        let status = Highs_readModel(highs, c_filename.as_ptr());
        if status != 0 {
            Highs_destroy(highs);
            return status;
        }

        let status = Highs_run(highs);
        *OBJ_VALUE.lock().unwrap() = Highs_getObjectiveValue(highs);

        Highs_destroy(highs);
        status
    }
}

#[no_mangle]
pub extern "C" fn highs_wasm_get_obj_value() -> f64 {
    *OBJ_VALUE.lock().unwrap()
}

#[no_mangle]
pub extern "C" fn highs_wasm_alloc(size: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[no_mangle]
pub extern "C" fn highs_wasm_free(ptr: *mut u8, size: usize) {
    unsafe {
        drop(Vec::from_raw_parts(ptr, 0, size));
    }
}
