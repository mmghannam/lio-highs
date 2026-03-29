importScripts('highs_wasm.js');

let Module = null;

HiGHSWasm().then(mod => {
    Module = mod;
    postMessage({type: 'ready'});
}).catch(err => {
    postMessage({type: 'error', message: 'Failed to load WASM: ' + err});
});

onmessage = function(e) {
    if (e.data.type !== 'solve') return;

    const {content, filename} = e.data;

    Module.FS.writeFile(filename, content);

    const len = Module.lengthBytesUTF8(filename);
    const ptr = Module._highs_wasm_alloc(len + 1);
    Module.stringToUTF8(filename, ptr, len + 1);

    const status = Module._highs_wasm_solve(ptr, len);
    Module._highs_wasm_free(ptr, len + 1);

    const obj = Module._highs_wasm_get_obj_value();
    postMessage({type: 'done', status, objective: obj});
};
