import cupy as cp
import os

kernel_dir = os.path.abspath(os.path.dirname(__file__))
compile_options =['-std=c++14', '-DPTYPY_CUPY_NVTRC=1', '-I' + kernel_dir, '-DNDEBUG']

def load_kernel(name, subs={}, file=None, options=None):

    if file is None:
        if isinstance(name, str):
            fn = "%s/%s.cu" % (kernel_dir, name)
        else:
            raise ValueError(
                "name parameter must be a string if not filename is given")
    else:
        fn = "%s/%s" % (kernel_dir, file)

    with open(fn, 'r') as f:
        kernel = f.read()
    for k, v in list(subs.items()):
        kernel = kernel.replace(k, str(v))
    # insert a preprocessor line directive to assist compiler errors
    escaped = fn.replace("\\", "\\\\")
    kernel = '#line 1 "{}"\n'.format(escaped) + kernel

    opt = [*compile_options]
    if options is not None:
        opt += list(options)
    module = cp.RawModule(code=kernel, options=tuple(opt))
    if isinstance(name, str):
        return module.get_function(name)
    else:  # tuple
        return tuple(module.get_function(n) for n in name)

