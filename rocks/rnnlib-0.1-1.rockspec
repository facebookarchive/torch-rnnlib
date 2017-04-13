package = "rnnlib"
version = "0.1-1"

source = {
    url = "git://github.com/facebookresearch/torch-rnnlib",
}

description = {
    summary = "torch-rnnlib",
    detailed = [[
    ]],
    homepage = "https://github.com/facebookresearch/torch-rnnlib",
    license = "BSD",
}

dependencies = {
    "torch >= 7.0",
    "nn >= 1.0 ",
    "nngraph >= 1.0",
}

build = {
    type = "command",
    build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
    ]],
    install_command = "cd build && $(MAKE) install"
}
