// Writes what the loader makes of tensors of a GGUF (src/tensor_convert.hpp): for each tensor named on the command line, the
// converted tensor's values as float32 to OUTDIR/<name>.f32 plus a line "name type_after nbytes" on stdout. tests/split/convert_check.py
// compares those with gguf-py's own dequantization of the same tensors in the file. CPU only.
//   bin/convert_dump MODEL.gguf OUTDIR name [name ...]      (a name of "all" dumps every tensor the loader converted)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../src/gguf.hpp"
#include "../src/tensor_convert.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: convert_dump MODEL.gguf OUTDIR name [name ...]\n");
        return 2;
    }
    try {
        fnx::Gguf g(argv[1]);
        std::printf("converted %d\n", g.converted());
        std::vector<std::string> names;
        for (int i = 3; i < argc; ++i) {
            if (std::string(argv[i]) == "all") {
                for (const auto& kv : g.tensors())
                    if (fnx::conv::wanted_type(kv.first) != 0xFFFFFFFFu) names.push_back(kv.first);
            } else {
                names.push_back(argv[i]);
            }
        }
        for (const std::string& n : names) {
            const fnx::Tensor& t = g.t(n);
            size_t cnt = 1;
            for (int d = 0; d < 4; ++d) cnt *= (size_t) t.ne[d];
            std::vector<float> f(cnt);
            fnx::conv::dequantize(t.type, t.data, cnt, f.data());
            const std::string path = std::string(argv[2]) + "/" + n + ".f32";
            FILE* fp = std::fopen(path.c_str(), "wb");
            if (!fp) throw std::runtime_error("cannot write " + path);
            std::fwrite(f.data(), 4, cnt, fp);
            std::fclose(fp);
            std::printf("%s %u %zu\n", n.c_str(), t.type, t.nbytes);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "convert_dump: %s\n", e.what());
        return 1;
    }
    return 0;
}
