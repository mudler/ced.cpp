// Smoke + parity test for the flat C-API: load -> classify PCM -> top tag.
// argv[1] = model gguf, argv[2] = baseline gguf
#include "ced_capi.h"
#include "parity.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string model = argc > 1 ? argv[1] : "models/ced-base-f32.gguf";
    std::string baseline = argc > 2 ? argv[2] : "tests/fixtures/ced-base.baseline.gguf";

    ced_ctx* ctx = ced_capi_load(model.c_str());
    if (!ctx) {
        std::fprintf(stderr, "FAIL: load: %s\n", ced_capi_last_error(nullptr));
        return 1;
    }
    bool ok = true;
    ok &= (ced_capi_abi_version() == 1);
    ok &= (ced_capi_num_classes(ctx) == 527);
    ok &= (ced_capi_sample_rate(ctx) == 16000);

    std::vector<float> wav, refp;
    std::vector<int64_t> shp;
    if (!cedtest::load_baseline(baseline, "audio_waveform", wav, shp) ||
        !cedtest::load_baseline(baseline, "probs", refp, shp)) {
        ced_capi_free(ctx);
        return 1;
    }
    int ref_argmax = 0;
    for (int i = 1; i < (int)refp.size(); ++i)
        if (refp[i] > refp[ref_argmax]) ref_argmax = i;

    // struct-array path
    ced_tag tags[5];
    int n = ced_capi_classify_pcm(ctx, wav.data(), (int)wav.size(), 16000, tags, 5);
    if (n <= 0) {
        std::fprintf(stderr, "FAIL: classify_pcm: %s\n", ced_capi_last_error(ctx));
        ced_capi_free(ctx);
        return 1;
    }
    std::fprintf(stderr, "top tag: [%d] %s score=%.4f (ref argmax=%d)\n", tags[0].index,
                 tags[0].label, tags[0].score, ref_argmax);
    ok &= (tags[0].index == ref_argmax);

    // JSON path
    char* json = ced_capi_classify_pcm_json(ctx, wav.data(), (int)wav.size(), 16000, 3);
    if (!json) { ok = false; } else {
        std::fprintf(stderr, "json: %s\n", json);
        ok &= (std::strstr(json, "\"label\"") != nullptr);
        ced_capi_free_string(json);
    }

    ced_capi_free(ctx);
    std::fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
