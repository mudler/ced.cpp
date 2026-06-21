#pragma once
#include <string>
#include <vector>

namespace ced {

struct Audio {
    std::vector<float> samples;  // mono, float, [-1,1]
    int sample_rate = 0;
};

std::vector<float> resample_linear(const std::vector<float>& in, int in_sr, int out_sr);

// Load a WAV file as mono float PCM resampled to `target_sr` (default 16 kHz).
bool load_audio_mono(const std::string& path, Audio& out, int target_sr = 16000);

}  // namespace ced
