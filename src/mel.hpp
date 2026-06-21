#pragma once
#include <vector>

#include "model_loader.hpp"

namespace ced {

// Compute the CED log-mel "input_values" from a mono 16 kHz waveform, matching
// torchaudio MelSpectrogram(center=True, power=2, HTK mel) + AmplitudeToDB(
// stype="power", top_db=120). `window` is the baked Hann window [win_size];
// `filterbank` is the baked mel filterbank, mel-major [n_mels * n_freqs]
// (row m, freq f at m*n_freqs + f). Output `input_values` is [n_mels * T],
// mel-major (m*T + t); `T` is the number of frames.
void mel_spectrogram(const CedConfig& cfg, const float* window, const float* filterbank,
                     const std::vector<float>& wav, std::vector<float>& input_values, int& T);

}  // namespace ced
