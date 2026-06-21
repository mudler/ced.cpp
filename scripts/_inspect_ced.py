import sys, json
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, AutoConfig

mid = "mispeech/ced-base"
print("== loading", mid, file=sys.stderr)
cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
print("== CONFIG ==")
print(json.dumps(cfg.to_dict(), indent=2, default=str))

model = AutoModelForAudioClassification.from_pretrained(mid, trust_remote_code=True)
model.eval()
sd = model.state_dict()
print("\n== STATE DICT (%d tensors) ==" % len(sd))
for k, v in sd.items():
    print(f"{k}\t{tuple(v.shape)}\t{v.dtype}")

try:
    fe = AutoFeatureExtractor.from_pretrained(mid, trust_remote_code=True)
    print("\n== FEATURE EXTRACTOR ==")
    print(json.dumps(fe.to_dict(), indent=2, default=str))
except Exception as e:
    print("\n== FEATURE EXTRACTOR load failed:", e)
