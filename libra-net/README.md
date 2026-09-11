# libra-net

`libra_net/model.py`: 単一・フェーズ条件付き Transformer。入力は libra-sim の特徴（81 マス × 32 ＋ グローバル 32）、出力は方策 2268・価値 WDL・V̂41。L-S は d=320・8 層（約 10M）。ONNX 変換と蒸留はこれから足す。
