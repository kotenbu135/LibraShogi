# libra-net

| ファイル | 内容 |
|---|---|
| `libra_net/model.py` | 単一・フェーズ条件付き Transformer（2D 相対位置バイアス）。入力は libra-sim の特徴（81 マス × 32 ＋ グローバル 32）、出力は方策 2268・価値 WDL・V̂41。L-S は d=320・8 層（約 10M、`libra_league/config.py` の `[net]`） |
| `libra_net/losses.py` | 損失: 方策（全読み局面だけ、疎な改善方策とのクロスエントロピー）、価値 WDL、補助 V̂41（布石局面だけ） |
| `libra_net/export_onnx.py` | チェックポイント（.pt）→ 推論用 ONNX（fp32、opset 17、バッチ可変、メタデータに step とネット設定）。`bin/libra export` から呼ぶ。libra-engine はこの形だけを読む |

学習ループ（AdamW、bf16）は `libra-league/libra_league/trainer.py`。蒸留は未実装。

テスト: `tests/test_export.py`（ONNX 書き出しと PyTorch との出力の一致）。
