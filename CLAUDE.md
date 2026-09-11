# LibraShogi

天秤将棋（https://fusekishogi.com/rules/）の AI「Libra」。設計は docs/libra-design.md、実行計画は docs/libra-local.md。両方を読んでから作業する。

## 不変の制約
- ゼロから開発。既存将棋AIのコード・評価関数・重み・棋譜を内部に使わない。GPL コードの流用禁止。
- ライセンス: コード Apache-2.0 / 文書 CC BY 4.0 / 自己対局データ CC0 / 重み Apache-2.0。依存追加時は LICENSES/ を更新。
- 水匠5・fuseki-shogi-ai は対戦相手専用（別プロセス、USI）。学習信号にしない。
- 秘密情報をコミットしない。v0.1 まで非公開、v0.1 で公開。
- 終局規定は docs/rules.md（大会規定: 千日手4回＝引き分け、連続王手は王手側負け、入玉宣言法27点、本将棋 320 手で引き分け。41 手目を 1 手目として数える）が唯一の正。マッチの裁定はハーネスが行う。

## 構成
libra-sim（C++ シミュレータ、pybind11）/ libra-net / libra-search（MCGS、df-pn、配置詰み、USI 拡張エンジン）/ libra-league（自己対局、リーグ、評価）/ libra-scale（玉配置表）/ libra-cloud（vast.ai）/ docs / data（マニフェストのみ）

## 作業ルール
- テストを先に書く。perft とルールテストは変更のたびに全部通す。
- 長時間実行は docs/libra-local.md §7 の停止・再開設計に従う。
- 決定は docs/decisions.md、実測値は docs/measurements.md に追記。
- ユーザーに確認が要る事項: 対局棋譜の公開可否、vast.ai の利用開始、desktop への終局判定追加の時期。

## 環境
WSL2 Ubuntu（RAM 32 GB）、RTX 5070 Ti、Ryzen 9 9950X3D。Windows バイナリは GitHub Actions windows-latest でビルド。データは WSL 内の ext4 に置く。

## 現在の目標
2026年内に Libra-L（自己対局のみで学習）を USI 拡張エンジンとして desktop に組み込み、v0.1 として公開する。「fuseki-shogi-ai 方策ネット＋水匠5」との100局は計測であり、勝てればよい。相手の分析や相手専用の対策はしない（docs/libra-local.md §4）。