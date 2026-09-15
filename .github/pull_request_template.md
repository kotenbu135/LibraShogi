## 何を・なぜ

<!-- 変更の内容と理由。決定や実測値があれば docs/decisions.md・docs/measurements.md に 1 行ずつ足す -->

## 確認（CONTRIBUTING.md）

- [ ] 追加した依存のライセンスを `LICENSES/README.md` に記録し、全文を `LICENSES/` に置いた（依存を足していなければチェック）
- [ ] 既存の将棋 AI のコード・評価関数・重み・棋譜・評価値を含んでいない（クリーンルーム方針）
- [ ] 秘密情報（API キー、SSH 鍵、ホストの IP）を含んでいない
- [ ] コミットに `git commit -s`（DCO）の Signed-off-by を付けた
- [ ] C++ と Python のテストが通る（README.md の「ビルドと実行」のテストのコマンド）
