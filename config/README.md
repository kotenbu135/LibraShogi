# 稼働中のランの設定

`ls.toml`・`lx.toml` が **`~/libra-run/<run>/config.toml` の正**（docs/runbook.md §設定の管理、2026-09-18 のユーザーの依頼「config.toml を毎回私が編集するのは面倒。あなたが編集・管理できるようにして」）。

- 直すのは Claude。ユーザーの操作は管理コンソールの**停止 → 起動**だけで、起動のたびにランナーが
  `origin/main` の中身から `~/libra-run/<run>/config.toml` を作り直す（手元のチェックアウトが古くても効く）。
- 反映待ちの違いは `bin/libra [--run lx] config` で見える。前の設定は `config.toml.bak-<時刻>` に残る。
- **その PC だけの上書き**（置き場所など）は `~/libra-run/<run>/config.local.toml` に置く。ここには入れない。
- 公開リポジトリなので、絶対パスは `~` で書く（読むときに展開される）。秘密情報は置かない。
- 学習データに関わる値（探索・目標・リプレイ・学習率）を変えるときは CLAUDE.md 11 に従う
  （1 日 1 つ、前後で物差しを比べる、出所を docs/ls2-settings.md に書く）。

まだこのファイルが無いランは、そのランの `config.toml` をそのまま写して作る（中身を推測して書かない）。
`bin/libra [--run lx] config --adopt`、または `progress` ブランチの `progress/<run-id>-config.toml` から。
