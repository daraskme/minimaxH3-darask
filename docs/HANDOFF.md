# セッション引き継ぎ — 2026-09-22

この資料を現在の状態の入口とする。`ASTRA_*` は各時点の設計・コード監査であり、実機での画質合格を意味しない。過去の「生成完了」は、ファイル書き出しの成功と映像品質の成功を分けて読むこと。

## 今回の区切り

ユーザーの最新指示は「完成したところまでコミットプッシュして。あとセッションを引き継げる用に資料を整理」。確認済みの通常生成・後処理・UIを保存する。Latent 2パスは画質不合格を検出したため利用を止め、修正後の実機再検証を次の作業に残す。新しいモデル取得やドライバー更新をやり直す必要はない。

ユーザーは実装を Sol、計画・敵対的レビュー・検証を Astra に担当させるよう指定した。Sol がアプリ実装、Astra が主要な設計／コード監査、親セッションが実機・ブラウザ確認を担当した。最後の追加 Astra 起動はエージェント数上限で拒否されており、最終修正まで別 Astra が承認したとは扱わない。

区切り時点の検証は **CPUテスト40件合格**、Python／JavaScript構文確認合格。サーバーを最終コードで再起動し、Latent有効リクエストがHTTP 422で拒否されることを実APIでも確認した。サーバーは起動中だが、再起動後のH3モデルは未ロードなので、次の生成前にUIのロードボタンを使う。

## 実装・確認済み

| 範囲 | 現在の状態 |
| --- | --- |
| 独立エンジン | FastAPI + 公式 Diffusers H3。ComfyUI のサーバー、Python環境、ノードAPIを呼ばない |
| 通常生成 | T2VA 実機成功。Turbo8、Sage、INT8、124f、24 FPS、生成音声付き |
| メモリ | Transformer / Qwen を GPU 上で INT8 化してから自動 CPU オフロード。CPU先行ロードのRAM圧迫を解消 |
| モデル操作 | 明示ロードボタン、状態表示、複数LoRAの順序・強度・有効状態。変更は共有GPUキューへ自動反映 |
| 推奨生成設定 | 横960×544／1344×768、縦544×960／768×1344、正方形704×704／1024×1024。24 FPS、124／243／345f、秒数表示 |
| 後処理 | 生成・アップスケール・フレーム補間を別ジョブとして実行。入力を上書きしない |
| Real-ESRGAN / RIFE | 実GPU試験済み。RIFEは2／3／4倍、整数位置の元フレームと音声・動画長を保持 |
| SeedVR2 | 固定3B FP16重みと独立CLI。実GPU、正確な有理数FPS、音声、metadataを確認 |
| DLSS SR | NVIDIA署名済みSDKを用いる2倍の書き出し。24回の実NGX評価を確認 |
| DLSS 5 NR | 実験機能。隔離した未署名コミュニティruntimeでfeature 18と24/24フレームを確認。元解像度1倍 |
| 保存 | `outputs/YYYY-MM-DD`。MP4 comment に生成設定JSONを埋め込み、同一JSONを隣へ保存 |
| 履歴 | SQLite、ジョブ取消、JPEGサムネイル、同時に1本だけの動画プレイヤー |

24 FPS は H3 の生成側の固定レート。48／72／96 FPS は独立したRIFE処理で作る。DLSS NRは超解像やフレーム生成として表示しない。DLSSによるフレーム補間の書き出しは未実装。

## 最優先の未完了事項: Latent 2パスの画質

比較は同じ日本語プロンプト（木の卓上の赤い陶器マグ、朝の光、ゆっくり近づくカメラ、室内の時計音）、seed 42、Turbo8 LoRA強度1、Sage、124f、24 FPSで行った。

| ジョブID先頭 | 設定 | 実際に見た結果 |
| --- | --- | --- |
| `df767073` | 256²の8評価 → 学習済みlatent拡大 → 512²の4評価、strength 0.18 | **不合格**。ピンクの格子状の乱れ。MP4形式・音声・metadataだけは正常 |
| `12d37501` | 512²の通常単パス8評価 | 赤いマグカップと木の卓・室内を確認 |
| `3cd972f1` | 256²の通常単パス8評価 | 丸いマットに置かれた赤いマグカップを確認 |

最初の2パス結果を「画質検証済み」と再解釈しないこと。`h3_two_pass_export_verification.json` はコンテナとmetadataの受領書であり、視覚的な合格証ではない。

Solが特定した原因は、学習済み3Dアップスケーラー専用の24ch平均・標準偏差による前後変換の欠落。公式upstreamノードは、Comfy/H3の潜在テンソルに対しても、モデルの直前に `(x - mean) / std`、直後に `out * std + mean` を適用する。DiffusersのVAE正規化とは別の契約。ネットワークの構造・重み形状だけを一致させても不十分だった。

修正コードと契約テストを保存するが、修正版の完全な2パス実機生成は未確認。公開UI/APIでは利用停止を維持する。次セッションは次の順で進める。

1. `h3studio/latent_upscale.py` と固定upstreamの前後変換、forward、時系列チャンク合成を Astra が再確認する。
2. UI/APIの通常利用停止を維持した状態で、専用のローカル検証から同じ256→512・124fの生成を実施する。検証目的の変更と一般利用の解除を混同しない。
3. 先頭・中間・末尾フレームを実際に見て、元のカップ・構図、格子・ゴースト、時間的一貫性を確認する。shape、finite、終了コードだけで合格にしない。
4. 音声の保持、124f / 24 FPS、埋め込みJSON一致、取消後のブロックグラフ復旧も確認する。
5. 合格後にUI/APIの停止を解除し、資料とテストを更新する。高解像度や345fの安定動作は別途検証が必要。

参照: [Latent 2パスの契約](LATENT_TWO_PASS.md)。固定重みは `models/latent_upscalers/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`、SHA256は `4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`。

## その他の制約・残る確認

- VC-Attentionは無効。既存の非公式カーネルでN=129を256へpaddingすると、期待値1.765625に対して50.5となる誤計算を実測。finiteだけでは検出できなかった。SageをVCと呼び替えない。
- 実機の通常生成は上記の小型T2VA／Turbo8で確認した範囲。Turbo4、複数スタイルLoRAの組み合わせ、FL2VAの最初／最後画像、別モデル切替、高解像度・最大フレームの画質は追加確認が必要。
- 内蔵ブラウザは動画プレビュー時に一度クラッシュし、新しい同URLタブで復旧した。FFmpegでMP4全フレームをdecodeできた。再発する場合はブラウザと動画再生経路を切り分ける。GPUドライバー更新後のOS再起動は実施していない。
- コンポーネントのロード中・LoRA変更中・生成中の競合はキューのテストで確認した。新たな変更がなければ同じ試験を無制限に繰り返さない。

## このマシンの実行環境

- Windows、128 GB RAM、RTX PRO 6000 Blackwell Workstation 96 GB。
- NVIDIAドライバー **616.92** が有効。署名済み公式display INFを導入し、CUDAとSageのGPU実行を確認した。ユーザーは更新を明示許可済み。再起動、DDU、ACL変更、セキュリティ無効化は実施していない。
- Python 3.12.10、`.venv`、PyTorch 2.14.0+cu130、Diffusers 0.40.0。依存はrequirementsとbootstrap参照。
- 起動URLは `http://127.0.0.1:7862/`。`start.cmd` または `launch.ps1` を使う。まず `/api/status` とポート所有プロセスを確認し、二重起動しない。
- モデルを再ロードする場合、今回の実測は約2分、最大host RSS **73.37 GB**、最小空きRAM **39.45 GB**、GPU最大 **68,935 MiB**。Windowsのprivate commitを物理RAM使用量と混同しない。
- Codexの制限付きシェルでは `.venv` が誤ってbase Python不在を報告することがあった。同じPythonが許可された通常実行では動く。環境を削除・再作成する前にこの実行制約を確認する。
- ログは `logs/ui.stdout.log` / `logs/ui.stderr.log`。PIDは再利用されるため、古いPIDだけを根拠にプロセスを停止しない。既存のKrea2や他アプリは終了しない。

## モデル・ローカルデータ

公式H3は `models/MiniMax-H3` に61ファイル／144,051,143,011 bytesが揃い、公式SHA256またはgit blob SHA1との一致を確認済み。固定revisionは `42ed227ee7df40d41602854ae760620d6eb651fe`。

`C:/ComfyUI/models` の追加モデル設定は `D:/comfyui-models` を参照していた。そこから公式native 27ファイル約77.3 GBと追加120重み約242.17 GB、付属12ファイルを回収し、原本を保持した。92個のLoRAは `models/loras`、その他は `models/recovered_comfyui`。回収済み量子化・pruned・embedding・VDNファイルすべてをnative H3で読み込めるわけではない。

モデル、動画、アップロード、SQLite、ログ、検証用生成物、取得済みDLL/EXEは `.gitignore` 対象で、GitHubへは送らない。同じマシンでは引き続きローカルに存在する。他マシンではモデル取得、後処理モデルのself-check、DLSSビルド／検証が必要。

SeedVR2は同梱ソース全体のバイトをハッシュ検証する。`.gitattributes` でそのディレクトリの改行変換を止め、`.gitignore` の `/data/` はルートに限定している。これを単なる `data/` に戻すと、必要な `src/data/image/transforms` までGitから落ちるため注意する。

主なローカル証跡（リポジトリ相対パス）:

| 証跡 | 内容 |
| --- | --- |
| `data/qa/h3_download_verification.json` | 公式61ファイルの確認 |
| `data/qa/h3_gpu_load_profile.json` | フルモデル＋Turbo8のロード／メモリ実測 |
| `data/qa/h3_initial_visual_comparison.json` | 2パス不合格と単パス2種類の目視結果 |
| `data/qa/h3_two_pass_frame60.png` | 修正前の格子状の失敗 |
| `data/qa/h3_single_pass_frame60.png` | 512²単パスの正常なマグ |
| `data/qa/h3_single_256_frame60.png` | 256²単パスの正常なマグ |
| `outputs/2026-09-22/h3-generate-12d37501.mp4` | 正常な512²単パス動画 |
| `outputs/2026-09-22/h3-generate-3cd972f1.mp4` | 正常な256²単パス動画 |
| `outputs/2026-09-22/h3-upscale-9d532da5.mp4` | SeedVR2、24f・24000/1001 FPS・音声付き統合試験 |
| `data/qa/dlss-integrated.mp4` | NVIDIA DLSS SR 2倍の実出力 |
| `data/qa/dlss5-nr-integrated.mp4` | DLSS 5 NR 1倍の実出力 |
| `data/qa/existing_vc_padding_probe.json` | VCカーネルのpadding反例 |
| `models/recovery_summary.json` | モデル回収の集計 |
| `models/recovered_comfyui/recovery_manifest.json` | 回収ファイルと出典 |

## コードの入口と検証コマンド

- `h3studio/main.py`: API、ローカルアクセス制約、inventory、ジョブ受付。
- `h3studio/runner.py`: GPUキュー、モデルロード／LoRA更新の直列化、取消。
- `h3studio/engine.py`: 独立H3、INT8ロード、LoRA、生成、Latent分岐。
- `h3studio/generation_options.py`: 推奨解像度、24 FPS、17k+5とフレーム範囲。
- `h3studio/latent_upscale.py` / `latent_refine.py`: 再検証が必要な2パス処理。
- `h3studio/postprocess.py`, `seedvr2.py`, `dlss.py`: 独立した後処理と実行可否の検証。
- `h3studio/video.py` / `metadata.py`: FPS、音声、書き出し、metadata。
- `static/index.html`, `app.js`, `styles.css`: UI。

```powershell
.\.venv\Scripts\python.exe -m pytest
node --check .\static\app.js
git diff --check
```

通常生成と後処理の実機確認を代替するためにテスト結果だけを使わない。GPU試験は共有キュー上で直列に実行する。引き継ぎ後のコミット位置は `git log -1`、未保存変更は `git status --short` で確認する。
