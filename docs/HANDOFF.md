# セッション引き継ぎ — 2026-09-22

この資料を現在の状態の入口とする。`ASTRA_*` は各時点の設計・コード監査であり、実機での画質合格を意味しない。過去の「生成完了」は、ファイル書き出しの成功と映像品質の成功を分けて読むこと。

## 今回の区切り

ユーザーの「続けて」を受け、画質不合格だったLatent 2パスを修正し、512×512と標準960×544の実機再検証を完了した。通常生成・後処理・UIに加え、この2パス経路も初期値OFFの任意機能として利用できる。新しいモデル取得やドライバー更新をやり直す必要はない。

ユーザーは実装を Sol、計画・敵対的レビュー・検証を Astra に担当させるよう指定した。Sol がアプリ実装、Astra が主要な設計／コード監査、親セッションが実機・ブラウザ確認を担当した。最後の追加 Astra 起動はエージェント数上限で拒否されており、最終修正まで別 Astra が承認したとは扱わない。

最新の検証は **CPUテスト41件合格**、JavaScript構文確認・差分チェック合格。24ch正規化修正版を共有GPUキューで2本生成し、先頭・中間・末尾の目視・全frame decode・音声・FPS・metadataを確認した。追加Astraの起動はエージェント上限で拒否されたため、最終修正はSolのsource parity監査と親セッションの独立実機確認で評価した。

最新コードでサーバーを再起動し、ブラウザでLatentの初期値OFF、有効化時の固定重み選択・強さ0.18・仕上げ4ステップを確認した。確認後はOFFへ戻した。サーバーは起動中、生成モデルは再起動により未ロードのため「モデルを読み込む」から準備する。

## 実装・確認済み

| 範囲 | 現在の状態 |
| --- | --- |
| 独立エンジン | FastAPI + 公式 Diffusers H3。ComfyUI のサーバー、Python環境、ノードAPIを呼ばない |
| 通常生成 | T2VA 実機成功。Turbo8、Sage、INT8、124f、24 FPS、生成音声付き |
| メモリ | Transformer / Qwen を GPU 上で INT8 化してから自動 CPU オフロード。CPU先行ロードのRAM圧迫を解消 |
| モデル操作 | 明示ロードボタン、状態表示、複数LoRAの順序・強度・有効状態。変更は共有GPUキューへ自動反映 |
| 推奨生成設定 | 横960×544／1344×768、縦544×960／768×1344、正方形704×704／1024×1024。24 FPS、124／243／345f、秒数表示 |
| Latent 2パス | 固定3D Conv v1を用いる任意機能。初期値OFF、既定strength 0.18／refine 4 evaluations。512²と標準960×544の124f実機画質を確認 |
| 後処理 | 生成・アップスケール・フレーム補間を別ジョブとして実行。入力を上書きしない |
| Real-ESRGAN / RIFE | 実GPU試験済み。RIFEは2／3／4倍、整数位置の元フレームと音声・動画長を保持 |
| SeedVR2 | 固定3B FP16重みと独立CLI。実GPU、正確な有理数FPS、音声、metadataを確認 |
| DLSS SR | NVIDIA署名済みSDKを用いる2倍の書き出し。24回の実NGX評価を確認 |
| DLSS 5 NR | 実験機能。隔離した未署名コミュニティruntimeでfeature 18と24/24フレームを確認。元解像度1倍 |
| 保存 | `outputs/YYYY-MM-DD`。MP4 comment に生成設定JSONを埋め込み、同一JSONを隣へ保存 |
| 履歴 | SQLite、ジョブ取消、JPEGサムネイル、同時に1本だけの動画プレイヤー |

24 FPS は H3 の生成側の固定レート。48／72／96 FPS は独立したRIFE処理で作る。DLSS NRは超解像やフレーム生成として表示しない。DLSSによるフレーム補間の書き出しは未実装。

## Latent 2パスの失敗と修正

比較は同じ日本語プロンプト（木の卓上の赤い陶器マグ、朝の光、ゆっくり近づくカメラ、室内の時計音）、seed 42、Turbo8 LoRA強度1、Sage、124f、24 FPSで行った。

| ジョブID先頭 | 設定 | 実際に見た結果 |
| --- | --- | --- |
| `df767073` | 256²の8評価 → 学習済みlatent拡大 → 512²の4評価、strength 0.18 | **不合格**。ピンクの格子状の乱れ。MP4形式・音声・metadataだけは正常 |
| `12d37501` | 512²の通常単パス8評価 | 赤いマグカップと木の卓・室内を確認 |
| `3cd972f1` | 256²の通常単パス8評価 | 丸いマットに置かれた赤いマグカップを確認 |
| `8d623d35` | 正規化修正版、256²→512²、8+4評価、strength 0.18 | **合格**。frame 0/60/123を確認。赤いマグと編みマットを維持し格子なし |
| `e009d9e2` | 正規化修正版、標準480×288→960×544、8+4評価、strength 0.18 | **合格**。frame 0/60/123でマグ・卓・椅子・窓を維持し格子なし |

最初の2パス結果を「画質検証済み」と再解釈しないこと。`h3_two_pass_export_verification.json` はコンテナとmetadataの受領書であり、視覚的な合格証ではない。

Solが特定した原因は、学習済み3Dアップスケーラー専用の24ch平均・標準偏差による前後変換の欠落。公式upstreamノードは、Comfy/H3の潜在テンソルに対しても、モデルの直前に `(x - mean) / std`、直後に `out * std + mean` を適用する。DiffusersのVAE正規化とは別の契約。ネットワークの構造・重み形状だけを一致させても不十分だった。

修正後の2本はいずれも124f／24 FPS／5.1667秒、生成音声付きで、MP4埋め込みJSONとsidecarが一致した。標準横長出力SHA256は `8af488ea3e38a493b72ad0e2a5545fbb5946c2ae5e1dec9bd951e2003ae000ab`。固定upstreamとの追加照合では、24ch前後変換、network forward、37 latent framesの32+5 overlap分割と加重合成が一致した。公開UI/APIの停止を解除し、固定重みがreadyの場合だけ選択可能に戻した。初期値はOFFのままである。高解像度1344×768と345fの品質・メモリは別途検証が必要。

参照: [Latent 2パスの契約](LATENT_TWO_PASS.md)。固定重みは `models/latent_upscalers/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`、SHA256は `4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`。

## その他の制約・残る確認

- VC-Attentionは無効。既存の非公式カーネルでN=129を256へpaddingすると、期待値1.765625に対して50.5となる誤計算を実測。finiteだけでは検出できなかった。SageをVCと呼び替えない。
- 実機の通常生成は上記の小型T2VA／Turbo8で確認した範囲。Turbo8とstyle LoRA 2件（重み1.0／0.7／0.4、登録module数312／312／208）の登録順は確認したが、映像への個別効果は未評価。Turbo4、FL2VAの最初／最後画像、別モデル切替、高解像度・最大フレームの画質は追加確認が必要。
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
| `data/qa/latent_corrected_export_verification.json` | 正規化修正版512²、124f、音声・metadata・全frame decode |
| `data/qa/latent_landscape_export_verification.json` | 正規化修正版960×544、124f、音声・metadata・全frame decode |
| `data/qa/latent_corrected_visual_review.json` | Solと親セッションによる修正版2本の先頭・中間・末尾の目視結果 |
| `data/qa/latent_corrected_frame{0,60,123}.png` | 修正版512²の目視フレーム |
| `data/qa/latent_landscape_frame{0,60,123}.png` | 修正版標準横長の目視フレーム |
| `data/qa/multiple_lora_loaded.json` | Turbo8＋style LoRA 2件の順序・重み・登録module数 |
| `outputs/2026-09-22/h3-generate-12d37501.mp4` | 正常な512²単パス動画 |
| `outputs/2026-09-22/h3-generate-3cd972f1.mp4` | 正常な256²単パス動画 |
| `outputs/2026-09-22/h3-generate-8d623d35.mp4` | 正規化修正後の512² Latent 2パス動画 |
| `outputs/2026-09-22/h3-generate-e009d9e2.mp4` | 正規化修正後の標準960×544 Latent 2パス動画 |
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
- `h3studio/latent_upscale.py` / `latent_refine.py`: 固定重み、24ch正規化、低sigma再精製を行う検証済み2パス処理。
- `h3studio/postprocess.py`, `seedvr2.py`, `dlss.py`: 独立した後処理と実行可否の検証。
- `h3studio/video.py` / `metadata.py`: FPS、音声、書き出し、metadata。
- `static/index.html`, `app.js`, `styles.css`: UI。

```powershell
.\.venv\Scripts\python.exe -m pytest
node --check .\static\app.js
git diff --check
```

通常生成と後処理の実機確認を代替するためにテスト結果だけを使わない。GPU試験は共有キュー上で直列に実行する。引き継ぎ後のコミット位置は `git log -1`、未保存変更は `git status --short` で確認する。
