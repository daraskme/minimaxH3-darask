# DLSS 書き出し契約

H3 Studio は、NVIDIA公式SDKの **DLSS Super Resolution** と、コミュニティ実装の **DLSS 5 Neural Rendering（実験）** を別の方式として扱います。前者は2倍の超解像、後者は元解像度の映像変換です。どちらかの成功を、もう一方の成功として表示しません。

## 固定した実装

- 公式SR: `NVIDIA/DLSS` v310.9.1、commit `374959484e79a640feaba44c93ac8cfb0a03f5b5`。x64 `nvngx_dlss.dll` の SHA-256 は `3975567b8943c53acce397f2b72380092f84f162d00b0d2c7d08a1025c563983` です。
- Neural Rendering: `2600th/dlss5-video-player` の pre-release `dlss5-video-player-v0.24.0`、commit `ffb01f5d015ee1a0544a6a46452cfe4b334553a3`。完全ZIPの SHA-256 は `66947ca33b84459d13b3002cfea153f295c1d0543260713eced6b9766a136e78` です。
- Neural worker は protocol 6、worker SHA-256 `f3bc8e69697e9db11e03fcbe16358608fc59be482d0ca44223322a1cad3778ea` に固定しています。`nvngx_dlssnr.dll` は未署名のコミュニティ改変版で、SHA-256 は `6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927` です。

Neural Rendering の配布物は `.cache/runtimes` へ隔離し、リポジトリへ含めません。`scripts/stage_dlss5_runtime.py` は公開manifestのサイズとSHA-256、ZIPパスの封じ込めを検証して展開するだけで、ダウンロードしたコードを実行しません。

## 利用可能になる条件

公式SRは、ローカルビルドのmanifest、公式DLLのハッシュ、現在のドライバーでの実動画スモークが一致した場合だけ有効です。ネイティブ exporter は実際のNGX出力を取り出し、NGX評価回数、入力フレーム数、出力フレーム数、2倍の寸法を照合します。

Neural Rendering は次をすべて満たした場合だけ有効です。

1. `--allow-unsigned-runtime` を付けて明示的に検証する。
2. runtime-lock の全12ファイルとworkerが固定ハッシュに一致する。
3. 現在のGPU・ドライバーでpreflightが feature 18 の作成、評価、inline interception、upscaling off、later failureなしを報告する。
4. 短いCFR MP4の全フレームを処理し、terminal receiptの job id、長さ、フレーム数、native evaluations、verified neural framesと実際のMatroskaを照合する。

検証記録は隔離runtime内の `h3studio-validation.json` に保存します。ドライバー、worker、runtime-lockのどれかが変わると利用不可に戻ります。検証コマンドは次の通りです。

```powershell
.\.venv\Scripts\python.exe .\scripts\stage_dlss5_runtime.py
.\.venv\Scripts\python.exe .\scripts\validate_dlss5_nr_runtime.py `
  --allow-unsigned-runtime --source .\short-cfr-sdr.mp4
```

## 出力と停止

Neural worker は inherited handle のバイナリ受領書だけを信頼し、exit code 0やファイルの存在だけでは成功にしません。メッセージは上限付きで保持し、破損した受領書、進捗停止、期限超過、キャンセル時はworkerとその子プロセスだけを停止して回収します。workerの作業ディレクトリは隔離runtimeです。

Neural Rendering のMatroskaは、検証済みの全NRフレームを一度だけH.264へ符号化し、元の有理数FPSへ正確に戻します。音声は元MP4からストリームコピーします。現段階の対象は短いCFR SDR MP4です。HDR/VFRを暗黙に変換しません。

RTX PRO 6000 / driver 616.92 では、192×128、24 fps、24 framesの実検証で feature 18 の全24フレーム、native evaluation 83回、最高evaluation 60を受領し、出力24 frames / 1.0秒 / CFRを確認しました。最終MP4のAAC packet SHA-256も入力と一致しました。この記録は他のGPUやドライバーの適合を保証しないため、環境ごとに再検証します。
