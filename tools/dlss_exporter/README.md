# H3 Studio DLSS export helper

This directory contains the source for a standalone **NVIDIA DLSS Super
Resolution** file exporter. It uses the official, signed NVIDIA DLSS SR
runtime. It is not the separate experimental DLSS 5 Neural Rendering path.

The build is pinned to:

- `2600th/dlss5-video-player` commit
  `ffb01f5d015ee1a0544a6a46452cfe4b334553a3` (tag
  `dlss5-video-player-v0.24.0`)
- `NVIDIA/DLSS` commit
  `374959484e79a640feaba44c93ac8cfb0a03f5b5` (tag `v310.9.1`)

Run `scripts/build_dlss_exporter.ps1`. The script verifies both revisions,
builds locally, stages only the exporter and official `nvngx_dlss.dll`, and
writes a hash manifest. A separate GPU smoke validation is required before the
GUI reports the method as available.

The experimental Neural Rendering runtime is not fetched, built into, or
executed by this helper. The public NVIDIA SDK currently supplies SR/DLAA,
Ray Reconstruction and Frame Generation runtimes, but no public
`nvngx_dlssnr.dll` implementation. The audited community path relies on an
unsigned modified runtime and remains unavailable without explicit opt-in and
its own verified worker protocol.
