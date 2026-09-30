# NVIDIA Warp prebuilt wheel notice

The CP2B backend targets the upstream `warp-lang==1.17.0` Linux x86-64 wheel.
The wheel is not bundled in this source package. NVIDIA Warp's project source is
published under Apache-2.0, while the upstream distribution states that the
prebuilt wheel statically links NVIDIA libmathdx and carries additional license
terms in `licenses/libmathdx-LICENSE.txt`.

Before redistributing a prebuilt Warp binary with a product, review and retain
the exact notices shipped by the selected wheel. This project does not copy or
use the non-commercial GarmentCode Warp fork.

Pinned upstream source:

- Repository: `NVIDIA/warp`
- Tag: `v1.17.0`
- Commit: `f4c57f26f1e3936a89afd283e39fcabf6d548dc7`
- Wheel SHA-256: `47cd93636828dd16e55eeb4e77a0e3547b5fb0f383000a3d228629df68f28fd8`
