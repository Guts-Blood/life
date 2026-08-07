# Day 12 A/B Config Diff

Status: `pass` — the only behavioral difference is the frozen Day 09 mixture.

| Path | A | B | Classification |
|---|---|---|---|
| `data.manifest_file_sha256` | `cb0d4819d383e4a22fd2c389c27c9cf058a20287c08e0fa5d924703f0af77492` | `d87e32b7925080c4e1abc6676354cb35e60af3f2311b969d6441a8aacc319837` | authorized mixture identity |
| `data.manifest_hash` | `19ea1a93e860fc5e93f463f7a136ae834ebe041c00f9fe3fa874f66ab1428c7a` | `8409de2d47c51cf0a29a22f5af0fd7bd136b98cfb1ed7de72682ffebb3bd5aaf` | authorized mixture identity |
| `data.manifest_path` | `learning/post-training-30-day-bootcamp/artifacts/data/day09-mix-A-balanced.json` | `learning/post-training-30-day-bootcamp/artifacts/data/day09-mix-B-targeted.json` | authorized mixture identity |
| `data.mix_name` | `mix_A_balanced` | `mix_B_targeted` | authorized mixture identity |
| `run.id` | `A` | `B` | run identity |

## Common training contract

- Base config SHA-256: `8675041b402ecbf54045a2afbbed4757b026ddf3c9a36be6bc16b81ad8f6690d`
- A resolved config hash: `f543163e8bb63fdfe4f42e495d658a171273414de06e7f70721d455a732e58d8`
- B resolved config hash: `6a38bb91de5ed44cb0d71e115594aeb73427f9af3bbe088fdb33ed7a5b46a793`
- Checkpoints use exact cumulative supervised-token budgets: `61,734 / 148,162 / 246,936`.
- A schedule hash: `86d57ca123e3dcb4144a535895c0cdab8d27fdda9976979b93520e0754c0d2fa`
- B schedule hash: `68e3b701e1743abb31f856456bc87baaf4bb0375c2d25996ed6af58738aaad05`
- The exact subset schedule preserves each run's preregistered skill ratios at every boundary up to deterministic integer rounding.
