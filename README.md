# Less is More in 3D
### 3D Reconstruction via Frame Selection and Token Merging

> **B3HIND** | 인공지능종합설계 기말발표  
> 김서진 · 박준혁 · 한지수 · 황징아이  
> 지도교수: 김선주 | 대학원생: 김민수 | 산학멘토: 박재웅

---

## Overview

**LiteVGGT** 기반의 효율적인 3D Reconstruction 파이프라인입니다.  
기존 VGGT의 메모리 병목(O((N·T)²))과 고정된 token 처리 방식의 한계를 극복하기 위해,  
**Frame Selection**과 **Adaptive Token Merging**을 계층적으로 통합하였습니다.

---

## Motivation

| 문제 | 설명 |
|------|------|
| 프레임 수준 중복 처리 부재 | 모든 입력 프레임이 Global Attention에 포함됨 |
| 고정 GA Token 비율 | 항상 상위 10% 고정 → 단순/복잡 장면 모두 비효율 |
| 하드코딩된 GA Map 가중치 | 경험적으로 설계된 고정 가중치로 최적성 미검증 |
| Frame & Token 분리 연구 | 두 기법을 독립적으로 다뤄 계층적 결합 효과 미탐구 |

---

## Pipeline

```
Multi-view Input
      ↓
[STEP 2] Frame Selection       ← Sobel-based edge-aware selection
      ↓
[STEP 3] Adaptive Token Merging         ← Depth-aware GA map + Quadtree-Bipartite
      ↓
[STEP 4] Global Attention      ← Adaptive cache reuse
      ↓
[STEP 5] 3D Reconstruction     ← Camera / Depth Map / Point Cloud
```

---

## Methods

### 1. Sobel-based Frame Selection
- Sobel gradient + local variance로 각 프레임의 **Info Score** 계산
- `Info Score = 0.3 × var + 0.7 × grad`
- Threshold 이하 프레임 제거 → OOM 해결 + local geometry 보존
- **결과**: Chamfer Distance 0.53 → **0.478** (vs Random Selection)

### 2. Quadtree-based Adaptive Token Merging
- 기존 고정 stride 2×2 → 장면 복잡도에 따라 동적으로 조절
- **STTM QT-Bipartite**: 공간·시간 방향 동시 redundancy 제거
- **결과**: 압축률 84.7%, 속도 **1.16×** 향상, CD 변화 -0.09%

### 3. External Depth Boundary Prior (GA Token Selection)
- Depth Anything V2-Small로 depth boundary map을 offline cache
- 기존 GA score에 약하게 추가 (`β=0.08`)
- **결과**: AUC@30 / AUC@15 모두 개선, seed 안정성 향상

### 4. Adaptive Cache Scheduling
- 매 레이어마다 dst 토큰의 cosine similarity 변화량(Drift) 측정
- `D_l = 1 - mean(cos_sim(dst_now, dst_cached))`
- Drift < τ 시 이전 merge indices 재사용 → 불필요한 재계산 제거

---

## Results

| Method | Input Frames | Kept Frames | Chamfer Distance |
|--------|-------------|-------------|-----------------|
| LiteVGGT | 300 | OOM | — |
| Baseline (Random) | 300 | 93 | 0.53 |
| **Ours (Sobel)** | 300 | 93 | **0.478** |

| Method | Avg Compression | CD Change | Speedup |
|--------|----------------|-----------|---------|
| Baseline (2×2) | 65.9% | +0.00 | 1.00× |
| Dynamic (4×4) | 86.7% | -1.06% | 1.17× |
| **QT-Bipartite** | **84.7%** | **-0.09%** | **1.16×** |

---

## Environment

- GPU: RTX 3090 (24GB) / RTX A6000 (48GB)
- Dataset: ScanNet, ScanNet++, DTU
- Baseline: [LiteVGGT](https://github.com/) (CVPR 2026)

---

## References

- Wang et al., *VGGT: Visual Geometry Grounded Transformer*, SIGGRAPH 2025
- Shu et al., *LiteVGGT*, CVPR 2026
- Bolya et al., *Token Merging: Your ViT But Faster (ToMe)*, ICLR 2023
- Luo et al., *Multi-Granular Spatio-Temporal Token Merging for Training-Free Acceleration of Video LLMs (STTM)*, 2024
