# Projected Inpainting for Wind Tunnel Pressure Data

## Korean

### 개요

이 저장소는 **사각 건물 풍압 풍동 실험 데이터에서 결측된 시계열 구간을 `projected inpainting`으로 보완하는 연구**를 위한 코드입니다.

연구의 핵심 목적은, 이미 **정확하게 측정된 구간의 정보를 최대한 신뢰하면서**, 결측된 나머지 구간을 생성적으로 복원하는 것입니다. 특히 풍압 시계열처럼 물리적 의미가 강한 데이터에서는 단순히 파형을 그럴듯하게 생성하는 것만으로는 충분하지 않습니다. 제대로 관측된 구간이 갖는 **평균, 분산, 최대값 같은 대푯값과 통계적 특성**을 강하게 만족하도록 결측 구간을 생성해야 전체 데이터 품질을 보존할 수 있습니다.

이 프로젝트는 바로 그 관점에서, **부분적으로 관측된 풍압 시계열을 복원할 때 대푯값 보존 제약을 어떻게 강하게 걸어줄 것인가**를 다룹니다.

### 연구 문제

이 연구가 다루는 문제는 다음과 같습니다.

- 사각 건물 표면의 풍압 탭 시계열에서 일부 구간이 결측되었을 때, 이를 생성모델로 보완할 수 있는가
- 단순 복원보다, 이미 측정된 구간의 대표 통계량을 강하게 만족시키는 방식이 더 높은 데이터 품질을 보장하는가
- `projected inpainting`이 일반적인 생성 방식보다 풍압 시계열 복원에 더 적합한가

즉, 이 저장소는 단순한 시계열 예측이 아니라,

> **정상적으로 측정된 부분의 대푯값을 강하게 보존하면서 결측 구간을 생성하는 조건부 시계열 복원 연구**

를 목표로 합니다.

### 연구 배경

풍동 실험에서 얻은 풍압 데이터는 구조물의 공기역학적 특성을 반영하는 중요한 기초 데이터입니다. 하지만 실제 실험 데이터에는 센서 이상, 측정 누락, 품질 저하 등으로 인해 일부 구간이 비거나 신뢰하기 어려운 경우가 생길 수 있습니다.

이때 결측치를 보완하는 가장 중요한 기준 중 하나는, **이미 신뢰 가능한 구간의 정보가 복원 결과에 정확히 반영되는가**입니다. 시계열 생성 모델이 관측된 부분과 모순되는 값을 만들어내면 전체 데이터의 물리적 해석 가능성과 품질이 크게 떨어집니다.

따라서 이 연구는 다음 가정을 둡니다.

- 제대로 측정된 구간은 강한 조건으로 유지되어야 한다
- 생성된 결측 구간은 관측 구간과 통계적으로 일관되어야 한다
- 대푯값 보존은 시계열 데이터 품질 유지에 핵심적이다

### 방법 개요

현재 코드베이스는 풍압 탭 데이터를 3초 윈도우로 나누고, 앞부분을 관측 구간으로 사용해 뒷부분을 복원하는 방식으로 구성되어 있습니다.

주요 구성은 다음과 같습니다.

- `data_preprocess.py`: MATLAB 풍압 데이터 로드, 윈도우 분할, train/test 분리, 정규화
- `model_DM.py`: 조건부 diffusion 기반 복원 모델
- `model_FM.py`: 조건부 flow matching 기반 복원 모델
- `baseline_models.py`: ANN/CNN 기반 비교 모델
- `test.py`: projected inpainting 및 통계 제약 기반 평가

### 이 연구에서 하려는 것

이 연구에서 궁극적으로 하려는 것은 다음입니다.

1. 사각 건물 풍압 풍동 실험의 시계열 데이터에서 결측 구간을 보완한다.
2. 단순한 생성이 아니라, 제대로 측정된 부분의 대푯값을 강하게 만족하도록 결측 구간을 생성한다.
3. 이를 통해 복원 결과의 물리적 일관성과 데이터 품질을 동시에 확보한다.
4. diffusion, flow matching, baseline을 비교해 어떤 방식이 projected inpainting에 가장 적합한지 평가한다.

### 실행 파일

이 프로젝트는 `params.py` 기반의 parser-free 구조입니다.

```bash
python train_DM.py
python train_FM.py
python train_baselines.py
python test.py
```

## English

### Overview

This repository studies **missing-value completion for rectangular-building wind-pressure wind-tunnel data using projected inpainting**.

The main goal is to reconstruct missing segments of pressure time series while **strongly respecting the information contained in the correctly measured segments**. For physical time-series data such as wind-pressure signals, generating a visually plausible waveform is not enough. The generated missing region should satisfy the **representative statistics and summary values** of the reliably observed region, such as mean, variance, and extrema, as strongly as possible in order to preserve overall data quality.

From that perspective, this project focuses on **how to generate missing time-series segments under strong representative-value preservation constraints**.

### Research Question

The repository is built around the following questions:

- Can missing segments in surface pressure-tap time series from rectangular-building wind-tunnel experiments be completed with generative models?
- Does strongly enforcing representative statistics from the observed segment lead to better data quality than unconstrained reconstruction?
- Is `projected inpainting` more suitable for wind-pressure sequence completion than ordinary generative reconstruction?

In short, this is not just a generic forecasting project. It is a study on:

> **conditional sequence reconstruction that generates missing regions while strongly preserving representative values from the correctly measured region**

### Motivation

Wind-pressure data obtained from wind-tunnel experiments is important for understanding the aerodynamic behavior of structures. However, experimental time-series data may contain missing or unreliable segments due to sensor failure, partial corruption, or measurement-quality issues.

In that setting, one of the most important criteria for imputation is whether the reconstructed result remains consistent with the trustworthy observed segment. If a generative model produces values that contradict the correctly measured part, the physical interpretability and practical quality of the completed data deteriorate.

This project therefore assumes:

- correctly measured segments must be treated as strong conditions
- generated missing segments should remain statistically consistent with observed segments
- representative-value preservation is critical for maintaining time-series data quality

### Method Summary

The current codebase splits pressure-tap signals into 3-second windows and reconstructs the back half from the observed front half plus conditional metadata.

Key components:

- `data_preprocess.py`: MATLAB data loading, windowing, train/test split, normalization
- `model_DM.py`: conditional diffusion reconstruction model
- `model_FM.py`: conditional flow matching reconstruction model
- `baseline_models.py`: ANN/CNN comparison models
- `test.py`: projected inpainting and statistic-constrained evaluation

### What This Research Aims To Do

This research aims to:

1. complete missing segments in rectangular-building wind-pressure wind-tunnel time-series data
2. generate missing regions not just plausibly, but in a way that strongly satisfies representative values from correctly measured segments
3. preserve both physical consistency and overall data quality in the reconstructed results
4. compare diffusion, flow matching, and baseline models to determine which approach is most suitable for projected inpainting

### Run

This project is parser-free and driven by `params.py`.

```bash
python train_DM.py
python train_FM.py
python train_baselines.py
python test.py
```
