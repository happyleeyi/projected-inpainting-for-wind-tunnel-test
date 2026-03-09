# Projected Inpainting for Wind Tunnel Pressure Data

## Korean

### 연구 배경

풍동 실험에서는 센서 오작동, 케이블 문제, 측정 시스템 불안정, 데이터 수집 과정의 오류 등으로 인해 시계열의 일부 구간이 결측될 수 있다. 특히 건물 표면 풍압 데이터는 시간 축을 따라 연속적으로 측정되기 때문에, 일부 구간의 결측은 단순한 누락을 넘어서 전체 파형 해석과 통계 해석 모두에 영향을 준다.

문제는 이러한 결측이 발생했을 때 실험을 다시 수행하는 것이 매우 어렵다는 점이다. 풍동 실험은 한 번 수행하는 데 준비 시간이 길고, 실험 조건을 다시 맞추는 비용도 크며, 사용되는 센서 역시 고가이기 때문에 동일한 설정으로 재실험하는 것은 현실적으로 부담이 크다. 따라서 이미 확보한 실험 데이터를 최대한 활용하면서, 결측된 시계열 구간을 신뢰성 있게 재생성하는 방법이 필요하다.

기존의 diffusion 모델, GAN 계열 생성모델, 혹은 일반적인 시계열 생성기는 결측 구간과 형태가 비슷한 파형을 만들어낼 수는 있다. 그러나 이러한 방법은 종종 겉보기에는 비슷한 시계열을 생성하는 데 그치고, 실제로는 평균, 분산, 진폭 범위, 에너지와 같은 중요한 대표 통계량을 제대로 보존하지 못한다. 풍압 시계열 데이터에서는 단순한 모양의 유사성보다 대푯값의 보존이 훨씬 중요하다. 대푯값이 무너지면 복원된 시계열은 물리적 의미와 데이터 품질을 동시에 잃게 된다.

따라서 본 연구는 다음과 같은 문제의식에서 출발한다.

- 결측 구간은 생성적으로 복원할 수 있어야 한다.
- 복원 결과는 단순히 그럴듯한 파형이 아니라, 이미 정확히 측정된 구간의 통계적 특성과 강하게 일치해야 한다.
- 특히 시계열의 대푯값을 강하게 만족시키는 복원 방식이 풍압 데이터 품질 보존에 핵심적이다.

즉, 본 연구는 **사각 건물 풍압 풍동 실험에서 발생한 결측 구간을 projected inpainting으로 복원하되, 관측된 구간의 대푯값을 강하게 만족시키는 생성 모델을 설계하고 검증하는 것**을 목표로 한다.

### 방법

본 연구에서 제안하는 방법은 **RePaint 기반 확산 복원 과정에 PDM을 결합하여, 결측 구간 생성 시 known part의 대표 통계량을 강하게 만족시키는 projected inpainting 방법**이다. 방법의 구성은 다음 세 단계로 설명할 수 있다.

#### 1. RePaint 방법 소개

RePaint는 부분 관측 조건이 있는 inpainting 문제를 위해 설계된 diffusion 기반 복원 방법이다. 핵심 아이디어는 unknown part를 역확산 과정에서 한 step씩 denoising하면서 생성하되, 매 step마다 그 step에 해당하는 수준의 노이즈를 섞은 known part를 다시 결합하여 다음 denoising step의 입력으로 넣어주는 데 있다.

즉, 각 step에서는 현재까지 복원된 unknown part만 사용하는 것이 아니라, 동일한 노이즈 수준으로 맞춘 known part와 concat된 상태로 다음 step이 진행된다. 이 구조 덕분에 unknown part는 매 denoising step마다 known part와 어울리도록 생성되며, 최종 복원 결과 역시 관측된 구간과 더 자연스럽고 연속적으로 연결된다.

따라서 RePaint는 결측 구간 복원을 위한 기본 생성 프레임워크로 사용된다. 그러나 RePaint만으로는 복원된 구간이 관측 구간의 대표 통계량을 충분히 만족한다고 보장하기 어렵다.

#### 2. PDM 방법 소개

PDM은 denoising 과정에 projection 제약을 삽입하여, 생성 결과가 관측된 구간의 대표 통계량을 강하게 만족하도록 만드는 방법이다. 본 연구에서 중요한 대표 통계량은 평균, 분산과 같은 값들이다. 풍압 시계열에서는 이러한 값들이 단순한 부가 정보가 아니라 데이터 품질과 물리적 해석 가능성을 좌우하는 핵심 요약 정보가 된다.

PDM의 역할은 다음과 같다.

- known part에서 대표 통계량을 계산한다.
- 각 denoising step마다 생성 중인 unknown part에 projection을 적용한다.
- projection을 통해 생성 결과가 대표 통계량 constraint를 매 step에서 항상 만족하도록 만든다.

즉, PDM은 그럴듯한 생성에 머무르지 않고, 대표 통계량을 만족하는 생성으로 복원 문제를 바꾸는 장치라고 볼 수 있다.

#### 3. 제안하는 RePaint + PDM 결합 방법 소개

본 연구의 제안 방법은 RePaint와 PDM을 결합한 projected inpainting 방식이다. 관측된 부분은 known part로, 복원해야 하는 부분은 unknown part로 두고, diffusion 기반 RePaint를 이용해 unknown part를 복원한다. 이때 각 denoising step에서 known part의 대표 통계량을 만족하도록 projection을 수행한다.

이 결합 구조의 핵심은 다음과 같다.

1. RePaint가 관측된 구간의 정보를 유지한 채 unknown part를 step-by-step으로 생성한다.
2. PDM이 각 denoising step에서 생성 구간이 대표 통계량을 만족하도록 projection을 진행한다.
3. 그 결과 복원된 구간은 단순히 파형 형태만 비슷한 것이 아니라, 관측된 부분의 통계적 특성을 강하게 반영하게 된다.

결과적으로 제안 방법은 **관측된 구간과의 연속성은 RePaint로 확보하고, 대표 통계량의 보존은 PDM으로 강제하는 방식**이라고 정리할 수 있다. 본 연구는 이 결합이 풍압 풍동 실험 시계열의 결측 보완에서 기존 생성 기반 복원보다 더 높은 데이터 품질을 제공할 수 있는지를 검증한다.

### 실험

실험 데이터는 Tokyo Polytech University에서 수행한 사각 건물 풍동 실험 데이터이다. 이 데이터는 사각 건물 모형의 각 표면에 배치된 pressure tap에서 시간에 따라 측정한 풍압 시계열로 구성되며, 각 측정점이 어느 면에 위치하는지와 건물 표면상 어느 좌표에 놓이는지에 대한 위치 정보도 함께 포함한다. 즉, 단순한 1차원 시계열 모음이 아니라, **건물 표면 위치 정보와 결합된 풍압 시계열 데이터**라는 점이 중요하다.

전처리 코드는 `data_preprocess.py`에 구현되어 있으며, MATLAB `.mat` 파일에서 다음 정보를 읽는다.

- `Wind_pressure_coefficients`
- `Location_of_measured_points`
- `Sample_frequency`

여기서 각 측정점은 `face_id`, `left_dist`, `bottom_dist`, `point_id` 등의 메타데이터를 가지며, 현재 학습 코드에서는 그중 `face_id`, `left_dist`, `bottom_dist`를 조건 정보로 사용한다. `model_DM.py`의 조건부 diffusion 모델은 이 조건들을 임베딩하여 복원 과정에 반영한다.

데이터 전처리 과정은 다음과 같다.

1. 원본 풍압 시계열을 읽는다.
2. 각 시계열을 3초 길이의 non-overlapping window로 자른다.
3. 샘플 주파수는 1000 Hz로 가정하므로, 각 window 길이는 3000 time step이 된다.
4. 전체 window를 `face_id`, `left_dist`, `bottom_dist` 조합 기준으로 stratified split 한다.
5. 전체의 80%를 학습 데이터, 20%를 테스트 데이터로 사용한다.
6. 정규화는 학습 데이터의 평균과 표준편차로만 수행하고, 동일한 통계를 테스트 데이터에도 적용한다.

학습 단계에서는 diffusion 모델이 전체 window를 생성할 수 있도록 학습된다. 현재 `train_DM.py`는 조건부 1D DiT 기반 diffusion 모델을 학습하며, `train_baselines.py`는 같은 데이터 분할 위에서 ANN과 CNN 비교 모델을 학습한다.

테스트 단계에서는 각 테스트 window의 뒤 절반을 제거하여 인위적인 결측 상황을 만든다. 즉, 앞 절반은 known part, 뒤 절반은 missing part가 된다. 그 후 모델은 앞 절반과 조건 정보(`face_id`, `left_dist`, `bottom_dist`)를 바탕으로 제거된 뒤 절반을 복원한다.

현재 평가 코드는 `test.py`에 구현되어 있으며, 다음 방법들을 비교하도록 작성되어 있다.

- projected RePaint (`projection_mode="step"`)
- plain RePaint (`projection_mode="none"`)
- data-roundtrip projection RePaint
- ANN baseline
- CNN baseline
- 추가 실험으로 flow matching 기반 projected/plain inpainting

평가 지표 역시 코드에 반영되어 있다. 대표적으로 다음 값들이 계산된다.

- MSE
- FID
- 평균 값 차이
- 분산 값 차이

따라서 이 실험은 단순히 뒤 절반을 얼마나 비슷하게 맞췄는가만 보는 것이 아니라, **복원된 시계열이 관측 구간의 대표 통계량을 얼마나 잘 보존하는가**까지 함께 측정하는 구조다.

### 결과

### 결론

## English

### Background

In wind-tunnel experiments, parts of a time series may become missing because of sensor malfunction, wiring issues, instability in the measurement system, or data acquisition errors. For building-surface pressure data, such missing segments are especially problematic because the signals are interpreted not only as waveforms over time but also through their statistical characteristics.

Repeating the experiment is usually not a practical solution. Wind-tunnel testing is time-consuming, expensive to set up, and difficult to reproduce under exactly the same conditions. The sensors used in these experiments are also costly. As a result, once missing values occur, it is highly desirable to reconstruct the missing region from the already observed data rather than rerunning the entire experiment.

Conventional generative models such as diffusion models, GANs, or generic sequence generators can often produce waveforms that look superficially similar to the target region. However, in many cases they preserve only the rough shape of the sequence and fail to preserve important representative statistics such as the mean, variance, amplitude range, or signal energy. For wind-pressure time-series data, preserving these representative values is far more important than visual similarity alone. If those statistics are not preserved, the reconstructed sequence loses both physical interpretability and data quality.

This study is therefore motivated by the following ideas:

- missing regions should be reconstructed generatively
- the reconstructed result should strongly agree with the trustworthy observed region
- representative-value preservation is essential for preserving the quality of wind-pressure time-series data

The goal of this project is to **reconstruct missing segments in rectangular-building wind-pressure wind-tunnel data through projected inpainting, while strongly enforcing representative statistics from the observed region**.

### Method

The method proposed in this study is a **projected inpainting framework that combines RePaint with PDM so that the generated missing region strongly satisfies representative statistics from the known region**. The method can be described in three stages.

#### 1. RePaint

RePaint is a diffusion-based inpainting method designed for partial-observation settings. The key idea is to denoise and generate the unknown part step by step, while at every denoising step concatenating the known part after mixing it with the noise level corresponding to that step.

In other words, each denoising step does not use only the current estimate of the unknown region. It uses the reconstructed unknown region together with the known region adjusted to the same noise level for that step. Because of this design, the unknown part is generated so that it remains compatible with the known part at every denoising step, leading to a final reconstruction that is more naturally connected to the observed region.

For that reason, RePaint serves as the basic generative framework in this study. However, RePaint alone does not guarantee that the reconstructed region will preserve representative statistics from the observed region strongly enough.

#### 2. PDM

PDM introduces projection constraints into the denoising process so that the generated sequence strongly satisfies representative statistics from the observed region. In this study, important representative values include statistics such as the mean and variance. For wind-pressure time-series data, these are not secondary descriptors; they are critical summaries that determine both data quality and physical interpretability.

The role of PDM is:

- compute representative statistics from the known part
- apply projection to the generated unknown part at every denoising step
- enforce the representative-statistic constraints at every step of the denoising process

In this sense, PDM moves the problem beyond merely generating a plausible sequence and toward generating a sequence that explicitly satisfies representative-value constraints.

#### 3. Proposed RePaint + PDM Combination

The proposed method in this project combines RePaint and PDM into a single projected inpainting procedure. The observed part is treated as the known region, and the missing part is treated as the unknown region. RePaint is then used to reconstruct the unknown region, while projection is applied at each denoising step so that the representative statistics of the known part are satisfied.

The key logic of the combined method is:

1. RePaint generates the unknown region step by step while preserving information from the observed region.
2. PDM performs projection at every denoising step so that representative statistics are satisfied.
3. The reconstructed region therefore becomes not only waveform-consistent, but also statistically consistent with the observed part.

As a result, the proposed method can be summarized as follows: **RePaint enforces continuity with the observed region, while PDM enforces preservation of representative statistics**. This study investigates whether that combination can provide higher-quality completion for missing wind-pressure time-series data than conventional generative reconstruction.

### Experiment

The experiments use rectangular-building wind-tunnel data collected at Tokyo Polytech University. The dataset consists of wind-pressure time series measured over time from multiple pressure taps placed on the surfaces of a rectangular building model, together with metadata describing which surface each tap belongs to and where it is located on that surface. In other words, this is not just a collection of 1D sequences, but **wind-pressure time-series data coupled with spatial metadata on the building surface**.

The preprocessing pipeline is implemented in `data_preprocess.py`, which reads the following entries from MATLAB `.mat` files:

- `Wind_pressure_coefficients`
- `Location_of_measured_points`
- `Sample_frequency`

Each measured pressure tap is associated with metadata such as `face_id`, `left_dist`, `bottom_dist`, and `point_id`. In the current learning pipeline, `face_id`, `left_dist`, and `bottom_dist` are used as conditional inputs. The diffusion model in `model_DM.py` embeds these variables and uses them during reconstruction.

The preprocessing procedure is as follows:

1. load the raw pressure time series
2. split each series into non-overlapping 3-second windows
3. assume a sampling frequency of 1000 Hz, resulting in 3000 time steps per window
4. perform a stratified split using the tuple (`face_id`, `left_dist`, `bottom_dist`)
5. use 80% of the windows for training and 20% for testing
6. normalize all windows using statistics computed only from the training split

During training, the diffusion model is trained on full windows. In the current implementation, `train_DM.py` trains a conditional 1D DiT-based diffusion model, while `train_baselines.py` trains the ANN and CNN comparison models on the same split.

At test time, the back half of each test window is removed to create an artificial missing-data scenario. The front half is kept as the known region, and the removed back half is reconstructed from the observed front half and the available conditional metadata (`face_id`, `left_dist`, `bottom_dist`).

The current evaluation logic in `test.py` compares the following methods:

- projected RePaint (`projection_mode="step"`)
- plain RePaint (`projection_mode="none"`)
- data-roundtrip projection RePaint
- ANN baseline
- CNN baseline
- additional flow-matching-based projected/plain inpainting experiments

The evaluation metrics implemented in code include:

- MSE
- FID
- mean difference
- variance difference

Therefore, the experiment is designed not only to measure how well the missing back half is reconstructed, but also to measure **how well the reconstructed sequence preserves representative statistics from the observed region**.

### Results

### Conclusion
