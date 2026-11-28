# SmartPack 3D

AI 물건 인식 기반 3D 가방 정리 서비스

물건 사진을 찍으면 크기를 재고, 가방 안에 테트리스처럼 빈틈 없이 배치해 3D로 보여줍니다.

## 주요 기능
- **사진으로 크기 측정:** ArUco 마커를 함께 찍어 물건의 가로·깊이(상면 사진), 높이(측면 사진)를 cm로 계산 (OpenCV)
- **AI 보조 판정 (선택):** API 키가 있으면 물건 종류·깨짐 주의 등을 참고용으로 표시. 치수 계산에는 쓰지 않음
- **물건 표 입력:** 크기(cm), 무게(kg), 깨짐/눌림 주의, 버팀 무게(kg), 방향 고정, 크기 잠금
- **3D 자동 배치:** 큰 물건부터, 6방향 회전, 겹침·범위 검사, 지지 검사(바닥 70% 이상 받침), 하중 계산
- **빈틈 줄이기:** 투영 극점(Projected Extreme Point) 후보 + 벽·바닥·다른 물건과 맞닿는 면적이 넓은 자리 우선
- **3D 시각화:** 3D·위·앞·옆 시점 버튼, 원근/정투영 전환

---

## 팀원용: 처음 받아서 실행하기 (Windows 기준)

### 1. 준비물 설치 (처음 한 번만)

**① Python**
1. https://www.python.org/downloads/ 에서 **Python install manager**를 받아 설치합니다.
   (마이크로소프트 스토어에서 "Python Install Manager"를 검색해 설치해도 됩니다.)
2. VS Code나 터미널을 껐다 켠 뒤 `py`를 입력하면 설정 질문이 나옵니다. **전부 `y`** 를 누릅니다.
3. 확인: 아래 명령을 입력해서 `Python 3.x.x`가 나오면 성공입니다.
   ```
   python --version
   ```

> VS Code의 "Python" 확장 프로그램만 설치해서는 실행되지 않습니다. 파이썬 본체를 꼭 설치하세요.

**② Git**
1. https://git-scm.com/download/win 에서 받아 기본 설정 그대로 설치합니다.
2. 처음 한 번 이름과 이메일을 등록합니다. (깃허브 계정 정보로 입력)
   ```
   git config --global user.name "깃허브아이디"
   git config --global user.email "깃허브이메일"
   ```

### 2. 코드 받기 (처음 한 번만)
VS Code에서 터미널을 열고(`` Ctrl + ` ``) 코드를 받을 폴더로 이동한 뒤 실행합니다.
```
git clone https://github.com/mokpoproject/smartpack3d
cd smartpack3d
```
VS Code 메뉴의 **파일 → 폴더 열기**로 `smartpack3d` 폴더를 열어 두면 편합니다.

### 3. 필요한 패키지 설치 (처음 한 번만)
```
python -m pip install -r requirements.txt
```

### 4. 실행
```
python -m streamlit run app.py
```
브라우저에서 http://localhost:8501 이 열리면 성공입니다. 끌 때는 터미널에서 `Ctrl + C`를 누릅니다.

> VS Code 오른쪽 위의 ▶ 실행 버튼으로 실행하면 화면이 뜨지 않고
> `missing ScriptRunContext` 경고만 반복됩니다. 반드시 위 명령어로 실행하세요.

---

## 최신 코드 받기 (매번 작업 전에)
```
git pull
```
다른 팀원이 올린 최신 코드가 내 폴더에 반영됩니다. 앱이 켜져 있으면 브라우저 오른쪽 위의 **Rerun**을 누르면 됩니다.

## 내가 고친 코드 올리기
> 올리려면 저장소 관리자가 깃허브 **Settings → Collaborators**에서 내 계정을 초대해야 합니다.
> 초대 메일을 수락한 뒤부터 올릴 수 있습니다.

```
git pull
git add .
git commit -m "무엇을 바꿨는지 짧게 적기"
git push
```
처음 `git push`를 하면 깃허브 로그인 창이 뜹니다. 로그인하면 다음부터는 자동입니다.

### 충돌을 줄이는 규칙
- **작업 시작 전에 항상 `git pull`** 을 먼저 합니다.
- 같은 파일을 여러 명이 동시에 고치지 않도록 단톡방에 "지금 ○○ 수정 중"이라고 알립니다.
- 커밋 메시지는 한 줄로 구체적으로 적습니다. 예: `무게 입력 칸 추가`, `3D 상자 테두리 수정`

## 자주 나오는 오류
| 오류 메시지 | 해결 |
|---|---|
| `'python'(또는 'pip', 'py') 용어가 ... 인식되지 않습니다` | 파이썬 본체가 설치되지 않았습니다. 위 1-①을 다시 하고 VS Code를 재시작하세요. |
| `python`을 쳐도 아무 반응이 없음 | 윈도우의 가짜 python 바로가기입니다. 설정 → 앱 → 고급 앱 설정 → 앱 실행 별칭에서 `python.exe`를 끄고 1-①을 하세요. |
| `No module named streamlit` / `cv2` / `openai` | 3번 패키지 설치(`python -m pip install -r requirements.txt`)를 먼저 하세요. |
| `missing ScriptRunContext` 반복 | ▶ 버튼 말고 `python -m streamlit run app.py`로 실행하세요. |
| `git push` 할 때 `Permission denied` / 403 | 저장소에 초대되지 않았습니다. 관리자에게 Collaborator 초대를 요청하세요. |
| `git pull` 할 때 충돌(conflict) | 고친 파일을 따로 복사해 두고 팀원과 상의한 뒤 합치세요. |

---

## 휴대폰에서 접속
PC에서 `python -m streamlit run app.py` 를 실행하면 표시되는 **Network URL**(예: `http://192.168.0.10:8501`)을
같은 Wi-Fi의 휴대폰 브라우저에서 열면 됩니다. 방화벽이 물으면 허용하세요.

## 설정 (선택)

AI 보조 판정(카테고리, 깨짐 주의, 권장 방향)을 쓰려면 `.env.example`을 `.env`로 복사하고 키를 넣으세요.

```
LLM_API_KEY=발급받은키
LLM_BASE_URL=https://factchat-cloud.mindlogic.ai/v1/gateway
LLM_MODEL=gpt-5.6-terra
```

- 키가 없으면 AI 판정만 건너뛰고 안내 문구를 보여 줍니다. 나머지 기능은 모두 동작합니다.
- 키는 코드에 넣지 마세요. `.env`는 `.gitignore`에 포함되어 있습니다.
- 모델은 **이미지 입력을 지원하는 모델**이어야 사진 판정이 됩니다.

## 사진으로 크기 추정하는 방법

크기를 아는 기준 물체로 **ArUco 마커**를 씁니다. 사진 한 장만으로는 cm를 알 수 없기 때문입니다.

1. 마커 생성 (인쇄 후 실제 한 변 길이를 재서 앱의 ‘마커 한 변의 실제 길이’에 입력)
   ```powershell
   python -c "import cv2; d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50); cv2.imwrite('marker_7.png', cv2.aruco.generateImageMarker(d, 7, 800))"
   ```
   인쇄 시 배율을 100%(실제 크기)로 설정하세요.
2. 물건과 마커를 **같은 평면(같은 바닥)** 에 놓습니다.
3. **① 상면 사진**: 위에서 수직으로 찍습니다 → 가로·깊이
4. **② 측면 사진(선택)**: 옆에서 찍고 마커를 똑바로 세웁니다 → 높이
5. 앱의 ‘2. 사진으로 크기 추정’ 탭에서 사진을 올리고 **AI 크기 추정**을 누릅니다.
6. 측정 검증 이미지(초록: 마커, 빨강: 물건 윤곽)를 확인하고 **이 행에 치수 적용**을 누릅니다.

> 참고: 정면 사진으로는 깊이를 알 수 없으므로 이 앱은 “바닥 평면을 위에서 찍은 사진”을 가로·깊이 사진으로 정의합니다.

### 촬영 팁
- 밝은 곳에서 찍고 그림자를 줄이세요.
- 물건과 배경의 색 차이를 크게 하세요(예: 흰 종이 위의 어두운 물건).
- 물건 전체가 사진 안에 들어와야 합니다(가장자리에 닿으면 경고).
- 배경이 복잡하면 ‘물건 영역 직접 지정’을 켜서 물건만 포함되게 하세요.
- 마커는 흐리지 않게, 화면에서 충분히 크게(권장 100px 이상) 찍으세요.

### 결과 해석
- **크기 출처 = 사진 추정(확인 필요)** 은 값을 한 번 눈으로 확인하라는 표시입니다.
- **크기 잠금**을 켠 행은 사진 추정으로 절대 바뀌지 않습니다.
- LLM 판정은 치수에 쓰지 않습니다. 카테고리/주의/방향 힌트만 참고용으로 보여 줍니다.

## 정확도 한계 (현재 버전)

| 항목 | 한계 |
|---|---|
| 물체 분할 | 밝기 차이(Otsu) 기반 간이 분할. 그림자·비슷한 색 배경에 약함. 객체 분할 모델은 후속 과제 |
| 높이 있는 물건 | 위에서 본 상면은 물건 윗면이 카메라에 가까워 실제보다 크게 나올 수 있음 |
| 평면 가정 | 물건과 마커가 같은 평면에 있어야 함. 떨어진 위치나 다른 높이면 오차 |
| 측면 높이 | 마커가 똑바로 세워져 있다고 가정. 옆으로 누우면 오차 |
| 불규칙한 물건 | 직육면체 외곽(최소 사각형)으로 근사. 둥근 물건은 여유 있게 측정됨 |
| 검증 범위 | 합성 이미지 테스트에서 약 ±0.2cm 수준이었으나 **실사진 검증은 아직 하지 않았습니다.** 실제 값은 촬영 조건에 따라 더 클 수 있으니 반드시 직접 확인 |

## 프로젝트 구조 (객체지향 역할 분리)

```
app.py          Streamlit UI (입력/표시만 담당, 로직은 서비스 호출)
packing.py      데이터 모델(Item, PlacedItem, Bin) + 배치 알고리즘(Packer)  - 순수 Python
vision.py       ArUco 측정: MarkerDetector, PlaneRectifier, ObjectSegmenter,
                ObjectMeasurer(파사드), SizeEstimator(결과 통합)           - OpenCV
llm_service.py  ItemAdvisor(추상) / NullAdvisor / OpenAICompatibleAdvisor  - LLM
plot3d.py       Plotly 3D 시각화
tests/          pytest 단위 테스트 (배치 알고리즘)
markers/        인쇄용 ArUco 마커 (DICT_4X4_50, ID 7, A4에 한 변 6cm)
old/            첫 프로토타입(smartpack3d_v1.py, 단일 파일 버전) 보관용
```

### 클래스 책임
- `MarkerDetector.find()` : 기준 마커 선택
- `PlaneRectifier` : 마커 4점으로 호모그래피를 구해 평면을 정면으로 펴기, 촬영 품질 점수
- `ObjectSegmenter.segment()` : 물체 윤곽 찾기. **객체 분할 모델로 교체할 때 이 클래스만 바꾸면 됩니다.**
- `ObjectMeasurer.measure_top / measure_side` : 사진 한 장 → `PhotoMeasurement`(치수, 신뢰도, 경고)
- `SizeEstimator.combine` : 상면+측면 결과 통합, 비현실 치수(0.5~150cm 밖) 제거
- `ItemAdvisor.advise` : LLM 보조 판정 인터페이스. 다른 모델/API로 바꿀 때 구현체만 추가

### 원칙
- LLM 치수는 사용하지 않습니다. 치수는 OpenCV 기하 측정값만 사용합니다.
- 모든 추정 값은 `needs_review=True`로 표시되고 사용자가 확인해야 합니다.
- 예외는 서비스 계층에서 결과 객체(`ok=False`, `error`)로 바꾸어 UI가 멈추지 않게 합니다.

## 개발자용: 테스트
```
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## 다음 개선 과제
1. 객체 분할 모델(SAM 등) 도입으로 물체 윤곽 정확도 향상
2. 마커 없이 사용자가 물체 4점(또는 기준 물체)을 클릭하는 보정 UI
3. 상면 사진의 높이 보정(물체 높이 입력 시 투시 보정)
4. 여러 사진(3면)으로 물체 실루엣 기반 3D 추정
5. 측정 결과 로그 저장 및 실측값과 비교하는 정확도 평가 스크립트
6. 배치 알고리즘 개선(전역 탐색, 회전 제약 세분화)
- 배치: 넣는 순서를 여러 가지로 시도해 빈틈 추가 감소
- 앱(프론트): 휴대폰용 화면 제작

## 문제 해결
- **마커를 찾지 못함**: 마커 전체가 보이는지, 흐리지 않은지, 사전(ArUco 사전)이 맞는지 확인
- **물체 윤곽을 찾지 못함**: ‘물건 영역 직접 지정’ 사용, 배경과 밝기 차이 키우기
- **모바일에서 사이드바가 안 보임**: 화면 왼쪽 위 `>>` 버튼을 누르세요
- **카메라 버튼이 안 보임**: 브라우저 카메라 권한을 허용하고, HTTPS 또는 `localhost` 접속인지 확인하세요
  (Wi-Fi IP 접속 시 일부 브라우저는 카메라를 막을 수 있습니다. 이때는 파일 업로드를 쓰세요.)
