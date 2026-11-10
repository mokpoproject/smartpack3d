# SmartPack 3D

AI 물건 인식 기반 3D 가방 정리 서비스 — 프로토타입

물건의 크기·무게·주의사항을 입력하면 수납함(가방) 안에 테트리스처럼 자동으로 배치하고 3D로 보여줍니다.

## 주요 기능
- 물건 표 입력: 크기(cm), 무게(kg), 깨짐/눌림 주의, 버팀 무게(kg), 방향 고정, 크기 잠금
- 3D 빈 패킹 휴리스틱: 큰 물건부터, 6방향 회전, 극점 후보 위치, 겹침·범위 검사
- 지지 검사(바닥 면적 70% 이상 받침)와 하중 계산(버팀 무게 초과 방지)
- Plotly 3D 시각화: 3D·위·앞·옆 시점 버튼, 원근/정투영 전환

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
python -m streamlit run smartpack3d.py
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
| `No module named streamlit` | 3번 패키지 설치를 먼저 하세요. |
| `missing ScriptRunContext` 반복 | ▶ 버튼 말고 `python -m streamlit run smartpack3d.py`로 실행하세요. |
| `git push` 할 때 `Permission denied` / 403 | 저장소에 초대되지 않았습니다. 관리자에게 Collaborator 초대를 요청하세요. |
| `git pull` 할 때 충돌(conflict) | 고친 파일을 따로 복사해 두고 팀원과 상의한 뒤 합치세요. |

---

## 다음 단계
- 사진으로 물건 종류 인식(비전 AI)
- 물건별 기본 크기·무게 데이터와 기준 물체로 크기 보정
- 빈 공간 줄이기: 투영 후보점, 접촉면 점수, 여러 순서 시도
- 넣는 순서 단계별 안내
