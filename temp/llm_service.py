"""
llm_service.py - LLM(비전 AI) 보조 판정 인터페이스

역할 제한 (중요)
----------------
- LLM은 물건명/사진을 보고 '카테고리', '깨짐 주의 여부', '권장 방향 힌트'만 판정합니다.
- LLM이 말한 치수는 사용하지 않습니다. 치수는 vision.py의 기하 측정값만 씁니다.
- 결과에는 항상 신뢰도와 '사용자 확인 필요' 상태가 포함됩니다.

설정 (.env 또는 환경 변수)
-------------------------
- LLM_API_KEY   : API 키. 없으면 BAZE_API_KEY 값을 사용합니다. 둘 다 없으면 LLM 판정을 건너뜁니다.
- LLM_BASE_URL  : OpenAI 호환 엔드포인트 (기본값: BAZE 게이트웨이)
- LLM_MODEL     : 이미지 입력을 지원하는 모델 이름
키는 코드에 하드코딩하지 않습니다.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

DEFAULT_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"
DEFAULT_MODEL = "gpt-5.6-terra"

SYSTEM_PROMPT = (
    "당신은 수납 정리 도우미입니다. 물건 이름(과 사진)을 보고 아래 JSON 형식으로만 답하세요.\n"
    "{\"category\": 문자열, \"fragile\": true 또는 false, "
    "\"orientation_hint\": \"자유\" | \"세우기 고정\" | \"완전 고정\", "
    "\"confidence\": 0~1 사이 숫자, \"notes\": 짧은 한국어 설명}\n"
    "치수(cm, kg)는 절대 추정하지 마세요. JSON 외의 다른 글자는 출력하지 마세요."
)


@dataclass(frozen=True)
class ItemAdvice:
    """LLM 보조 판정 결과. available=False면 판정을 수행하지 않았다는 뜻입니다."""

    available: bool
    category: Optional[str] = None
    fragile: Optional[bool] = None
    orientation_hint: Optional[str] = None
    confidence: float = 0.0
    notes: str = ""
    needs_review: bool = True  # AI 판정은 항상 사용자 확인 대상
    error: Optional[str] = None


class ItemAdvisor(ABC):
    """물건 보조 판정기의 공통 인터페이스. 구현체를 바꿔도 UI 코드는 그대로 둘 수 있습니다."""

    @abstractmethod
    def advise(self, item_name: str, image_jpeg: Optional[bytes] = None) -> ItemAdvice:
        """물건 이름과 (선택) 사진으로 카테고리/주의/방향 힌트를 판정합니다. 실패해도 예외를 던지지 않습니다."""


class NullAdvisor(ItemAdvisor):
    """API 키가 없을 때 쓰는 대체 구현. 아무 판정도 하지 않고 안내 메시지만 돌려줍니다."""

    REASON = "LLM API 키가 없어 AI 판정을 건너뛰었습니다. (.env에 LLM_API_KEY를 설정하면 사용할 수 있습니다.)"

    def advise(self, item_name: str, image_jpeg: Optional[bytes] = None) -> ItemAdvice:
        return ItemAdvice(available=False, notes=self.REASON)


class OpenAICompatibleAdvisor(ItemAdvisor):
    """OpenAI 호환 Chat Completions API를 호출하는 구현체."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL, timeout: float = 30.0) -> None:
        # openai 패키지는 실제로 쓸 때만 불러옵니다(키가 없는 환경에서 import 오류를 피하기 위해).
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._model = model

    def advise(self, item_name: str, image_jpeg: Optional[bytes] = None) -> ItemAdvice:
        import base64

        content: list[dict[str, Any]] = [{"type": "text", "text": f"물건 이름: {item_name}"}]
        if image_jpeg:
            b64 = base64.b64encode(image_jpeg).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:  # 네트워크/인증/모델 오류 모두 앱을 멈추지 않도록 흡수합니다.
            return ItemAdvice(available=False, error=f"LLM 호출 실패: {e}",
                              notes="AI 판정 없이 직접 입력하거나 나중에 다시 시도하세요.")
        return parse_advice(text)


def parse_advice(text: str) -> ItemAdvice:
    """LLM 응답 문자열에서 JSON을 찾아 ItemAdvice로 바꿉니다. 치수 필드는 무시합니다."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return ItemAdvice(available=False, error="LLM 응답에서 JSON을 찾지 못했습니다.", notes=text[:200])
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        return ItemAdvice(available=False, error=f"LLM 응답 JSON 파싱 실패: {e}", notes=text[:200])

    try:
        confidence = float(min(1.0, max(0.0, float(data.get("confidence", 0.0)))))
    except (TypeError, ValueError):
        confidence = 0.0
    return ItemAdvice(
        available=True,
        category=str(data["category"]) if data.get("category") else None,
        fragile=bool(data["fragile"]) if isinstance(data.get("fragile"), bool) else None,
        orientation_hint=data.get("orientation_hint"),
        confidence=confidence,
        notes=str(data.get("notes", "")),
    )


def create_advisor(env: Optional[Mapping[str, str]] = None) -> ItemAdvisor:
    """환경 변수(.env 포함)를 읽어 적절한 ItemAdvisor를 만듭니다. 키가 없으면 NullAdvisor."""
    if env is None:
        try:
            from dotenv import load_dotenv

            load_dotenv()  # 프로젝트 루트의 .env를 읽습니다(없어도 오류 없음).
        except ImportError:
            pass
        env = os.environ

    api_key = env.get("LLM_API_KEY") or env.get("BAZE_API_KEY")
    if not api_key:
        return NullAdvisor()
    return OpenAICompatibleAdvisor(
        api_key=api_key,
        base_url=env.get("LLM_BASE_URL") or DEFAULT_BASE_URL,
        model=env.get("LLM_MODEL") or DEFAULT_MODEL,
    )
