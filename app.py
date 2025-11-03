"""Streamlit app for generating imaginative scenes from book passages using Vertex AI Imagen."""

from __future__ import annotations
import os
import streamlit as st
import vertexai
from vertexai.vision_models import ImageGenerationModel


PROJECT_ID = "sesac-yoojikol"
LOCATION = "us-central1"
KEY_PATH = "gcp-key.json"
MODEL_NAME = "imagen-4.0-generate-001"


def _init_vertex_ai() -> None:
    """Initialise the Vertex AI SDK if the service account key is present."""

    if not os.path.exists(KEY_PATH):
        st.error(f"'{KEY_PATH}' 키 파일이 없습니다.")
        st.stop()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
    except Exception as exc:
        st.error(f"Vertex AI 초기화 실패: {exc}")
        st.stop()


def _build_prompt(
    passage: str,
    *,
    art_style: str,
    mood: str,
    color_palette: str,
    detail_level: str,
    camera_focus: str,
    era: str,
) -> str:
    """Combine the passage and stylistic controls into a single Imagen prompt."""

    parts: list[str] = []
    passage = passage.strip()
    if passage:
        parts.append(passage)

    def _append_if_content(label: str, value: str) -> None:
        if value and value != "기본" and value != "(선택 안 함)":
            parts.append(f"{label}: {value}")

    _append_if_content("Art style", art_style)
    _append_if_content("Mood", mood)
    _append_if_content("Colour palette", color_palette)
    _append_if_content("Detail", detail_level)
    _append_if_content("Camera", camera_focus)
    _append_if_content("Era", era)

    return " | ".join(parts)


def main() -> None:
    st.set_page_config(page_title="Imagen Book Scene Studio", page_icon="📖")
    st.title("📖 책 장면을 이미지로 그려보기")
    st.caption(
        "책의 문장을 기반으로 상상 속 장면을 시각화하세요. 옵션을 활용해 원하는 분위기와 스타일을 더해볼 수 있습니다."
    )

    _init_vertex_ai()

    sample_passages = {
        "마법 학교의 연회장": "촛불이 허공에 떠 있고 긴 식탁이 늘어선 고딕풍 연회장.",
        "SF 우주 정거장": "거대한 유리창 너머로 푸른 행성이 보이고, 금속 질감의 복도가 이어진다.",
        "고전 추리극": "비에 젖은 런던 골목, 가스등 아래 실루엣으로 보이는 탐정의 모습.",
    }

    with st.sidebar:
        st.header("🎨 스타일 옵션")
        selected_sample = st.selectbox("샘플 장면 불러오기", ["직접 입력"] + list(sample_passages))
        art_style = st.selectbox(
            "아트 스타일",
            [
                "기본",
                "수채화 일러스트",
                "시네마틱 사진",
                "디지털 페인팅",
                "유화",
                "픽셀 아트",
            ],
        )
        mood = st.selectbox(
            "분위기",
            [
                "기본",
                "따뜻하고 포근한",
                "어둡고 미스터리한",
                "서스펜스 넘치는",
                "감성적인",
                "장엄하고 웅장한",
            ],
        )
        color_palette = st.selectbox(
            "색상",
            ["기본", "따뜻한 색조", "차가운 색조", "모노톤", "파스텔", "선명한 대비"],
        )
        detail_level = st.selectbox(
            "디테일",
            ["기본", "초고해상도", "울트라 디테일", "꿈결 같은 소프트 포커스"],
        )
        camera_focus = st.selectbox(
            "카메라 연출",
            [
                "(선택 안 함)",
                "광각 뷰",
                "드론 뷰",
                "클로즈업",
                "시점 샷 (POV)",
                "시네마틱 와이드샷",
            ],
        )
        era = st.selectbox(
            "시대/배경",
            ["(선택 안 함)", "현대", "중세 판타지", "빅토리아 시대", "사이버펑크", "포스트 아포칼립스"],
        )

    default_passage = ""
    if selected_sample != "직접 입력":
        default_passage = sample_passages[selected_sample]

    passage = st.text_area(
        "책 속 문장이나 장면 설명",
        value=default_passage,
        height=180,
        placeholder="장면을 묘사하는 문장을 입력하세요",
    )

    final_prompt = _build_prompt(
        passage,
        art_style=art_style,
        mood=mood,
        color_palette=color_palette,
        detail_level=detail_level,
        camera_focus=camera_focus,
        era=era
    )

    if st.button("🎨 이미지 생성", type="primary"):
        if not passage.strip():
            st.warning("책 속 장면을 입력해주세요.")
            st.stop()

        with st.spinner("Imagen 모델이 상상을 그리는 중입니다..."):
            try:
                model = ImageGenerationModel.from_pretrained(MODEL_NAME)
                response = model.generate_images(
                    prompt=final_prompt
                )

                st.image(response[0]._image_bytes, caption="Result #1", width='stretch')
                st.success("완료!")
            
            except Exception as exc:
                st.error(f"오류: {exc}")


if __name__ == "__main__":
    main()