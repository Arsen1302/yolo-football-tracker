from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from football_analysis import process_video

# Էջի հիմնական կարգավորումներ
st.set_page_config(
    page_title="Football Match Analyzer",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Կարգավորված CSS ոճեր
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3em;
        background-color: #2563EB;
        color: white;
    }
    .stButton button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Կողային վահանակ (Sidebar) ---
with st.sidebar:
  st.markdown("### ⚽ Ֆուտբոլային Վերլուծություն")
  st.markdown("---")
  st.markdown("**Հիմնական հնարավորությունները:**")
  st.markdown("- 🏃‍♂️ Խաղացողների և մրցավարների հայտնաբերում")
  st.markdown("- 🎯 Օբյեկտների հետևում (Tracking)")
  st.markdown("- 📊 Շարժման վիճակագրություն")
  st.markdown("---")
  st.caption("AI Computer Vision • Powered by YOLO & Supervision")

# --- Հիմնական բովանդակություն ---
st.markdown(
    '<p class="main-title">⚽ AI Football Match Tracker</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Վերբեռնեք խաղի տեսանյութը՝ խաղացողներին հետևելու'
    " և շարժման ցուցանիշները վերլուծելու համար։</p>",
    unsafe_allow_html=True,
)

# Ֆայլի վերբեռնման քարտ
with st.container(border=True):
  st.markdown("#### 📤 Քայլ 1: Վերբեռնեք տեսանյութը")
  uploaded_video = st.file_uploader(
      "Ընտրեք ֆուտբոլային տեսանյութ (MP4, MOV, AVI, MKV)",
      type=["mp4", "mov", "avi", "mkv"],
      label_visibility="collapsed",
  )

  if uploaded_video is not None:
    st.success(f"Ֆայլը հաջողությամբ ընտրված է՝ **{uploaded_video.name}**")

# Վերլուծության կոճակ և գործընթաց
if uploaded_video is not None:
  st.markdown("<br>", unsafe_allow_html=True)
  if st.button("🚀 Սկսել վերլուծությունը", type="primary"):
    temporary_video_path = None
    try:
      suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"
      with tempfile.NamedTemporaryFile(
          suffix=suffix, delete=False
      ) as temporary_video:
        temporary_video_path = Path(temporary_video.name)
        temporary_video.write(uploaded_video.getbuffer())

      with st.spinner(
          "⏳ Տեսանյութը մշակվում է YOLO-ի և համակարգչային տեսողության"
          " միջոցով․․․"
      ):
        result_path = process_video(temporary_video_path)

      st.session_state["processed_video_path"] = result_path
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
      st.session_state.pop("processed_video_path", None)
      st.error(f"❌ Տեսանյութի մշակումը չհաջողվեց․ {error}")
    finally:
      if temporary_video_path is not None:
        temporary_video_path.unlink(missing_ok=True)

# Արդյունքների ցուցադրում
processed_video_path = st.session_state.get("processed_video_path")
if processed_video_path is not None:
  result = Path(processed_video_path)
  if result.is_file():
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
      st.markdown("#### ✨ Քայլ 2: Վերլուծության արդյունքը")
      st.success("🎉 Վերլուծությունն հաջողությամբ ավարտված է։")
      st.video(str(result))
      st.caption(f"📁 Արդյունքի ֆայլը պահպանվել է՝ `{result}`")
  else:
    st.session_state.pop("processed_video_path", None)
    st.error("⚠️ Մշակված տեսանյութի ֆայլը այլևս հասանելի չէ։")