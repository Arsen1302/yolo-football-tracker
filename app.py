from pathlib import Path
import tempfile

import streamlit as st

from football_analysis import process_video


st.set_page_config(page_title="Football Analysis", page_icon="⚽")

st.title("Football Analysis")
st.write(
    "Վերբեռնեք խաղի տեսանյութ՝ խաղացողներին ու գնդակը հայտնաբերելու, "
    "հետևելու և շարժման ցուցանիշները գնահատելու համար։"
)
st.caption(
    "Մշակման ժամանակը կախված է տեսանյութի տևողությունից և սարքի հզորությունից։ "
    "Արագությունն ու հեռավորությունը մոտավոր գնահատականներ են։"
)

uploaded_video = st.file_uploader(
    "Վերբեռնեք ֆուտբոլային տեսանյութ",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_video is not None and st.button("Վերլուծել տեսանյութը", type="primary"):
    temporary_video_path = None
    try:
        suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_video:
            temporary_video_path = Path(temporary_video.name)
            temporary_video.write(uploaded_video.getbuffer())

        with st.spinner("Տեսանյութը մշակվում է․ խնդրում ենք սպասել։"):
            result_path = process_video(temporary_video_path)

        st.session_state["processed_video_path"] = result_path
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        st.session_state.pop("processed_video_path", None)
        st.error(f"Տեսանյութի մշակումը չհաջողվեց․ {error}")
    finally:
        if temporary_video_path is not None:
            temporary_video_path.unlink(missing_ok=True)

processed_video_path = st.session_state.get("processed_video_path")
if processed_video_path is not None:
    result = Path(processed_video_path)
    if result.is_file():
        st.success("Վերլուծությունն ավարտված է։")
        st.video(str(result))
        st.caption(f"Արդյունքը պահպանվել է՝ {result}")
    else:
        st.session_state.pop("processed_video_path", None)
        st.error("Մշակված տեսանյութի ֆայլը այլևս հասանելի չէ։")