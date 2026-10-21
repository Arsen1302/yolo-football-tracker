from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "presentation" / "football_analysis_presentation_hy_v2.pptx"

SLIDE_WIDTH = 13.333
SLIDE_HEIGHT = 7.5
BG = "0B1220"
PANEL = "121F30"
PANEL_LIGHT = "192A3D"
WHITE = "F4F7FB"
MUTED = "A9B7C8"
GREEN = "9BE564"
CYAN = "5DD6D0"
GOLD = "FFC96B"
RED = "FF7C78"
FONT = "Arial"


def color(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def add_shape(
    slide,
    shape_type,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    line: str | None = None,
    radius: bool = True,
):
    actual_shape_type = (
        shape_type
        if shape_type == MSO_SHAPE.OVAL or not radius
        else MSO_SHAPE.ROUNDED_RECTANGLE
    )
    shape = slide.shapes.add_shape(
        actual_shape_type,
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color(fill)
    shape.line.fill.background() if line is None else None
    if line is not None:
        shape.line.color.rgb = color(line)
        shape.line.width = Pt(1)
    if radius and shape_type == MSO_SHAPE.ROUNDED_RECTANGLE:
        shape.adjustments[0] = 0.12
    return shape


def add_text(
    slide,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    size: float,
    fill: str = WHITE,
    bold: bool = False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.MIDDLE,
    margin: float = 0.02,
):
    box = slide.shapes.add_textbox(
        Inches(x), Inches(y), Inches(width), Inches(height)
    )
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color(fill)
    return box


def add_paragraphs(
    slide,
    lines: list[tuple[str, str, float, bool]],
    x: float,
    y: float,
    width: float,
    height: float,
):
    box = slide.shapes.add_textbox(
        Inches(x), Inches(y), Inches(width), Inches(height)
    )
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.04)
    frame.margin_right = Inches(0.04)
    frame.margin_top = Inches(0.04)
    frame.margin_bottom = Inches(0.04)
    for index, (text, text_color, size, bold) in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(10)
        paragraph.level = 0
        run = paragraph.add_run()
        run.text = text
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color(text_color)
    return box


def new_slide(prs: Presentation, section: str, title: str, number: int):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color(BG)
    add_text(slide, section.upper(), 0.65, 0.35, 7.8, 0.27, 10, GREEN, True)
    add_text(slide, title, 0.65, 0.72, 11.9, 0.6, 25, WHITE, True)
    add_shape(slide, MSO_SHAPE.RECTANGLE, 0.65, 1.44, 0.55, 0.045, GREEN, radius=False)
    add_text(slide, f"{number:02d}", 12.15, 0.35, 0.48, 0.28, 10, MUTED, True, PP_ALIGN.RIGHT)
    return slide


def add_speaker_cue(slide, cue: str):
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.65, 6.84, 12.03, 0.4, PANEL)
    add_text(slide, "ԱՍԵԼ", 0.83, 6.89, 0.68, 0.22, 9, GREEN, True)
    add_text(slide, cue, 1.55, 6.87, 10.9, 0.26, 10.5, MUTED)


def add_card(
    slide,
    x: float,
    y: float,
    width: float,
    height: float,
    heading: str,
    body: str,
    accent: str = GREEN,
):
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, width, height, PANEL)
    add_shape(slide, MSO_SHAPE.RECTANGLE, x + 0.19, y + 0.25, 0.055, height - 0.5, accent, radius=False)
    add_text(slide, heading, x + 0.38, y + 0.2, width - 0.57, 0.42, 16, WHITE, True)
    add_text(slide, body, x + 0.38, y + 0.72, width - 0.6, height - 0.88, 12, MUTED, valign=MSO_ANCHOR.TOP)


def add_crop_picture(slide, image_path: Path, x: float, y: float, width: float, height: float):
    picture = slide.shapes.add_picture(
        str(image_path),
        Inches(x),
        Inches(y),
        width=Inches(width),
        height=Inches(height),
    )
    image_ratio = picture.image.size[0] / picture.image.size[1]
    box_ratio = width / height
    if image_ratio > box_ratio:
        crop = (1 - box_ratio / image_ratio) / 2
        picture.crop_left = crop
        picture.crop_right = crop
    else:
        crop = (1 - image_ratio / box_ratio) / 2
        picture.crop_top = crop
        picture.crop_bottom = crop
    return picture


def create_illustrative_frame(target: Path) -> None:
    frame = np.full((720, 1280, 3), (48, 105, 47), dtype=np.uint8)
    frame[:] = (56, 112, 47)
    height, width = frame.shape[:2]
    white = (235, 242, 232)
    cv2.rectangle(frame, (46, 38), (width - 46, height - 38), white, 3)
    cv2.line(frame, (width // 2, 38), (width // 2, height - 38), white, 3)
    cv2.circle(frame, (width // 2, height // 2), 92, white, 3)
    cv2.rectangle(frame, (46, 205), (210, 515), white, 3)
    cv2.rectangle(frame, (width - 210, 205), (width - 46, 515), white, 3)
    cv2.rectangle(frame, (46, 285), (115, 435), white, 3)
    cv2.rectangle(frame, (width - 115, 285), (width - 46, 435), white, 3)

    players = [
        (365, 230, (224, 93, 66), 8),
        (450, 385, (224, 93, 66), 11),
        (610, 285, (224, 93, 66), 16),
        (720, 440, (224, 93, 66), 4),
        (875, 235, (204, 190, 108), 6),
        (950, 365, (204, 190, 108), 12),
        (790, 515, (204, 190, 108), 17),
        (1035, 505, (204, 190, 108), 20),
    ]
    for x, y, team_color, player_id in players:
        cv2.ellipse(frame, (x, y + 30), (25, 9), 0, 0, 360, team_color, 4)
        cv2.circle(frame, (x, y), 18, team_color, cv2.FILLED)
        cv2.circle(frame, (x, y), 18, white, 2)
        cv2.putText(
            frame,
            str(player_id),
            (x - 11, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (15, 25, 32),
            2,
        )
    ball = (690, 385)
    cv2.circle(frame, ball, 9, (245, 245, 236), cv2.FILLED)
    cv2.circle(frame, ball, 9, (15, 25, 32), 2)
    cv2.putText(
        frame,
        "ILLUSTRATIVE TRACKING OUTPUT",
        (68, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        white,
        2,
    )
    if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"Could not save illustrative preview image: {target}")


def build_presentation() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_WIDTH)
    prs.slide_height = Inches(SLIDE_HEIGHT)
    prs.core_properties.title = "Ֆուտբոլային տեսանյութի վերլուծություն"
    prs.core_properties.subject = "Նախագծի պարզ, 10 րոպեանոց ներկայացում"
    prs.core_properties.author = "Football Analysis"
    prs.core_properties.keywords = "football, computer vision, YOLO, tracking"

    with tempfile.TemporaryDirectory(prefix="football-slides-") as temp_directory:
        sample_frame = Path(temp_directory) / "illustrative-frame.jpg"
        create_illustrative_frame(sample_frame)

        # 1 — Title
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = color(BG)
        add_text(slide, "ՆԱԽԱԳԾԻ ՆԵՐԿԱՅԱՑՈՒՄ  /  COMPUTER VISION", 0.78, 0.63, 6.3, 0.3, 10, GREEN, True)
        add_text(slide, "Ֆուտբոլային խաղի\nտեսավերլուծություն", 0.78, 1.36, 6.3, 1.6, 31, WHITE, True, valign=MSO_ANCHOR.TOP)
        add_text(slide, "Տեսանյութից՝ հայտնաբերում, հետևում և խաղային գնահատումներ", 0.82, 3.13, 5.8, 0.74, 16, MUTED, valign=MSO_ANCHOR.TOP)
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.82, 4.32, 2.55, 0.48, PANEL_LIGHT)
        add_text(slide, "Streamlit  •  YOLO  •  ByteTrack", 0.99, 4.41, 2.2, 0.25, 10, GREEN, True, PP_ALIGN.CENTER)
        add_crop_picture(slide, sample_frame, 7.0, 1.05, 5.55, 4.65)
        add_speaker_cue(slide, "Այս նախագիծը վերլուծում է ֆուտբոլային տեսանյութը և արդյունքը ցույց տալիս նշագրված տեսանյութով։")

        # 2 — Problem and goal
        slide = new_slide(prs, "01 / Նպատակ", "Ի՞նչ խնդիր է լուծում նախագիծը", 2)
        add_card(slide, 0.72, 1.83, 3.77, 3.85, "Մուտք", "Ֆուտբոլային տեսանյութ\n\nԿադրերը մշակվում են հերթականությամբ՝ առանց ձեռքով յուրաքանչյուր խաղացողին նշելու։", CYAN)
        add_card(slide, 4.78, 1.83, 3.77, 3.85, "Վերլուծություն", "Համակարգիչն ավտոմատ գտնում և հետևում է խաղի հիմնական օբյեկտներին։", GREEN)
        add_card(slide, 8.84, 1.83, 3.77, 3.85, "Ելք", "Ստանում ենք նշագրված MP4՝ խաղացողների ID-ներով և մոտավոր խաղային ցուցանիշներով։", GOLD)
        add_speaker_cue(slide, "Մուտքը տեսանյութն է, իսկ ելքը՝ կադրերի վրա հայտնաբերումներն ու գնահատումները ցույց տվող նոր տեսանյութը։")

        # 3 — Pipeline
        slide = new_slide(prs, "02 / Ընթացքը", "Տեսանյութը՝ 5 հասկանալի քայլով", 3)
        steps = [
            ("1", "Վերբեռնում", "Ֆայլ, FPS, առաջին կադր"),
            ("2", "YOLO", "Խաղացող, մրցավար, գնդակ"),
            ("3", "ByteTrack", "Նույնականացում կադրերի միջև"),
            ("4", "Խաղային գնահատում", "Թիմ, տիրություն, շարժում"),
            ("5", "MP4 արդյունք", "Նշագրում և ցուցադրում"),
        ]
        y = 1.9
        for index, (number, heading, body) in enumerate(steps):
            x = 0.9 + index * 2.48
            add_shape(slide, MSO_SHAPE.OVAL, x + 0.65, y, 0.55, 0.55, GREEN if index < 4 else CYAN)
            add_text(slide, number, x + 0.65, y + 0.06, 0.55, 0.36, 14, BG, True, PP_ALIGN.CENTER)
            add_text(slide, heading, x, y + 0.83, 1.85, 0.52, 15, WHITE, True, PP_ALIGN.CENTER)
            add_text(slide, body, x - 0.05, y + 1.5, 1.95, 0.82, 11, MUTED, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.TOP)
            if index < 4:
                add_shape(slide, MSO_SHAPE.RECTANGLE, x + 1.24, y + 0.25, 1.12, 0.035, PANEL_LIGHT, radius=False)
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 1.12, 4.72, 11.06, 1.15, PANEL)
        add_text(slide, "Կարևոր", 1.4, 4.91, 1.1, 0.3, 13, GOLD, True)
        add_text(slide, "Վերբեռնման պահին մոդելը չի մարզվում․ այն արդեն ուսուցված քաշերով կատարում է հայտնաբերում և հետևում։", 2.55, 4.86, 9.1, 0.54, 16, WHITE, True)
        add_speaker_cue(slide, "Ասա այս հոսքը՝ վերբեռնումից մինչև MP4։ Հիշիր՝ վերլուծությունը inference է, ոչ թե ուսուցում։")

        # 4 — Actual output frame
        slide = new_slide(prs, "03 / Իրական արդյունք", "Ի՞նչ է երևում մշակված կադրում", 4)
        add_crop_picture(slide, sample_frame, 0.72, 1.75, 8.15, 4.57)
        add_card(slide, 9.15, 1.83, 3.45, 1.18, "Շրջաններ և ID", "Հայտնաբերված խաղացողներ", GREEN)
        add_card(slide, 9.15, 3.18, 3.45, 1.18, "Գնդակի նշիչ", "Ընտրված գնդակի դիրք", CYAN)
        add_card(slide, 9.15, 4.53, 3.45, 1.48, "Խաղային overlay", "Տիրության բաժին,\nարագություն և հեռավորություն", GOLD)
        add_speaker_cue(slide, "Սա նախագծի մշակած տեսանյութից կադր է․ ցույց տուր ID-ները, գնդակի նշիչը և ցուցանիշները։")

        # 5 — How estimates are made
        slide = new_slide(prs, "04 / Մեթոդ", "Ինչպե՞ս են ստացվում գնահատումները", 5)
        add_card(slide, 0.82, 1.83, 5.65, 3.95, "Թիմ և գնդակի տիրություն", "1. Խաղացողների մարզաշապիկների գույները խմբավորվում են 2 թիմի։\n\n2. Գնդակի դիրքը համեմատվում է խաղացողների ոտքերի դիրքերի հետ։\n\n3. Ամենամոտ խաղացողը կարող է գնահատվել որպես տիրող։", GREEN)
        add_card(slide, 6.82, 1.83, 5.65, 3.95, "Շարժում և հեռավորություն", "1. Optical flow-ը գնահատում է տեսախցիկի շարժումը։\n\n2. Տեսադաշտի 4 կետի չափաբերումն օգնում է դիրքը փոխարկել դաշտի կոորդինատների։\n\n3. Կադրերի միջև փոփոխությունից ստացվում են մոտավոր արագություն և հեռավորություն։", CYAN)
        add_text(slide, "Գույների տարբերակում", 1.23, 5.99, 2.1, 0.26, 10, MUTED, True)
        add_text(slide, "K-Means", 3.3, 5.99, 1.3, 0.26, 10, GREEN, True)
        add_text(slide, "Դաշտի չափաբերում", 7.25, 5.99, 2.1, 0.26, 10, MUTED, True)
        add_text(slide, "Perspective transform", 9.33, 5.99, 2.45, 0.26, 10, CYAN, True)
        add_speaker_cue(slide, "Թվերը մոտավոր են․ դրանց ճշտությունը կախված է տեսախցիկից, հայտնաբերումից և դաշտի չափաբերումից։")

        # 6 — Data and experiments
        slide = new_slide(prs, "05 / Տվյալներ և փորձ", "Ի՞նչ ստուգեցի և ի՞նչ արդյունք ստացա", 6)
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.78, 1.8, 4.15, 4.68, PANEL)
        add_text(slide, "ROBOFLOW DATASET  •  V1", 1.08, 2.05, 3.5, 0.32, 10, GREEN, True)
        add_text(slide, "663", 1.1, 2.55, 1.2, 0.7, 34, WHITE, True)
        add_text(slide, "նկար", 2.1, 2.75, 0.8, 0.3, 13, MUTED)
        add_text(slide, "15,798", 1.1, 3.45, 2.1, 0.7, 32, CYAN, True)
        add_text(slide, "վավեր bbox", 3.12, 3.67, 1.2, 0.3, 12, MUTED)
        add_text(slide, "Train 612  ·  Val 38  ·  Test 13", 1.1, 4.45, 3.5, 0.36, 12, WHITE, True)
        add_text(slide, "Պիտակների բացակայություն / սխալ՝ 0", 1.1, 5.16, 3.5, 0.4, 11, MUTED)

        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 5.22, 1.8, 7.3, 4.68, PANEL)
        add_text(slide, "YOLOv8n  •  20 EPOCH  •  MPS", 5.56, 2.05, 4.5, 0.32, 10, GREEN, True)
        add_text(slide, "Validation mAP50–95", 5.56, 2.58, 3.2, 0.35, 14, WHITE, True)
        experiments = [
            ("lr 0.001", 0.4272, CYAN),
            ("lr 0.01  •  ընտրված", 0.4378, GREEN),
        ]
        for index, (label, score, accent) in enumerate(experiments):
            y = 3.2 + index * 0.78
            add_text(slide, label, 5.58, y, 2.25, 0.28, 11, MUTED, True)
            add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 7.92, y + 0.04, 3.4, 0.2, PANEL_LIGHT)
            add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 7.92, y + 0.04, 3.4 * score / 0.5, 0.2, accent)
            add_text(slide, f"{score:.4f}", 11.42, y - 0.02, 0.68, 0.3, 11, accent, True, PP_ALIGN.RIGHT)
        add_text(slide, "Ընտրվածի test mAP50–95  •  0.4421", 5.58, 5.13, 5.9, 0.42, 17, WHITE, True)
        add_text(slide, "Test հավաքածուն՝ ընդամենը 13 նկար", 5.58, 5.78, 5.9, 0.3, 11, GOLD, True)
        add_speaker_cue(slide, "Երկու YOLOv8n learning-rate փորձ ավարտվեց։ Test-ը փոքր է, դրա արդյունքը անկայուն է։")

        # 7 — Limitations and next steps
        slide = new_slide(prs, "06 / Սահմանափակումներ", "Արդյունքները ներկայացնում եմ ազնվորեն", 7)
        add_card(slide, 0.82, 1.83, 3.75, 3.95, "Փոքր test", "13 նկարից ստացված չափանիշը բավարար չէ լայն ընդհանրացման պնդման համար։", GOLD)
        add_card(slide, 4.79, 1.83, 3.75, 3.95, "Չավարտված փորձ", "YOLOv8s-ի ուսուցումը կանգ առավ՝ սկավառակի ազատ տարածքի պակասի պատճառով։", RED)
        add_card(slide, 8.76, 1.83, 3.75, 3.95, "Չափաբերում", "Արագությունն ու հեռավորությունը կախված են տեսախցիկի տեսանկյունից և դաշտի 4 կետից։", CYAN)
        add_text(slide, "Հաջորդ քայլեր", 0.98, 6.0, 1.75, 0.32, 12, GREEN, True)
        add_text(slide, "Ազատել պահեստը → ավարտել մյուս փորձերը → բարելավել դաշտի չափաբերումը", 2.7, 5.98, 9.2, 0.38, 13, WHITE, True)
        add_speaker_cue(slide, "Մի թաքցրու սահմանափակումները․ ասա, որ մոդելների ամբողջական համեմատությունը դեռ ավարտված չէ։")

        # 8 — Wrap-up
        slide = new_slide(prs, "07 / Ամփոփում", "Մի նախադասությամբ՝ նախագծիս մասին", 8)
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.82, 1.85, 11.65, 2.1, PANEL)
        add_text(
            slide,
            "«Համակարգը ֆուտբոլային տեսանյութը վերածում է\n"
            "հայտնաբերումներով և խաղային գնահատումներով նշագրված MP4-ի»",
            1.22,
            2.18,
            10.8,
            1.27,
            23,
            WHITE,
            True,
            PP_ALIGN.CENTER,
        )
        add_text(slide, "Վերբեռնում", 1.25, 4.62, 2.0, 0.42, 16, GREEN, True, PP_ALIGN.CENTER)
        add_text(slide, "→", 3.42, 4.62, 0.45, 0.42, 19, MUTED, True, PP_ALIGN.CENTER)
        add_text(slide, "Հայտնաբերում և հետևում", 3.94, 4.62, 3.2, 0.42, 15, CYAN, True, PP_ALIGN.CENTER)
        add_text(slide, "→", 7.3, 4.62, 0.45, 0.42, 19, MUTED, True, PP_ALIGN.CENTER)
        add_text(slide, "Նշագրված արդյունք", 7.85, 4.62, 3.05, 0.42, 16, GREEN, True, PP_ALIGN.CENTER)
        add_text(slide, "Շնորհակալություն", 4.55, 5.55, 4.2, 0.52, 23, WHITE, True, PP_ALIGN.CENTER)
        add_speaker_cue(slide, "Ամփոփիր մեկ նախադասությամբ, նշիր սահմանափակումը, հետո անցիր հարցերին։")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        prs.save(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(build_presentation())
