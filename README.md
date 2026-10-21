# Ֆուտբոլային խաղերի վերլուծություն

Նախագիծը ֆուտբոլային տեսանյութ է մշակում Computer Vision մեթոդներով։ YOLO
մոդելը հայտնաբերում է խաղացողներին, մրցավարներին և գնդակը, ByteTrack-ը
հետևում է հայտնաբերումներին, իսկ հաջորդ փուլերը գնահատում են թիմերը,
գնդակի տիրումն ու խաղացողների շարժումը։ Արդյունքը դիտվում է Streamlit
ինտերֆեյսում և պահպանվում MP4 ձևաչափով։

## Հնարավորություններ

- Խաղացողների, մրցավարների և գնդակի հայտնաբերում ու track ID-ների պահպանում։
- Գնդակի թեկնածուների ընտրություն՝ չափի, ձևի, վստահության և դիրքի շարունակականությամբ։
- Թիմերի մոտավոր տարբերակում՝ համազգեստի գույների K-Means խմբավորմամբ։
- Տեսախցիկի շարժման գնահատում՝ Lucas–Kanade optical flow-ով։
- Կոորդինատների հեռանկարային փոխակերպում և արագության/հեռավորության մոտավոր գնահատում։
- Տեսանյութի վերբեռնում և արդյունքի դիտում Streamlit վեբ հավելվածում։
- YOLO dataset-ի audit/EDA և train/validation-ով մոդելների համեմատման փորձարկիչ։

Արագությունն ու հեռավորությունը ground truth չափումներ չեն։ Դրանց ճշտությունը
կախված է տեսանյութից, հայտնաբերումներից և խաղադաշտի չափաբերումից՝
`view_transformer/`-ում։

## Պահանջներ

- Python 3.11
- `requirements.txt`-ի փաթեթները
- Հավելվածի քաշերը՝ `models/yolo-football-player-detection.pt` ճանապարհին

Մոդելի քաշերը, տեսանյութերը, dataset-ը և ուսուցման արդյունքները Git-ում չեն
ներառվում։ Հավելվածի համար կարող եք նշել այլ քաշերի ճանապարհ՝
`FOOTBALL_ANALYSIS_MODEL_PATH` միջավայրի փոփոխականով։

## Տեղային գործարկում

Նախագծի հիմնական պանակից՝

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py --server.port 7860
```

Բացեք [http://localhost:7860](http://localhost:7860), վերբեռնեք MP4, MOV, AVI
կամ MKV տեսանյութ և սեղմեք «Վերլուծել տեսանյութը»։ Արդյունքը պահպանվում է
`output_videos/`-ում։ Երկար տեսանյութերի մշակումը կարող է ժամանակ պահանջել։
Streamlit-ի մեկ ֆայլի լռելյայն սահմանաչափը 200 MB է։

Հավելվածի համար այլ մոդել օգտագործելու օրինակ՝

```bash
FOOTBALL_ANALYSIS_MODEL_PATH=/path/to/best.pt streamlit run app.py --server.port 7860
```

Առանց վեբ ինտերֆեյսի գործարկելու համար օգտագործեք CLI-ն․

```bash
python main.py path/to/match.mp4 --output output_videos/result.mp4 --model /path/to/best.pt
```

Մուտքային տեսանյութը, `--output`-ը և `--model`-ը կամընտիր են։ Առանց մուտքային
ճանապարհի օգտագործվում է
`input_videos/Screen Recording 2026-09-20 at 16.27.37.mov`-ը, իսկ մոդելի համար՝
`FOOTBALL_ANALYSIS_MODEL_PATH`-ը կամ նախագծի լռելյայն քաշերի ճանապարհը։

## Dataset

EDA-ի և ուսուցման համար ընտրված է Roboflow-ի
[football-players-detection v1](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)
YOLOv5 export-ը։ Dataset-ի լիցենզիան **CC BY 4.0** է․ օգտագործման կամ
տարածման ժամանակ նշեք Roboflow-ի աղբյուրը և լիցենզիան։

Տեղային dataset-ի սպասվող կառուցվածքը՝

```text
<dataset-root>/
├── data.yaml
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

Օրինակ՝

```bash
export FOOTBALL_DATASET_ROOT="$HOME/Downloads/football-players-detection.v1i.yolov5pytorch"
```

Գործիքները ստեղծում են ուղղված Ultralytics YAML՝ փորձերի output պանակում և չեն
փոփոխում սկզբնական dataset-ի YAML-ը կամ պիտակները։

## EDA

Բացեք `notebooks/01_football_dataset_eda.ipynb` VS Code-ի Jupyter միջավայրում
կամ այլ notebook editor-ում։ Notebook-ը ստուգում է split-երը, բացակայող,
դատարկ կամ որբ label-երը, անվավեր bbox-երը, դասերի բաշխումը, bbox չափերը,
նկարագրական engineered հատկանիշներն ու նրանց կորելացիաները։ Այն նաև ցուցադրում
է պիտակավորված պատկերների օրինակներ։

Dataset-ի նախնական աուդիտի արդյունքը՝

| Split | Նկարներ | Bounding box-եր |
|---|---:|---:|
| Train | 612 | 14,583 |
| Validation | 38 | 905 |
| Test | 13 | 310 |
| **Ընդամենը** | **663** | **15,798** |

Dataset-ի աուդիտի ժամանակ label ֆայլերի բացակայություն, դատարկ label-եր կամ
անվավեր bbox տողեր չեն հայտնաբերվել։ Notebook-ը կրկնարկելիս համեմատեք
արդյունքները տեղային dataset-ի փաստացի ելքի հետ։

Train/validation/test split-երը վերցված են dataset-ից։ Test-ը ընդամենը 13 նկար
ունի, հետևաբար դրա չափանիշները շատ անկայուն են և ընդհանրացվող արդյունք չեն։
Notebook-ի բջիջները կրկնեք ամբողջությամբ՝ տեղային տվյալներից աուդիտն ու
գծապատկերները ստանալու համար։

## Մոդելների ուսուցում, tuning և փորձերի հետագծում

Հրամանն իրականացնում է մոդելների և learning rate-երի grid, բոլոր տարբերակները
համեմատում է validation mAP50-95-ով, ընտրում լավագույնը, ապա test split-ը
գնահատում է միայն այդ ընտրված մոդելի համար։

```bash
python -m src.train_models \
  --dataset-root "$FOOTBALL_DATASET_ROOT" \
  --models yolov8n.pt,yolov8s.pt \
  --learning-rates 0.001,0.01 \
  --epochs 20 \
  --imgsz 640 \
  --batch 4
```

Լռելյայն օգտագործվում է հասանելի սարքը՝ CUDA, Apple MPS կամ CPU։ Կարող եք
նշել սարքը `--device`-ով և փոխել փորձի պարամետրերը։ Արդյունքները պահվում են
`outputs/experiments/`-ում, որը Git-ից բացառված է․

- `experiment_metrics.csv` — բոլոր run-երի պարամետրերը, վիճակները, սխալներն ու validation արդյունքները,
- `selected_model.json` — validation չափանիշով ընտրված քաշերը և test արդյունքը,
- `runs/` — յուրաքանչյուր run-ի լավագույն քաշերը և Ultralytics run տվյալները։

Training-ը կարող է երկար տևել և մեծ հիշողություն/պահեստ օգտագործել։ Հրամանի
ավարտից հետո ստուգեք CSV-ն և JSON-ը, նախքան արդյունքները ներկայացնելը։

### Իրական փորձարկման արդյունք (Apple M1, MPS)

Այս միջավայրում 20 epoch-ով ավարտվել են երկու YOLOv8n learning-rate փորձերը։
Validation mAP50-95-ով ընտրվել է `lr0=0.01`-ը։ YOLOv8s փորձը դադարել է
առաջին epoch-ի ընթացքում՝ checkpoint-ը պահելիս պահեստի ազատ տարածքի պակասի
պատճառով, ուստի ամբողջական ճարտարապետությունների համեմատությունը չի ավարտվել։

| Մոդել | `lr0` | Validation mAP50 | Validation mAP50-95 | Վիճակ |
|---|---:|---:|---:|---|
| YOLOv8n | 0.001 | 0.6637 | 0.4272 | Ավարտված |
| YOLOv8n | 0.01 | 0.6868 | 0.4378 | Ընտրված |
| YOLOv8s | 0.001 | — | — | Չավարտված՝ սկավառակի տարածքի սխալ |
| YOLOv8s | 0.01 | — | — | Չի գործարկվել |

Ընտրված YOLOv8n մոդելը test split-ում մեկ անգամ տվել է mAP50 **0.7186** և
mAP50-95 **0.4421** (precision 0.8570, recall 0.6303)։ Այս test արդյունքը
միայն 13 պատկերից է և խիստ անկայուն է․ այն չի կարելի դիտարկել որպես
ընդհանրացվող որակի ապացույց։ Ճարտարապետությունների ամբողջական համեմատությունն
ավարտելուց առաջ համոզվեք, որ համակարգում բավարար պահեստ կա։ Արդյունքները և
ընտրված քաշերի ճանապարհը պահվում են `outputs/experiments/`-ում։ MPS-ի որոշ
օպերացիաների համար PyTorch-ը զգուշացրել է, որ դրանք խիստ դետերմինիստիկ չեն,
ուստի նույն seed-ով կրկնված փորձերի արդյունքները կարող են փոքր-ինչ տարբերվել։

## Docker

Համոզվեք, որ հավելվածի քաշերը հասանելի են build-ի ժամանակ՝
`models/yolo-football-player-detection.pt` ճանապարհին, ապա՝

```bash
docker build -t football-analysis .
docker run --rm -p 7860:7860 \
  -v "$(pwd)/output_videos:/app/output_videos" \
  football-analysis
```

Կոնտեյները լսում է `PORT` փոփոխականը (կանխադրված՝ 7860)։ Տեղային մուտքային
տեսանյութերն ու ուսուցման dataset-ը Docker image չեն պատճենվում, իսկ մոդելի
քաշը պետք է տրամադրվի build-ի կամ deploy-ի ժամանակ։

## Deploy

Railway-ում միացրեք GitHub պահոցը կամ օգտագործեք Railway CLI-ն․ կօգտագործվի
նախագծի `Dockerfile`-ը։ Համոզվեք, որ մոդելի քաշը deploy-ին հասանելի է, և
ծառայությունը լսում է Railway-ի `PORT` փոփոխականը։ Մեծ մոդելների համար
օգտագործեք ձեր տեղակայման հարթակի աջակցած մոդելային պահեստավորումը և
`FOOTBALL_ANALYSIS_MODEL_PATH`-ով նշեք ներբեռնված քաշի ճանապարհը։ Նախ փորձարկեք
կարճ հոլովակով և հետևեք հիշողության օգտագործմանը։

## Կառուցվածք

```text
.
├── app.py
├── main.py
├── football_analysis/
│   ├── config.py
│   └── pipeline.py
├── src/
│   ├── football_dataset.py
│   └── train_models.py
├── notebooks/
│   └── 01_football_dataset_eda.ipynb
├── data/                         # Dataset-ի տեղադրման ուղեցույց
├── trackers/
├── camera_movement_estimator/
├── view_transformer/
├── speed_and_distance_estimator/
├── team_assigner/
├── player_ball_assigner/
├── utils/
└── models/                       # Տեղային YOLO քաշեր, Git-ից դուրս
```

Հավելվածի շերտերը բաժանված են հետևյալ կերպ․

- `app.py` — Streamlit UI, միայն վերբեռնման և արդյունքի ցուցադրման համար։
- `main.py` — CLI մուտքի կետ։
- `football_analysis/config.py` — մոդելի և ելքային ֆայլերի ճանապարհների կառավարում։
- `football_analysis/pipeline.py` — տեսանյութի վերլուծության use-case-ի հերթականություն։
- `trackers/`, `team_assigner/`, `camera_movement_estimator/`,
  `view_transformer/`, `speed_and_distance_estimator/`,
  `player_ball_assigner/` — առանձնացված վերլուծական բաղադրիչներ։
- `utils/` — ընդհանուր տեսանյութի և bounding-box օժանդակ ֆունկցիաներ։
- `src/` և `notebooks/` — training/EDA գործիքներ, որոնք տարանջատված են
  production inference pipeline-ից։
- `presentation/football_analysis_presentation_hy_v2.pptx` — նախագծի 8 սլայդանոց
  հայերեն ներկայացումը՝ խոսելու կարճ հուշումներով։
- `presentation/build_presentation.py` — ներկայացման վերակառուցման script։

UI-ն ու CLI-ն նույն `football_analysis.process_video` ծառայությունն են
օգտագործում, ուստի processing տրամաբանությունը կրկնօրինակված չէ։ Կարգավորման
և pipeline-ի հիմնական վարքը ստուգելու համար գործարկեք՝

```bash
python -m unittest discover -s tests -v
```
