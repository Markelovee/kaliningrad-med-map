import io
import zipfile

import streamlit as st
import pandas as pd
from lxml import etree
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

import streamlit as st

DARK_CSS = """
<style>
/* Основная область приложения */
[data-testid="stAppViewContainer"] {
    background-color: #111111 !important;
    color: #f5f5f5 !important;
}

/* Боковая панель */
[data-testid="stSidebar"] {
    background-color: #141414 !important;
    color: #f5f5f5 !important;
}

/* Заголовки и обычный текст */
h1, h2, h3, h4, h5, h6, p, span, label {
    color: #f5f5f5 !important;
}
</style>
"""

LIGHT_CSS = """
<style>
/* Основная область приложения */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
    color: #111111 !important;
}

/* Боковая панель */
[data-testid="stSidebar"] {
    background-color: #f5f5f5 !important;
    color: #111111 !important;
}

/* Заголовки и обычный текст */
h1, h2, h3, h4, h5, h6, p, span, label {
    color: #111111 !important;
}

/* Кнопки: st.button, st.download_button и др. */
.stButton > button, .stDownloadButton > button {
    background-color: #f0f0f0 !important;
    color: #111111 !important;
    border: 1px solid #cccccc !important;
    border-radius: 4px !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #e0e0e0 !important;
}

/* Зона загрузки файлов */
[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    border: 1px dashed #999999 !important;
    color: #111111 !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #111111 !important;
}

/* --- ДОПОЛНИТЕЛЬНО: делаем светлыми инпуты и таблицу --- */

/* Обычные поля ввода (текст, числа, select и т.п.) */
input, textarea, select {
    background-color: #ffffff !important;
    color: #111111 !important;
}

/* Таблица st.data_editor / st.dataframe */
[data-testid="stDataFrame"], [data-testid="stDataFrame"] div {
    background-color: #ffffff !important;
    color: #111111 !important;
}
</style>
"""




# === Константы ===
SVG_FILE = "map2.svg"          # твой шаблон карты
NEUTRAL_FILL = "#eeeeee"       # цвет «заглушки» для фильтра top5/bottom5

# Список округов строго по id в SVG
DISTRICTS = [
    "kaliningrad",
    "baltiysk",
    "ladushkin",
    "mamonovo",
    "pionersky",
    "svetly",
    "svetlogorsk",
    "sovetsky",
    "yantarny",
    "zelenogradsky",
    "gurevsky",
    "bagrationovsky",
    "pravdinsky",
    "ozersky",
    "nesterovsky",
    "gusevsky",
    "chernyahovsky",
    "gvardeisky",
    "polessky",
    "nemansky",
    "slavsky",
    "krasnoznamensky",
]

DISTRICT_LABELS = {
    "kaliningrad": "Калининград",
    "baltiysk": "Балтийск",
    "ladushkin": "Ладушкин",
    "mamonovo": "Мамоново",
    "pionersky": "Пионерский",
    "svetly": "Светлый",
    "svetlogorsk": "Светлогорск",
    "sovetsky": "Советск",
    "yantarny": "Янтарный",
    "zelenogradsky": "Зеленоградский",
    "gurevsky": "Гурьевский",
    "bagrationovsky": "Багратионовский",
    "pravdinsky": "Правдинский",
    "ozersky": "Озёрский",
    "nesterovsky": "Нестеровский",
    "gusevsky": "Гусевский",
    "chernyahovsky": "Черняховский",
    "gvardeisky": "Гвардейский",
    "polessky": "Полесский",
    "nemansky": "Неманский",
    "slavsky": "Славский",
    "krasnoznamensky": "Краснознаменский",
}

# ---------- Вспомогательные функции ----------


def value_to_color_fn(values, cmap_name):
    """По набору значений создаёт функцию value -> hex-цвет."""
    positives = [float(v) for v in values if v > 0]
    if positives:
        vmin = min(positives)
        vmax = max(positives)
        if vmin == vmax:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0  # фиктивный диапазон

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    def _f(v):
        r, g, b, _ = [int(255 * c) for c in cmap(norm(float(v)))]
        return f"#{r:02x}{g:02x}{b:02x}"

    return _f


def set_fill(elem, color):
    """Задаёт цвет fill для элемента и аккуратно правит style."""
    elem.set("fill", color)
    style = elem.get("style")
    if style:
        parts = style.split(";")
        new_parts = []
        fill_handled = False
        for p in parts:
            p_stripped = p.strip()
            if p_stripped.startswith("fill:"):
                new_parts.append(f"fill:{color}")
                fill_handled = True
            elif p_stripped:
                new_parts.append(p_stripped)
        if not fill_handled:
            new_parts.append(f"fill:{color}")
        elem.set("style", ";".join(new_parts))
    else:
        elem.set("style", f"fill:{color}")


def get_filtered_ids(values_dict, mode, n=5):
    """Список id округов для фильтра (top5 / bottom5) или None."""
    items = list(values_dict.items())
    if mode == "top5":
        sorted_items = sorted(items, key=lambda kv: kv[1], reverse=True)
    elif mode == "bottom5":
        sorted_items = sorted(items, key=lambda kv: kv[1])
    else:
        return None
    return [k for k, v in sorted_items[:n]]


def colorize_svg(values_dict, cmap_name, filter_mode, show_labels=True):
    """Возвращает (svg_text, svg_bytes) с перекрашенной картой + hover-эффект и tooltip."""
    parser = etree.XMLParser(ns_clean=True, recover=True)
    tree = etree.parse(SVG_FILE, parser)
    root = tree.getroot()

    # --- CSS для эффекта наведения ---
    # Добавляем style в корень SVG (один раз)
    style_elem = etree.Element("{http://www.w3.org/2000/svg}style")
    style_elem.text = """
    .region:hover,
    .region:hover path,
    .region:hover polygon,
    .region:hover rect {
        stroke: #000000;
        stroke-width: 2;
        cursor: pointer;
        filter: brightness(1.15);
    }
    """
    # Вставляем style в начало, чтобы он применялся ко всем элементам
    root.insert(0, style_elem)

    # Функция перевода значения в цвет
    color_fn = value_to_color_fn(values_dict.values(), cmap_name)

    # Для фильтра top5/bottom5
    selected_ids = None
    if filter_mode in ("top5", "bottom5"):
        ids = get_filtered_ids(values_dict, filter_mode, n=5)
        selected_ids = set(ids) if ids else set()

    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Проходим по всем элементам с id (округа)
    for elem in root.xpath(".//*[@id]"):
        elem_id = elem.get("id")
        if elem_id not in values_dict:
            continue

        # ---- помечаем округ классом region (для hover-эффекта) ----
        existing_class = elem.get("class") or ""
        classes = existing_class.split()
        if "region" not in classes:
            classes.append("region")
        elem.set("class", " ".join(classes))

        value = values_dict[elem_id]

        # Выбор цвета с учётом фильтра
        if selected_ids is not None and elem_id not in selected_ids:
            color = NEUTRAL_FILL
        else:
            if value == 0:
                color = "#ffffff"
            else:
                color = color_fn(value)

        # Задаём цвет самому элементу
        set_fill(elem, color)

        # И всем дочерним path/polygon/rect внутри
        for child in elem.xpath(".//*"):
            if child.tag.endswith("path") or child.tag.endswith("polygon") or child.tag.endswith("rect"):
                set_fill(child, color)

        # ---------- TOOLTIP (подсказки) ----------

        # 1) Удаляем все старые <title> внутри этого округа
        for old_title in elem.xpath(".//svg:title", namespaces=ns):
            parent = old_title.getparent()
            if parent is not None:
                parent.remove(old_title)

        # 2) Текст подсказки
        pretty_name = DISTRICT_LABELS.get(elem_id, elem_id)
        title_text = f"{pretty_name}: {value}"

        # helper: повесить <title> на узел
        def attach_title(node):
            t = etree.SubElement(node, "{http://www.w3.org/2000/svg}title")
            t.text = title_text

        # 3) Вешаем title на группу округа
        attach_title(elem)

        # 4) И отдельно на все геометрические фигуры внутри округа
        for child in elem.xpath(".//*"):
            if child.tag.endswith("path") or child.tag.endswith("polygon") or child.tag.endswith("rect"):
                attach_title(child)

    # Скрываем подписи, если надо
    if not show_labels:
        label_ids = ["_блок_текста", "_названия", "_цифры"]
        for lid in label_ids:
            for elem in root.xpath(f".//*[@id='{lid}']"):
                style = elem.get("style") or ""
                parts = [p.strip() for p in style.split(";") if p.strip()]
                parts = [p for p in parts if not p.startswith("display:")]
                parts.append("display:none")
                elem.set("style", ";".join(parts))

    # Сохраняем SVG в память
    buffer = io.BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=True)
    svg_bytes = buffer.getvalue()
    svg_text = svg_bytes.decode("utf-8")
    return svg_text, svg_bytes




def show_legend(container, cmap_name, values_dict):
    """Рисует легенду по ненулевым значениям в указанном контейнере."""
    positive_values = [v for v in values_dict.values() if v > 0]
    if not positive_values:
        container.info("Все значения равны 0 — диапазон не требуется.")
        return

    vmin = min(positive_values)
    vmax = max(positive_values)
    if vmin == vmax:
        vmax = vmin + 1.0

    cmap = cm.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(4, 0.6))
    fig.subplots_adjust(bottom=0.5)

    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
    )
    cb.set_label("Значение (для > 0)")

    container.pyplot(fig)


def build_filter_table(values_dict, mode, n=5):
    """Таблица из top5/bottom5 округов."""
    ids = get_filtered_ids(values_dict, mode, n=n)
    if not ids:
        return None
    rows = [{"district_id": did, "value": values_dict[did]} for did in ids]
    return pd.DataFrame(rows)


def render_map(values, cmap_name, filter_mode, show_labels,
               title, legend_title, download_name):
    """Рисует одну карту и возвращает svg_bytes."""
    svg_text, svg_bytes = colorize_svg(values, cmap_name, filter_mode, show_labels)

    st.subheader(title)

    svg_responsive = svg_text.replace(
        "<svg ",
        "<svg style='width:100%;height:auto;' ",
        1,
    )
    st.components.v1.html(svg_responsive, height=700, scrolling=False)

    st.subheader(legend_title)
    show_legend(st, cmap_name, values)

    if filter_mode in ("top5", "bottom5"):
        filt_df = build_filter_table(values, filter_mode, n=5)
        if filt_df is not None:
            if filter_mode == "top5":
                table_title = "5 округов с максимальными значениями"
            else:
                table_title = "5 округов с минимальными значениями"
            st.subheader(table_title)
            st.table(filt_df)

    st.subheader("Скачать карту")
    st.download_button(
        label="SVG",
        data=svg_bytes,
        file_name=download_name,
        mime="image/svg+xml",
    )

    return svg_bytes


# ---------- UI ----------

st.set_page_config(page_title="Карта Калининградской области", layout="wide")

with st.sidebar:
    st.header("Настройки")

    theme = st.radio(
        "Тема интерфейса",
        ["Светлая", "Тёмная"],
        index=0,
        key="ui_theme",
    )


# Простейшее переключение фона
if theme == "Тёмная":
    st.markdown(DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

st.title(
    "Визуализация медицинских параметров по административно-территориальным "
    "образованиям Калининградской области"
)

st.markdown(
    """
1. Введите данные по каждому округу или загрузите таблицу (CSV/Excel).  
2. Выберите показатель, палитру и режим фильтра.  
3. Укажите, сколько карт (лет) нужно сравнить.  
4. Нажмите **«Сгенерировать карту»**.  
"""
)

base_df = pd.DataFrame({"district_id": DISTRICTS, "value": [0] * len(DISTRICTS)})

# --- Настройки в сайдбаре ---
with st.sidebar:
    # Палитры
    cmap_options = {
        "Оранжевый → красный (OrRd)": "OrRd",
        "Красный (Reds)": "Reds",
        "Синий (Blues)": "Blues",
        "Зелёный (Greens)": "Greens",
        "Фиолетовый (Purples)": "Purples",
        "Viridis": "viridis",
    }
    cmap_label = st.selectbox(
        "Цветовая палитра",
        list(cmap_options.keys()),
        index=0,
    )
    cmap_name = cmap_options[cmap_label]

    filter_label = st.radio(
        "Фильтр округов",
        [
            "Показывать все округа",
            "5 с максимальными значениями",
            "5 с минимальными значениями",
        ],
        index=0,
    )
    if filter_label == "Показывать все округа":
        filter_mode = "all"
    elif filter_label == "5 с максимальными значениями":
        filter_mode = "top5"
    else:
        filter_mode = "bottom5"

    show_labels = st.checkbox(
        "Показывать подписи и текст на карте",
        value=True,
    )

    num_maps = st.selectbox(
        "Количество карт (лет)",
        [1, 2, 3, 4],
        index=1,
    )

# --- Данные для каждой карты ---

titles = []
edited_dfs = []

for i in range(num_maps):
    st.subheader(f"Данные для карты {i + 1}")

    period_title = st.text_input(
        f"Название периода для карты {i + 1}",
        value=f"Год {i + 1}",
        key=f"period_title_{i}",
    )

    uploaded_file = st.file_uploader(
        f"Загрузить таблицу для карты {i + 1} (CSV или Excel; столбцы: district_id, value + другие)",
        type=["csv", "xlsx", "xls"],
        key=f"uploader_{i}",
    )

    df = base_df.copy()
    metric_name = "value"

    if uploaded_file is not None:
        try:
            filename = uploaded_file.name.lower()
            if filename.endswith(".csv"):
                src_df = pd.read_csv(uploaded_file)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                src_df = pd.read_excel(uploaded_file)
            else:
                src_df = None
                st.error("Формат файла не поддерживается. Загрузите CSV или Excel.")

            if src_df is not None:
                if "district_id" not in src_df.columns:
                    st.error("В таблице должен быть столбец district_id.")
                else:
                    numeric_cols = [
                        c for c in src_df.columns
                        if c != "district_id" and pd.api.types.is_numeric_dtype(src_df[c])
                    ]
                    if not numeric_cols:
                        st.error("Нет числовых столбцов (кроме district_id).")
                    else:
                        metric_name = st.selectbox(
                            f"Показатель для карты {i + 1} (столбец)",
                            numeric_cols,
                            key=f"metric_{i}",
                        )
                        merged = df[["district_id"]].merge(
                            src_df[["district_id", metric_name]],
                            on="district_id",
                            how="left",
                        )
                        df["value"] = merged[metric_name].fillna(0)
        except Exception as e:
            st.error(f"Ошибка при чтении файла для карты {i + 1}: {e}")

    st.write("Можно отредактировать значения вручную:")
    edited_df = st.data_editor(
        df,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "district_id": st.column_config.TextColumn("Округ (id)", disabled=True),
            "value": st.column_config.NumberColumn("Значение"),
        },
        key=f"editor_{i}",
    )

    titles.append(period_title)
    edited_dfs.append(edited_df)

# --- Генерация всех карт и ZIP ---
if st.button("Сгенерировать карту"):
    value_dicts = [
        dict(zip(ed["district_id"], ed["value"]))
        for ed in edited_dfs
    ]

    all_svgs = []

    if num_maps > 1:
        cols = st.columns(num_maps)
    else:
        cols = [None]

    for i in range(num_maps):
        values = value_dicts[i]
        period_title = titles[i] or f"Карта {i + 1}"

        map_title = f"Карта: {period_title}"
        legend_title = f"Диапазон ({period_title})"
        download_name = f"kaliningrad_{i + 1}.svg"

        if num_maps > 1:
            with cols[i]:
                svg_bytes = render_map(
                    values,
                    cmap_name,
                    filter_mode,
                    show_labels,
                    map_title,
                    legend_title,
                    download_name,
                )
        else:
            svg_bytes = render_map(
                values,
                cmap_name,
                filter_mode,
                show_labels,
                map_title,
                legend_title,
                download_name,
            )

        all_svgs.append((download_name, svg_bytes))

    if all_svgs:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, data in all_svgs:
                zf.writestr(fname, data)
        zip_buffer.seek(0)

        st.subheader("Скачать все карты одним архивом")
        st.download_button(
            label="ZIP (все карты)",
            data=zip_buffer,
            file_name="kaliningrad_maps.zip",
            mime="application/zip",
        )
