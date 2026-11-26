import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps

# ==========================================
# CONFIGURACIÓN
# ==========================================

IMG_SIZE = (384, 384)

ENFERMEDADES = ["Roya", "Coco", "Minador"]

# Umbrales seleccionados durante la validación
UMBRALES = np.array([0.57, 0.65, 0.43], dtype=np.float32)

RUTA_MODELO = "modelo/CoffeeLeaf_EfficientNetB0.keras"

st.set_page_config(
    page_title="CoffeeLeaf - Detector de enfermedades",
    page_icon="🍃",
    layout="wide"
)

# ==========================================
# ESTILOS
# ==========================================

st.markdown(
    """
    <style>
    .block-container {max-width: 1050px; padding-top: 3.5rem;}

    .encabezado {
        background: linear-gradient(135deg, #14452f, #2e7d4f);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 20px;
    }
    .encabezado h1 {color: #ffffff; margin: 0; padding: 0; font-size: 2.2rem;}
    .encabezado h3 {color: #d8f3dc; margin: 4px 0 12px 0; padding: 0;
                    font-size: 1.15rem; font-weight: 500;}
    .encabezado p {color: #ffffff; margin: 0;}

    .tarjeta {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-left: 6px solid #2e7d4f;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .tarjeta.detectada {
        border-left-color: #d9480f;
        background: rgba(217, 72, 15, 0.09);
    }
    .fila {display: flex; justify-content: space-between; align-items: center;
           gap: 8px; flex-wrap: wrap;}
    .nombre {font-size: 1.15rem; font-weight: 700;}
    .porcentaje {font-size: 1.5rem; font-weight: 700;}
    .barra {
        position: relative;
        height: 10px;
        border-radius: 6px;
        background: rgba(128, 128, 128, 0.25);
        margin: 10px 0 8px 0;
    }
    .relleno {height: 100%; border-radius: 6px; background: #2e7d4f;}
    .detectada .relleno {background: #d9480f;}
    .marca {
        position: absolute;
        top: -4px;
        width: 3px;
        height: 18px;
        background: currentColor;
    }
    .detalle {font-size: 0.85rem; opacity: 0.8;}

    [data-testid="stBaseButton-primary"] {
        background: #2e7d4f;
        border-color: #2e7d4f;
        color: #ffffff;
    }
    [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stBaseButton-primary"]:focus,
    [data-testid="stBaseButton-primary"]:active {
        background: #14452f !important;
        border-color: #14452f !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model(RUTA_MODELO)
    umbrales = np.load(UMBRALES)
    return modelo, umbrales

# ==========================================
# PREPROCESAMIENTO
# ==========================================

def preprocesar(imagen):

    # Corregir orientación y convertir a RGB
    imagen = ImageOps.exif_transpose(imagen).convert("RGB")

    # Convertir a float32
    arreglo = np.asarray(imagen, dtype=np.float32)

    # Redimensionar conservando proporciones,
    # igual que tf.image.resize_with_pad del entrenamiento
    arreglo = tf.image.resize_with_pad(
        arreglo,
        IMG_SIZE[0],
        IMG_SIZE[1],
        antialias=True
    ).numpy()

    # Agregar dimensión del lote
    return np.expand_dims(arreglo, axis=0)

# ==========================================
# REALIZAR PREDICCIÓN
# ==========================================

def predecir(modelo, umbrales, imagen):

    entrada = preprocesar(imagen)

    # Obtener probabilidades independientes
    probabilidades = modelo.predict(entrada, verbose=0)[0]

    # Aplicar umbral individual a cada enfermedad
    detectadas = probabilidades >= umbrales

    return probabilidades, detectadas

# ==========================================
# TARJETA DE RESULTADO
# ==========================================

def mostrar_tarjeta(enfermedad, probabilidad, umbral, detectada):

    clase = "tarjeta detectada" if detectada else "tarjeta"
    estado = (
        "🔴 Enfermedad detectada" if detectada
        else "⚪ Enfermedad no detectada"
    )

    st.markdown(
        f"""
<div class="{clase}">
<div class="fila">
<span class="nombre">{enfermedad}</span>
<span class="porcentaje">{probabilidad * 100:.2f} %</span>
</div>
<div class="barra">
<div class="relleno" style="width:{probabilidad * 100:.2f}%"></div>
<div class="marca" style="left:{umbral * 100:.2f}%"></div>
</div>
<div class="fila detalle">
<span>{estado}</span>
<span>Umbral: {umbral * 100:.0f} %</span>
</div>
</div>
""",
        unsafe_allow_html=True
    )

# ==========================================
# INTERFAZ
# ==========================================

st.markdown(
    """
<div class="encabezado">
<h1>🍃 CoffeeLeaf</h1>
<h3>Detector de enfermedades del café</h3>
<p>Sube una fotografía de una hoja de café para identificar
la presencia de Roya, Coco y Minador mediante inteligencia artificial.</p>
</div>
""",
    unsafe_allow_html=True
)

st.info(
    "El modelo puede identificar una o varias enfermedades "
    "en una misma fotografía."
)

# Cargar modelo y umbrales
try:
    modelo, umbrales = cargar_modelo()

except Exception as e:
    st.error(f"No fue posible cargar el modelo: {e}")
    st.stop()

# ==========================================
# CARGAR IMAGEN
# ==========================================

archivo = st.file_uploader(
    "Selecciona una fotografía de una hoja de café",
    type=["jpg", "jpeg", "png"]
)

if archivo is not None:

    try:
        imagen = Image.open(archivo)

        columna_imagen, columna_resultados = st.columns(2, gap="large")

        with columna_imagen:

            st.image(
                ImageOps.exif_transpose(imagen),
                caption="Imagen seleccionada",
                width="stretch"
            )

            analizar = st.button(
                "Analizar imagen",
                type="primary",
                width="stretch"
            )

        with columna_resultados:

            if analizar:

                with st.spinner("Analizando la imagen..."):

                    probabilidades, detectadas = predecir(
                        modelo,
                        umbrales,
                        imagen
                    )

                st.subheader("Resultados del análisis")

                cantidad = int(np.sum(detectadas))

                if cantidad > 0:

                    st.error(
                        f"Se detectaron {cantidad} "
                        f"enfermedad(es) en la hoja."
                    )

                else:

                    st.success(
                        "No se detectaron las enfermedades evaluadas "
                        "en esta imagen."
                    )

                # Mostrar todas las probabilidades
                st.markdown("#### Probabilidades por enfermedad")

                for i, enfermedad in enumerate(ENFERMEDADES):

                    mostrar_tarjeta(
                        enfermedad,
                        float(probabilidades[i]),
                        float(umbrales[i]),
                        bool(detectadas[i])
                    )

            else:

                st.caption(
                    "Pulsa «Analizar imagen» para ver los resultados."
                )

    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")

# ==========================================
# INFORMACIÓN
# ==========================================

st.divider()

st.caption(
    "CoffeeLeaf | Clasificación multietiqueta de enfermedades "
    "en hojas de café mediante EfficientNetB0."
)

st.caption(
    "Las predicciones son orientativas y no sustituyen "
    "la evaluación de un profesional agrónomo."
)
