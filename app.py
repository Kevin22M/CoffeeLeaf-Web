import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps

# ==========================================
# CONFIGURACIÓN
# ==========================================

IMG_SIZE = (224, 224)

ENFERMEDADES = ["Roya", "Coco", "Minador"]

# Umbrales seleccionados durante la validación
UMBRALES = np.array([0.17, 0.64, 0.33], dtype=np.float32)

RUTA_MODELO = "modelo/CoffeeLeaf_EfficientNetB0_Final.keras"

st.set_page_config(
    page_title="CoffeeLeaf - Detector de enfermedades",
    page_icon="🍃",
    layout="centered"
)

# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model(RUTA_MODELO)
    return modelo

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

def predecir(modelo, imagen):

    entrada = preprocesar(imagen)

    # Obtener probabilidades independientes
    probabilidades = modelo.predict(entrada, verbose=0)[0]

    # Aplicar umbral individual a cada enfermedad
    detectadas = probabilidades >= UMBRALES

    return probabilidades, detectadas

# ==========================================
# INTERFAZ
# ==========================================

st.title("🍃 CoffeeLeaf")
st.subheader("Detector de enfermedades del café")

st.write(
    "Sube una fotografía de una hoja de café para identificar "
    "la presencia de Roya, Coco y Minador mediante inteligencia artificial."
)

st.info(
    "El modelo puede identificar una o varias enfermedades "
    "en una misma fotografía."
)

# Cargar modelo
try:
    modelo = cargar_modelo()

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

        st.image(
            imagen,
            caption="Imagen seleccionada",
            width=350
        )

        if st.button("Analizar imagen", type="primary"):

            with st.spinner("Analizando la imagen..."):

                probabilidades, detectadas = predecir(
                    modelo,
                    imagen
                )

            st.divider()
            st.subheader("Resultados del análisis")

            cantidad = int(np.sum(detectadas))

            if cantidad > 0:

                st.success(
                    f"Se detectaron {cantidad} "
                    f"enfermedad(es) en la hoja."
                )

                for i, enfermedad in enumerate(ENFERMEDADES):

                    if detectadas[i]:

                        st.markdown(
                            f"### 🟢 {enfermedad}"
                        )

                        st.metric(
                            "Probabilidad",
                            f"{probabilidades[i] * 100:.2f} %"
                        )

            else:

                st.warning(
                    "No se detectaron las enfermedades evaluadas "
                    "en esta imagen."
                )

            # Mostrar todas las probabilidades
            st.divider()
            st.subheader("Probabilidades por enfermedad")

            for i, enfermedad in enumerate(ENFERMEDADES):

                probabilidad = float(probabilidades[i])
                umbral = float(UMBRALES[i])

                st.write(f"**{enfermedad}**")

                st.progress(probabilidad)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Probabilidad",
                        f"{probabilidad * 100:.2f} %"
                    )

                with col2:
                    st.metric(
                        "Umbral",
                        f"{umbral * 100:.0f} %"
                    )

                if detectadas[i]:
                    st.caption("🟢 Enfermedad detectada")
                else:
                    st.caption("⚪ Enfermedad no detectada")

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
