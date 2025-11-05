"""
Dashboard Interactivo de Análisis de Diabetes
Proyecto de Análisis de Datos - Diabetes Health Indicators

Este dashboard permite explorar de manera interactiva las relaciones entre
variables de salud y diabetes usando Plotly y Streamlit.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Análisis de Diabetes",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Cambiar color de las etiquetas a azul en el sidebar */
    section[data-testid="stSidebar"] label {
        color: #2e7bcf !important;
        font-weight: 500;
    }

    /* Cambiar color de los chips/tags de multiselect a azul */
    section[data-testid="stSidebar"] span[data-baseweb="tag"] {
        background-color: #2e7bcf !important;
        border-color: #2e7bcf !important;
    }

    /* Cambiar color del slider a azul - simplificado */
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #2e7bcf !important;
    }
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] > div {
        background-color: #2e7bcf !important;
    }

    /* Quitar el fondo azul de los números del slider */
    section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
        background-color: transparent !important;
        color: #ffffff !important;
        padding: 0 !important;
    }

    /* Cambiar color de las pestañas activas */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #2e7bcf !important;
        border-bottom-color: #2e7bcf !important;
    }

    /* Cambiar hover de pestañas */
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #2e7bcf !important;
    }
    </style>
""", unsafe_allow_html=True)

# Función para cargar datos con caché
@st.cache_data
def load_data():
    """Carga el dataset limpio y preprocesado de diabetes"""
    # Construir la ruta al archivo de datos limpio
    dir_actual = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_actual, 'data', 'diabetes_clean.csv')

    # Si el archivo limpio no existe, usar el archivo original y procesarlo
    if not os.path.exists(data_path):
        data_path = os.path.join(dir_actual, 'data', 'diabetes_binary_health_indicators_BRFSS2021.csv')
        df = pd.read_csv(data_path)

        # Estandarizar nombres de columnas
        df.columns = (
            df.columns
            .str.replace(r'(?<!^)(?=[A-Z])', '_', regex=True)
            .str.replace(" ", "_")
            .str.lower()
        )

        # Eliminar columnas no deseadas
        columns_to_drop = ['no_docbc_cost', 'income', 'education']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    else:
        # Cargar el archivo limpio directamente
        df = pd.read_csv(data_path)

    return df

# Función para crear mapeo de etiquetas
def get_label_mappings():
    """Retorna diccionarios de mapeo para variables categóricas"""
    return {
        'diabetes_binary': {0: 'Sin Diabetes', 1: 'Con Diabetes'},
        'high_b_p': {0: 'No', 1: 'Sí'},
        'high_chol': {0: 'No', 1: 'Sí'},
        'smoker': {0: 'No', 1: 'Sí'},
        'stroke': {0: 'No', 1: 'Sí'},
        'heart_diseaseor_attack': {0: 'No', 1: 'Sí'},
        'phys_activity': {0: 'No', 1: 'Sí'},
        'fruits': {0: 'No', 1: 'Sí'},
        'veggies': {0: 'No', 1: 'Sí'},
        'hvy_alcohol_consump': {0: 'No', 1: 'Sí'},
        'any_healthcare': {0: 'No', 1: 'Sí'},
        'diff_walk': {0: 'No', 1: 'Sí'},
        'sex': {0: 'Mujer', 1: 'Hombre'},
        'gen_hlth': {1: 'Excelente', 2: 'Muy Buena', 3: 'Buena', 4: 'Regular', 5: 'Mala'},
        'age': {
            1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44',
            6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69',
            11: '70-74', 12: '75-79', 13: '80+'
        }
    }

# Cargar datos
df = load_data()
label_mappings = get_label_mappings()

# ========== HEADER ==========
st.markdown('<h1 class="main-header">Dashboard de Análisis de Diabetes</h1>', unsafe_allow_html=True)
st.markdown("---")

# ========== SIDEBAR ==========
st.sidebar.header("Configuración")
st.sidebar.markdown("### Filtros de Datos")

# Filtro por estado de diabetes
diabetes_filter = st.sidebar.multiselect(
    "Estado de Diabetes",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda x: label_mappings['diabetes_binary'][x]
)

# Filtro por rango de edad
age_filter = st.sidebar.slider(
    "Rango de Edad (categorías)",
    min_value=int(df['age'].min()),
    max_value=int(df['age'].max()),
    value=(int(df['age'].min()), int(df['age'].max()))
)

# Filtro por sexo
sex_filter = st.sidebar.multiselect(
    "Sexo",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda x: label_mappings['sex'][x]
)

# Aplicar filtros
df_filtered = df[
    (df['diabetes_binary'].isin(diabetes_filter)) &
    (df['age'] >= age_filter[0]) &
    (df['age'] <= age_filter[1]) &
    (df['sex'].isin(sex_filter))
]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Registros filtrados:** {len(df_filtered):,} de {len(df):,}")

# ========== MÉTRICAS PRINCIPALES ==========
st.header("Resumen General")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total de Registros",
        value=f"{len(df_filtered):,}",
        delta=f"{(len(df_filtered)/len(df)*100):.1f}% del total"
    )

with col2:
    diabetes_pct = (df_filtered['diabetes_binary'].sum() / len(df_filtered) * 100)
    st.metric(
        label="Prevalencia de Diabetes",
        value=f"{diabetes_pct:.1f}%",
        delta=f"{df_filtered['diabetes_binary'].sum():,} casos"
    )

with col3:
    avg_bmi = df_filtered['b_m_i'].mean()
    st.metric(
        label="BMI Promedio",
        value=f"{avg_bmi:.1f}",
        delta="Normal" if 18.5 <= avg_bmi <= 24.9 else "Fuera de rango"
    )

with col4:
    avg_age = df_filtered['age'].mean()
    st.metric(
        label="Edad Promedio (categoría)",
        value=f"{avg_age:.1f}",
        delta=label_mappings['age'].get(int(round(avg_age)), "N/A")
    )

st.markdown("---")

# ========== TABS PARA DIFERENTES ANÁLISIS ==========
tab1, tab2, tab3 = st.tabs([
    "Distribuciones",
    "Comparaciones",
    "Correlaciones"
])

# ========== TAB 1: DISTRIBUCIONES ==========
with tab1:
    st.header("Distribuciones de Variables Clave para la Diabetes")

    st.markdown("""
    Analiza la distribución de variables críticas asociadas con la diabetes:
    - **Variables Continuas**: BMI, Días de problemas de salud mental y física
    - **Variables Categóricas Binarias**: Colesterol alto, Fumador, Alcohol, Presión Alta
    """)

    # Selector de categoría
    var_category = st.radio(
        "Selecciona la categoría de variable:",
        options=['Continuas', 'Categóricas Binarias'],
        horizontal=True
    )

    if var_category == 'Continuas':
        col1, col2 = st.columns(2)

        with col1:
            # Selector de variable continua
            num_var = st.selectbox(
                "Selecciona una variable continua",
                options=['b_m_i', 'ment_hlth', 'phys_hlth', 'age'],
                format_func=lambda x: {'b_m_i': 'BMI', 'ment_hlth': 'Salud Mental (días)',
                                      'phys_hlth': 'Salud Física (días)', 'age': 'Edad (categoría)'}[x]
            )

        with col2:
            # Tipo de gráfico
            chart_type = st.selectbox(
                "Tipo de gráfico",
                options=['Histograma', 'Box Plot', 'Violin Plot', 'Densidad']
            )

        # Diccionario de labels
        var_labels = {
            'b_m_i': 'BMI',
            'ment_hlth': 'Salud Mental (días)',
            'phys_hlth': 'Salud Física (días)',
            'age': 'Edad (categoría)'
        }

        # Crear gráfico según selección
        if chart_type == 'Histograma':
            fig = px.histogram(
                df_filtered,
                x=num_var,
                color='diabetes_binary',
                barmode='overlay',
                title=f'Distribución de {var_labels[num_var]}',
                labels={'diabetes_binary': 'Estado de Diabetes', num_var: var_labels[num_var]},
                color_discrete_map={0: 'skyblue', 1: 'salmon'},
                opacity=0.7
            )
            fig.update_traces(name='Sin Diabetes', selector=dict(marker_color='skyblue'))
            fig.update_traces(name='Con Diabetes', selector=dict(marker_color='salmon'))

        elif chart_type == 'Box Plot':
            fig = px.box(
                df_filtered,
                x='diabetes_binary',
                y=num_var,
                color='diabetes_binary',
                title=f'Box Plot de {var_labels[num_var]}',
                labels={'diabetes_binary': 'Estado de Diabetes', num_var: var_labels[num_var]},
                color_discrete_map={0: 'skyblue', 1: 'salmon'}
            )
            fig.update_xaxes(ticktext=['Sin Diabetes', 'Con Diabetes'], tickvals=[0, 1])

        elif chart_type == 'Violin Plot':
            fig = px.violin(
                df_filtered,
                x='diabetes_binary',
                y=num_var,
                color='diabetes_binary',
                box=True,
                title=f'Violin Plot de {var_labels[num_var]}',
                labels={'diabetes_binary': 'Estado de Diabetes', num_var: var_labels[num_var]},
                color_discrete_map={0: 'skyblue', 1: 'salmon'}
            )
            fig.update_xaxes(ticktext=['Sin Diabetes', 'Con Diabetes'], tickvals=[0, 1])

        else:  # Densidad
            fig = go.Figure()
            for diabetes_status in [0, 1]:
                data = df_filtered[df_filtered['diabetes_binary'] == diabetes_status][num_var]
                label = 'Sin Diabetes' if diabetes_status == 0 else 'Con Diabetes'
                color = 'skyblue' if diabetes_status == 0 else 'salmon'

                fig.add_trace(go.Histogram(
                    x=data,
                    name=label,
                    opacity=0.7,
                    histnorm='probability density',
                    marker_color=color
                ))

            fig.update_layout(
                title=f'Densidad de {var_labels[num_var]}',
                xaxis_title=var_labels[num_var],
                yaxis_title='Densidad',
                barmode='overlay'
            )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Estadísticas descriptivas
        st.subheader("Estadísticas Descriptivas")
        stats = df_filtered.groupby('diabetes_binary')[num_var].describe().round(2)
        stats.index = ['Sin Diabetes', 'Con Diabetes']
        st.dataframe(stats, use_container_width=True)

    else:  # Categóricas Binarias
        st.subheader("Distribuciones de Variables Binarias Clave")

        # Variables binarias de interés
        binary_vars = {
            'high_chol': 'Colesterol Alto',
            'smoker': 'Fumador',
            'hvy_alcohol_consump': 'Consumo Alto de Alcohol',
            'high_b_p': 'Presión Arterial Alta'
        }

        # Crear gráficos para cada variable binaria
        cols = st.columns(2)
        for idx, (var, label) in enumerate(binary_vars.items()):
            with cols[idx % 2]:
                # Tabla de contingencia
                contingency = pd.crosstab(
                    df_filtered[var],
                    df_filtered['diabetes_binary'],
                    normalize='columns'
                ) * 100

                contingency.columns = ['Sin Diabetes', 'Con Diabetes']
                contingency.index = contingency.index.map(
                    lambda x: 'Sí' if x == 1 else 'No'
                )

                # Gráfico de barras
                fig = go.Figure(data=[
                    go.Bar(name='Sin Diabetes', x=contingency.index, y=contingency['Sin Diabetes'], marker_color='skyblue'),
                    go.Bar(name='Con Diabetes', x=contingency.index, y=contingency['Con Diabetes'], marker_color='salmon')
                ])

                fig.update_layout(
                    title=f'{label}',
                    xaxis_title='',
                    yaxis_title='Porcentaje (%)',
                    barmode='group',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

# ========== TAB 2: COMPARACIONES ==========
with tab2:
    st.header("Comparación de Variables entre Grupos")

    st.markdown("""
    Selecciona comparaciones significativas entre variables relacionadas:
    - **BMI y Salud Física**: Relación entre peso corporal y días con problemas físicos
    - **Salud Mental y Salud Física**: Relación entre bienestar mental y físico
    - **Edad y variables de salud**: Cómo cambian las variables de salud con la edad
    """)

    # Selector de tipo de comparación
    comparison_type = st.radio(
        "Tipo de comparación:",
        options=['BMI vs Salud', 'Salud Mental vs Física', 'Edad vs Salud', 'Personalizado'],
        horizontal=True
    )

    # Definir las variables según el tipo de comparación
    if comparison_type == 'BMI vs Salud':
        var_x = 'b_m_i'
        var_y_options = ['phys_hlth', 'ment_hlth']
        var_y = st.selectbox(
            "Variable de salud a comparar:",
            options=var_y_options,
            format_func=lambda x: 'Salud Física (días)' if x == 'phys_hlth' else 'Salud Mental (días)',
            key='var_y_bmi'
        )
    elif comparison_type == 'Salud Mental vs Física':
        var_x = 'ment_hlth'
        var_y = 'phys_hlth'
    elif comparison_type == 'Edad vs Salud':
        var_x = 'age'
        var_y_options = ['b_m_i', 'phys_hlth', 'ment_hlth']
        var_y = st.selectbox(
            "Variable de salud a comparar:",
            options=var_y_options,
            format_func=lambda x: {'b_m_i': 'BMI', 'phys_hlth': 'Salud Física (días)', 'ment_hlth': 'Salud Mental (días)'}[x],
            key='var_y_age'
        )
    else:  # Personalizado
        col1, col2 = st.columns(2)
        with col1:
            var_x = st.selectbox(
                "Variable X",
                options=['b_m_i', 'ment_hlth', 'phys_hlth', 'age'],
                format_func=lambda x: {'b_m_i': 'BMI', 'ment_hlth': 'Salud Mental (días)',
                                      'phys_hlth': 'Salud Física (días)', 'age': 'Edad (categoría)'}[x],
                key='var_x_custom'
            )
        with col2:
            var_y = st.selectbox(
                "Variable Y",
                options=['b_m_i', 'ment_hlth', 'phys_hlth', 'age'],
                format_func=lambda x: {'b_m_i': 'BMI', 'ment_hlth': 'Salud Mental (días)',
                                      'phys_hlth': 'Salud Física (días)', 'age': 'Edad (categoría)'}[x],
                key='var_y_custom',
                index=1
            )

    # Scatter plot
    var_labels = {
        'b_m_i': 'BMI (Índice de Masa Corporal)',
        'ment_hlth': 'Salud Mental (días con problemas)',
        'phys_hlth': 'Salud Física (días con problemas)',
        'age': 'Edad (categoría)'
    }

    fig = px.scatter(
        df_filtered.sample(min(5000, len(df_filtered))),  # Muestra para mejor rendimiento
        x=var_x,
        y=var_y,
        color='diabetes_binary',
        title=f'Relación entre {var_labels[var_x]} y {var_labels[var_y]}',
        labels={'diabetes_binary': 'Estado de Diabetes', var_x: var_labels[var_x], var_y: var_labels[var_y]},
        color_discrete_map={0: 'skyblue', 1: 'salmon'},
        opacity=0.6,
        trendline='ols'
    )

    # Actualizar nombres de las trazas en la leyenda de forma segura
    for trace in fig.data:
        if hasattr(trace, 'legendgroup'):
            if '0' in str(trace.legendgroup):
                trace.name = trace.name.replace('0', 'Sin Diabetes')
            elif '1' in str(trace.legendgroup):
                trace.name = trace.name.replace('1', 'Con Diabetes')

    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Gráfico de barras comparativo
    st.subheader("Comparación de Promedios - Variables Clave")

    variables = ['b_m_i', 'ment_hlth', 'phys_hlth']
    var_names = ['BMI', 'Salud Mental (días)', 'Salud Física (días)']
    means_no_diabetes = [df_filtered[df_filtered['diabetes_binary'] == 0][var].mean() for var in variables]
    means_diabetes = [df_filtered[df_filtered['diabetes_binary'] == 1][var].mean() for var in variables]

    fig = go.Figure(data=[
        go.Bar(name='Sin Diabetes', x=var_names, y=means_no_diabetes, marker_color='skyblue'),
        go.Bar(name='Con Diabetes', x=var_names, y=means_diabetes, marker_color='salmon')
    ])

    fig.update_layout(
        barmode='group',
        title='Comparación de Promedios: BMI y Días con Problemas de Salud',
        yaxis_title='Valor Promedio',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ========== TAB 3: CORRELACIONES ==========
with tab3:
    st.header("Análisis de Correlaciones con Diabetes")

    # Crear tres pestañas dentro de TAB 3
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "Condiciones Prevalentes en Diabetes",
        "Correlación General",
        "Análisis en Presencia de Diabetes"
    ])

    # ===== SUB-TAB 1: CONDICIONES PREVALENTES =====
    with sub_tab1:
        st.subheader("Condiciones y Aspectos más Frecuentes en Personas con Diabetes")
        st.markdown("""
        Esta sección muestra qué condiciones de salud y comportamientos son más comunes
        en personas **con diabetes** comparadas con personas **sin diabetes**.
        """)

        # Filtrar solo casos con diabetes
        df_diabetes = df_filtered[df_filtered['diabetes_binary'] == 1]
        df_no_diabetes = df_filtered[df_filtered['diabetes_binary'] == 0]

        if len(df_diabetes) > 0 and len(df_no_diabetes) > 0:
            # Diccionario de etiquetas
            var_labels_full = {
                'diabetes_binary': 'Diabetes',
                'high_b_p': 'Presión Alta',
                'high_chol': 'Colesterol Alto',
                'chol_check': 'Colesterol Verificado',
                'b_m_i': 'BMI',
                'smoker': 'Fumador',
                'stroke': 'Derrame',
                'heart_diseaseor_attack': 'Enfermedad Cardíaca',
                'phys_activity': 'Actividad Física',
                'fruits': 'Consumo de Frutas',
                'veggies': 'Consumo de Verduras',
                'hvy_alcohol_consump': 'Consumo Alto Alcohol',
                'any_healthcare': 'Cobertura de Salud',
                'gen_hlth': 'Salud General',
                'ment_hlth': 'Salud Mental (días)',
                'phys_hlth': 'Salud Física (días)',
                'diff_walk': 'Dificultad para Caminar',
                'sex': 'Sexo',
                'age': 'Edad'
            }

            # Variables categóricas binarias a analizar
            binary_vars = [
                'high_b_p', 'high_chol', 'chol_check', 'smoker', 'stroke',
                'heart_diseaseor_attack', 'phys_activity', 'fruits', 'veggies',
                'hvy_alcohol_consump', 'any_healthcare', 'diff_walk', 'sex'
            ]

            # 1. Gráfico de prevalencia de condiciones binarias
            st.markdown("### 1. Prevalencia de Condiciones de Salud y Comportamientos")

            prevalence_data = []
            for var in binary_vars:
                if var in df_diabetes.columns:
                    prev_diabetes = (df_diabetes[var] == 1).sum() / len(df_diabetes) * 100
                    prev_no_diabetes = (df_no_diabetes[var] == 1).sum() / len(df_no_diabetes) * 100
                    difference = prev_diabetes - prev_no_diabetes

                    prevalence_data.append({
                        'Condición': var_labels_full.get(var, var),
                        'Con Diabetes (%)': prev_diabetes,
                        'Sin Diabetes (%)': prev_no_diabetes,
                        'Diferencia (%)': difference
                    })

            prev_df = pd.DataFrame(prevalence_data).sort_values('Con Diabetes (%)', ascending=False)

            # Gráfico de barras comparativo
            fig = go.Figure()

            fig.add_trace(go.Bar(
                name='Con Diabetes',
                x=prev_df['Condición'],
                y=prev_df['Con Diabetes (%)'],
                marker_color='#e74c3c',
                text=prev_df['Con Diabetes (%)'].round(1),
                textposition='auto',
                texttemplate='%{text:.1f}%'
            ))

            fig.add_trace(go.Bar(
                name='Sin Diabetes',
                x=prev_df['Condición'],
                y=prev_df['Sin Diabetes (%)'],
                marker_color='#3498db',
                text=prev_df['Sin Diabetes (%)'].round(1),
                textposition='auto',
                texttemplate='%{text:.1f}%'
            ))

            fig.update_layout(
                title='Prevalencia de Condiciones: Con Diabetes vs Sin Diabetes',
                xaxis_title='Condición de Salud / Comportamiento',
                yaxis_title='Prevalencia (%)',
                barmode='group',
                height=600,
                xaxis_tickangle=-45,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

            # 2. Gráfico de diferencia (más común en diabetes)
            st.markdown("### 2. Diferencia de Prevalencia (Con Diabetes - Sin Diabetes)")

            fig2 = go.Figure()

            colors = ['#e74c3c' if x > 0 else '#27ae60' for x in prev_df['Diferencia (%)']]

            fig2.add_trace(go.Bar(
                x=prev_df['Diferencia (%)'],
                y=prev_df['Condición'],
                orientation='h',
                marker_color=colors,
                text=prev_df['Diferencia (%)'].round(1),
                textposition='auto',
                texttemplate='%{text:+.1f}%'
            ))

            fig2.update_layout(
                title='Diferencia de Prevalencia entre Grupos (positivo = más común en diabetes)',
                xaxis_title='Diferencia en Puntos Porcentuales',
                yaxis_title='Condición',
                height=600,
                showlegend=False
            )

            st.plotly_chart(fig2, use_container_width=True)

            # 3. Tabla con estadísticas detalladas
            st.markdown("### 3. Tabla Detallada de Prevalencias")

            prev_df_display = prev_df.copy()
            prev_df_display['Con Diabetes (%)'] = prev_df_display['Con Diabetes (%)'].round(2)
            prev_df_display['Sin Diabetes (%)'] = prev_df_display['Sin Diabetes (%)'].round(2)
            prev_df_display['Diferencia (%)'] = prev_df_display['Diferencia (%)'].round(2)

            st.dataframe(prev_df_display, use_container_width=True, hide_index=True)

            # 4. Top 5 condiciones más prevalentes en diabetes
            st.markdown("### 4. Top 5 Condiciones Más Comunes en Personas con Diabetes")

            top5 = prev_df.nlargest(5, 'Con Diabetes (%)')

            col1, col2 = st.columns([3, 2])

            with col1:
                fig3 = px.bar(
                    top5,
                    x='Con Diabetes (%)',
                    y='Condición',
                    orientation='h',
                    title='Top 5 Aspectos Más Prevalentes',
                    text='Con Diabetes (%)',
                    color='Con Diabetes (%)',
                    color_continuous_scale='Reds'
                )
                fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig3.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

            with col2:
                st.markdown("#### Porcentajes:")
                for idx, row in top5.iterrows():
                    st.metric(
                        label=row['Condición'],
                        value=f"{row['Con Diabetes (%)']:.1f}%",
                        delta=f"+{row['Diferencia (%)']:.1f}% vs sin diabetes"
                    )

            # 5. Análisis de variables continuas
            st.markdown("### 5. Comparación de Variables Continuas")

            continuous_vars = ['b_m_i', 'phys_hlth', 'ment_hlth', 'age']

            continuous_data = []
            for var in continuous_vars:
                if var in df_diabetes.columns:
                    mean_diabetes = df_diabetes[var].mean()
                    mean_no_diabetes = df_no_diabetes[var].mean()

                    continuous_data.append({
                        'Variable': var_labels_full.get(var, var),
                        'Promedio Con Diabetes': mean_diabetes,
                        'Promedio Sin Diabetes': mean_no_diabetes,
                        'Diferencia': mean_diabetes - mean_no_diabetes
                    })

            cont_df = pd.DataFrame(continuous_data)

            fig4 = go.Figure()

            fig4.add_trace(go.Bar(
                name='Con Diabetes',
                x=cont_df['Variable'],
                y=cont_df['Promedio Con Diabetes'],
                marker_color='#e74c3c',
                text=cont_df['Promedio Con Diabetes'].round(2),
                textposition='auto'
            ))

            fig4.add_trace(go.Bar(
                name='Sin Diabetes',
                x=cont_df['Variable'],
                y=cont_df['Promedio Sin Diabetes'],
                marker_color='#3498db',
                text=cont_df['Promedio Sin Diabetes'].round(2),
                textposition='auto'
            ))

            fig4.update_layout(
                title='Comparación de Promedios: Variables Continuas',
                xaxis_title='Variable',
                yaxis_title='Valor Promedio',
                barmode='group',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig4, use_container_width=True)

            # Tabla de variables continuas
            cont_df_display = cont_df.copy()
            cont_df_display['Promedio Con Diabetes'] = cont_df_display['Promedio Con Diabetes'].round(2)
            cont_df_display['Promedio Sin Diabetes'] = cont_df_display['Promedio Sin Diabetes'].round(2)
            cont_df_display['Diferencia'] = cont_df_display['Diferencia'].round(2)

            st.dataframe(cont_df_display, use_container_width=True, hide_index=True)

            # 6. Correlaciones con Diabetes (diabetes_binary = 1)
            st.markdown("### 6. Correlaciones de Variables con la Presencia de Diabetes")
            st.markdown("""
            Coeficiente de correlación de Pearson entre cada variable y la presencia de diabetes.
            - **Valores positivos**: la variable aumenta con la presencia de diabetes
            - **Valores negativos**: la variable disminuye con la presencia de diabetes
            """)

            # Calcular correlaciones
            numeric_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if 'diabetes_binary' in numeric_cols:
                correlations_with_diabetes = df_filtered[numeric_cols].corr()['diabetes_binary'].drop('diabetes_binary').sort_values(ascending=False)

                # Crear labels descriptivos
                corr_labels = [var_labels_full.get(var, var) for var in correlations_with_diabetes.index]
                corr_values = correlations_with_diabetes.values

                fig5 = go.Figure(go.Bar(
                    x=corr_values,
                    y=corr_labels,
                    orientation='h',
                    marker=dict(
                        color=corr_values,
                        colorscale='RdBu_r',
                        showscale=True,
                        cmin=-1,
                        cmax=1,
                        colorbar=dict(title="Correlación")
                    ),
                    text=[f'{val:.3f}' for val in corr_values],
                    textposition='auto'
                ))

                fig5.update_layout(
                    title='Correlación de Variables con Presencia de Diabetes (diabetes_binary = 1)',
                    xaxis_title='Coeficiente de Correlación de Pearson',
                    yaxis_title='Variable',
                    height=700,
                    showlegend=False
                )
                st.plotly_chart(fig5, use_container_width=True)

                # Tabla de correlaciones
                corr_table = pd.DataFrame({
                    'Variable': corr_labels,
                    'Correlación con Diabetes': [f'{val:.4f}' for val in corr_values],
                    'Magnitud': ['Fuerte' if abs(val) > 0.5 else 'Moderada' if abs(val) > 0.3 else 'Débil' for val in corr_values],
                    'Dirección': ['Positiva' if val > 0 else 'Negativa' for val in corr_values]
                })

                st.dataframe(corr_table, use_container_width=True, hide_index=True)

        else:
            st.warning("No hay suficientes datos para realizar la comparación. Ajusta los filtros.")

    # ===== SUB-TAB 2: CORRELACIONES GENERALES =====

    with sub_tab2:
        st.markdown("""
        Se analizan las correlaciones entre TODAS las variables disponibles y la presencia de diabetes.
        Esta vista completa permite identificar los factores más y menos asociados con la diabetes.
        """)

        # Obtener todas las columnas numéricas (excluyendo diabetes_binary que es la variable objetivo)
        all_numeric_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'diabetes_binary' in all_numeric_cols:
            all_numeric_cols.remove('diabetes_binary')
        all_numeric_cols.insert(0, 'diabetes_binary')  # Poner diabetes_binary al inicio

        # Matriz de correlación con TODAS las variables
        correlation_matrix = df_filtered[all_numeric_cols].corr().round(3)

        # Crear etiquetas descriptivas para las variables
        var_labels_full = {
            'diabetes_binary': 'Diabetes',
            'high_b_p': 'Presión Alta',
            'high_chol': 'Colesterol Alto',
            'chol_check': 'Colesterol Verificado',
            'b_m_i': 'BMI',
            'smoker': 'Fumador',
            'stroke': 'Derrame',
            'heart_diseaseor_attack': 'Enfermedad Cardíaca',
            'phys_activity': 'Actividad Física',
            'fruits': 'Consumo de Frutas',
            'veggies': 'Consumo de Verduras',
            'hvy_alcohol_consump': 'Consumo Alto Alcohol',
            'any_healthcare': 'Cobertura de Salud',
            'gen_hlth': 'Salud General',
            'ment_hlth': 'Salud Mental (días)',
            'phys_hlth': 'Salud Física (días)',
            'diff_walk': 'Dificultad para Caminar',
            'sex': 'Sexo',
            'age': 'Edad'
        }

        # Renombrar las columnas para mejor visualización
        correlation_matrix_display = correlation_matrix.copy()
        correlation_matrix_display.index = [var_labels_full.get(col, col) for col in correlation_matrix_display.index]
        correlation_matrix_display.columns = [var_labels_full.get(col, col) for col in correlation_matrix_display.columns]

        # Matriz de correlación completa
        fig = px.imshow(
            correlation_matrix_display,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Matriz de Correlación Completa - Todas las Variables',
            labels=dict(color='Correlación'),
            color_continuous_midpoint=0
        )
        fig.update_layout(height=800, width=1000)
        st.plotly_chart(fig, use_container_width=True)

        # Gráfico de correlaciones con Diabetes
        st.subheader("Correlaciones con Diabetes - Ordenadas por Magnitud")

        correlations_with_diabetes = correlation_matrix['diabetes_binary'].drop('diabetes_binary').sort_values(ascending=False)

        # Crear labels descriptivos
        corr_labels = [var_labels_full.get(var, var) for var in correlations_with_diabetes.index]

        fig = go.Figure(go.Bar(
            x=correlations_with_diabetes.values,
            y=corr_labels,
            orientation='h',
            marker=dict(
                color=correlations_with_diabetes.values,
                colorscale='RdBu_r',
                showscale=True,
                cmin=-1,
                cmax=1
            ),
            text=correlations_with_diabetes.values.round(3),
            textposition='auto'
        ))

        fig.update_layout(
            title='Coeficiente de Correlación de cada Variable con Diabetes',
            xaxis_title='Coeficiente de Correlación de Pearson',
            yaxis_title='Variable',
            height=600,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabla de correlaciones
        st.subheader("Tabla de Correlaciones Detallada")
        corr_table = pd.DataFrame({
            'Variable': corr_labels,
            'Correlación': correlations_with_diabetes.values.round(4)
        }).reset_index(drop=True)

        st.dataframe(corr_table, use_container_width=True)

    with sub_tab3:
        st.markdown("""
        Análisis de correlaciones **ÚNICAMENTE para casos CON diabetes (diabetes_binary = 1)**.

        Esto permite entender cómo las variables se relacionan entre sí en personas que YA tienen diabetes.
        - Correlación positiva alta → Factores que co-ocurren con diabetes
        - Correlación negativa alta → Factores que varían inversamente en diabetes

        **Objetivo:** Identificar patrones de comorbilidad y factores de riesgo asociados.
        """)

        # Filtrar SOLO casos con diabetes
        df_with_diabetes = df_filtered[df_filtered['diabetes_binary'] == 1]

        st.info(f"🔍 Analizando {len(df_with_diabetes):,} casos CON diabetes de {len(df_filtered):,} totales ({len(df_with_diabetes)/len(df_filtered)*100:.1f}%)")

        if len(df_with_diabetes) > 0:
            # Obtener todas las columnas numéricas
            all_numeric_cols_diabetes = df_with_diabetes.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # Matriz de correlación SOLO para casos con diabetes
            correlation_diabetes = df_with_diabetes[all_numeric_cols_diabetes].corr().round(3)

            # Renombrar columnas
            correlation_diabetes_display = correlation_diabetes.copy()
            correlation_diabetes_display.index = [var_labels_full.get(col, col) for col in correlation_diabetes_display.index]
            correlation_diabetes_display.columns = [var_labels_full.get(col, col) for col in correlation_diabetes_display.columns]

            # Sección 1: Matriz de correlación completa para casos con diabetes
            st.subheader("1. Matriz de Correlación - Cuando Diabetes está Presente")
            st.markdown("Correlaciones entre TODAS las variables considerando solo personas CON diabetes")

            fig = px.imshow(
                correlation_diabetes_display,
                text_auto=True,
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title='Matriz de Correlación Completa (Diabetes Presente)',
                labels=dict(color='Correlación'),
                color_continuous_midpoint=0,
                zmin=-1,
                zmax=1
            )
            fig.update_layout(height=800, width=1000)
            st.plotly_chart(fig, use_container_width=True)

            # Sección 2: Factores de riesgo clave
            st.subheader("2. Incidencia de Factores de Riesgo Clave")
            st.markdown("Cómo se relacionan los factores de riesgo más importantes en personas CON diabetes")

            # Variables de riesgo principales
            risk_factors = ['high_chol', 'smoker', 'hvy_alcohol_consump', 'high_b_p', 'b_m_i', 'phys_hlth', 'ment_hlth']
            risk_factors_available = [var for var in risk_factors if var in df_with_diabetes.columns]

            if risk_factors_available:
                risk_correlation = df_with_diabetes[risk_factors_available].corr().round(3)

                # Renombrar
                risk_labels = [var_labels_full.get(var, var) for var in risk_factors_available]
                risk_correlation.index = risk_labels
                risk_correlation.columns = risk_labels

                fig = px.imshow(
                    risk_correlation,
                    text_auto=True,
                    aspect='auto',
                    color_continuous_scale='RdYlBu_r',
                    title='Co-ocurrencia de Factores de Riesgo (Diabetes Presente)',
                    labels=dict(color='Correlación'),
                    color_continuous_midpoint=0,
                    zmin=-1,
                    zmax=1
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

            # Sección 3: Ranking de correlaciones más fuertes
            st.subheader("3. Correlaciones Más Fuertes - Variables Principales")
            st.markdown("Pares de variables que tienen mayor relación cuando la diabetes está presente")

            # Crear lista de correlaciones (sin diagonal)
            corr_pairs = []
            for i in range(len(correlation_diabetes.columns)):
                for j in range(i+1, len(correlation_diabetes.columns)):
                    var1 = correlation_diabetes.columns[i]
                    var2 = correlation_diabetes.columns[j]
                    corr_val = correlation_diabetes.iloc[i, j]
                    corr_pairs.append({
                        'Variable 1': var_labels_full.get(var1, var1),
                        'Variable 2': var_labels_full.get(var2, var2),
                        'Correlación': corr_val,
                        'Abs_Corr': abs(corr_val)
                    })

            corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Abs_Corr', ascending=False).head(15)

            fig = px.bar(
                corr_pairs_df,
                x='Correlación',
                y=[f"{row['Variable 1']}\n↔\n{row['Variable 2']}" for _, row in corr_pairs_df.iterrows()],
                orientation='h',
                color='Correlación',
                color_continuous_scale='RdBu_r',
                title='Top 15 - Correlaciones más Fuertes (Diabetes Presente)',
                labels={'Correlación': 'Coeficiente de Correlación'}
            )
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Sección 4: Estadísticas descriptivas en presencia de diabetes
            st.subheader("4. Estadísticas Descriptivas de Variables Clave (Con Diabetes)")
            st.markdown("Cómo se distribuyen las variables principales en personas CON diabetes")

            key_stats_data = []
            for var in ['b_m_i', 'high_chol', 'smoker', 'hvy_alcohol_consump', 'high_b_p', 'phys_hlth', 'ment_hlth', 'age', 'gen_hlth', 'diff_walk']:
                if var in df_with_diabetes.columns:
                    col_data = df_with_diabetes[var]
                    key_stats_data.append({
                        'Variable': var_labels_full.get(var, var),
                        'Promedio': f'{col_data.mean():.2f}',
                        'Desv. Est.': f'{col_data.std():.2f}',
                        'Mín': f'{col_data.min():.0f}',
                        'Máx': f'{col_data.max():.0f}',
                        'Mediana': f'{col_data.median():.2f}'
                    })

            stats_df_diabetes = pd.DataFrame(key_stats_data)
            st.dataframe(stats_df_diabetes, use_container_width=True)

            # Sección 5: Comparación de correlaciones (General vs Con Diabetes)
            st.subheader("5. Comparativa: Correlación General vs Con Diabetes Presente")
            st.markdown("Cómo cambian las correlaciones cuando pasamos de análisis general a solo casos con diabetes")

            # Obtener variables comunes
            common_vars = list(set(all_numeric_cols) & set(all_numeric_cols_diabetes))
            common_vars = [v for v in common_vars if v != 'diabetes_binary']

            if len(common_vars) > 0:
                # Crear comparación
                comparison_data = []
                for var in common_vars:
                    if var in correlation_matrix.index and var in correlation_diabetes.index:
                        corr_general = correlation_matrix.loc['diabetes_binary', var] if 'diabetes_binary' in correlation_matrix.index else None

                        # Para diabetes presente, correlacionar con otras variables
                        corr_with_first = correlation_diabetes.loc[var, common_vars[0]] if common_vars[0] != var else 0

                        if corr_general is not None:
                            comparison_data.append({
                                'Variable': var_labels_full.get(var, var),
                                'Correlación con Diabetes (General)': f'{corr_general:.3f}',
                                'Correlación Media (Con Diabetes)': f'{correlation_diabetes.loc[var].mean():.3f}'
                            })

                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

        else:
            st.warning("No hay casos con diabetes en los filtros seleccionados")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p><strong>Dashboard de Análisis de Diabetes</strong></p>
        <p>Proyecto de Ciencia de Datos</p>
        <p>Datos: BRFSS 2021 - Behavioral Risk Factor Surveillance System</p>
    </div>
""", unsafe_allow_html=True)
