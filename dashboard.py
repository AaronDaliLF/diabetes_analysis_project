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
    """Carga y preprocesa el dataset de diabetes"""
    # Construir la ruta al archivo de datos
    dir_actual = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_actual, 'data', 'diabetes_binary_health_indicators_BRFSS2021.csv')

    # Cargar datos
    df = pd.read_csv(data_path)

    # Estandarizar nombres de columnas
    df.columns = (
        df.columns
        .str.replace(r'(?<!^)(?=[A-Z])', '_', regex=True)
        .str.replace(" ", "_")
        .str.lower()
    )

    # Eliminar columnas no deseadas si existen
    columns_to_drop = ['no_docbc_cost', 'income', 'education']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

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
    st.header("Distribuciones de Variables Numéricas")

    col1, col2 = st.columns(2)

    with col1:
        # Selector de variable
        num_var = st.selectbox(
            "Selecciona una variable numérica",
            options=['b_m_i', 'ment_hlth', 'phys_hlth', 'age'],
            format_func=lambda x: x.upper().replace('_', ' ')
        )

    with col2:
        # Tipo de gráfico
        chart_type = st.selectbox(
            "Tipo de gráfico",
            options=['Histograma', 'Box Plot', 'Violin Plot', 'Densidad']
        )

    # Crear gráfico según selección
    if chart_type == 'Histograma':
        fig = px.histogram(
            df_filtered,
            x=num_var,
            color='diabetes_binary',
            barmode='overlay',
            title=f'Distribución de {num_var.upper()}',
            labels={'diabetes_binary': 'Estado de Diabetes', num_var: num_var.upper()},
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
            title=f'Box Plot de {num_var.upper()} por Estado de Diabetes',
            labels={'diabetes_binary': 'Estado de Diabetes', num_var: num_var.upper()},
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
            title=f'Violin Plot de {num_var.upper()} por Estado de Diabetes',
            labels={'diabetes_binary': 'Estado de Diabetes', num_var: num_var.upper()},
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
            title=f'Densidad de {num_var.upper()}',
            xaxis_title=num_var.upper(),
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
    st.header("Análisis de Correlaciones")

    # Matriz de correlación
    numeric_vars = ['b_m_i', 'ment_hlth', 'phys_hlth', 'age', 'diabetes_binary']
    correlation_matrix = df_filtered[numeric_vars].corr().round(3)

    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title='Matriz de Correlación',
        labels=dict(color='Correlación')
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap interactivo
    st.subheader("Correlaciones con Diabetes")
    correlations_with_diabetes = correlation_matrix['diabetes_binary'].drop('diabetes_binary').sort_values(ascending=False)

    fig = go.Figure(go.Bar(
        x=correlations_with_diabetes.values,
        y=[var.upper().replace('_', ' ') for var in correlations_with_diabetes.index],
        orientation='h',
        marker=dict(
            color=correlations_with_diabetes.values,
            colorscale='RdBu_r',
            showscale=True,
            cmin=-1,
            cmax=1
        )
    ))

    fig.update_layout(
        title='Correlación de Variables con Diabetes',
        xaxis_title='Coeficiente de Correlación',
        yaxis_title='Variable',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p><strong>Dashboard de Análisis de Diabetes</strong></p>
        <p>Proyecto de Ciencia de Datos</p>
        <p>Datos: BRFSS 2021 - Behavioral Risk Factor Surveillance System</p>
    </div>
""", unsafe_allow_html=True)
