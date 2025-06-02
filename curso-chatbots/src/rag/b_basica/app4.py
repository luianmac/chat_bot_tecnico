"""
Chatbot de Soporte Técnico Claro - Interfaz Profesional Mejorada
"""

import logging
import time

import streamlit as st

from src.config.parameters import MAX_PAGES
from src.rag.b_basica.nlp_proc import compute_embeddings, response_generator
from src.rag.b_basica.utils import extract_context, retrieve_dataframe, store_dataframe

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CSS personalizado para interfaz profesional mejorada
def load_css():
    st.markdown(
        """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Variables CSS */
        :root {
            --primary-color: #1049A6;
            --primary-dark: #0d3a8a;
            --secondary-color: #D3E1EB;
            --accent-color: #2563eb;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --text-primary: #1f2937;
            --text-secondary: #6b7280;
            --background: #f8fafc;
            --surface: #ffffff;
            --border: #e5e7eb;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }

        /* Reset y estilos base */
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
            color: var(--text-primary);
        }

        /* Ocultar elementos por defecto de Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Cabecera mejorada */
        .header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 2rem 2.5rem;
            margin: -1rem -1rem 2rem -1rem;
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 20"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="20" fill="url(%23grid)"/></svg>');
            opacity: 0.3;
        }

        .header-content {
            position: relative;
            z-index: 1;
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }

        .logo-container {
            width: 60px;
            height: 60px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.3);
        }

        .header-text h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(45deg, #ffffff, #e2e8f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header-text p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
            font-weight: 400;
            font-size: 1rem;
        }

        /* Tarjetas mejoradas */
        .modern-card {
            background: var(--surface);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: var(--shadow-lg);
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .modern-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        }

        .modern-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-xl);
        }

        /* Área de chat mejorada */
        .chat-container {
            background: var(--surface);
            border-radius: 20px;
            box-shadow: var(--shadow-lg);
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
            overflow: hidden;
            position: relative;
        }

        .chat-header {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .status-indicator {
            width: 8px;
            height: 8px;
            background: var(--success-color);
            border-radius: 50%;
            animation: pulse-status 2s infinite;
        }

        @keyframes pulse-status {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .chat-messages {
            height: 65vh;
            overflow-y: auto;
            padding: 1.5rem;
            scroll-behavior: smooth;
        }

        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }

        .chat-messages::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 3px;
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 3px;
        }

        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }

        /* Mensajes mejorados */
        .message-container {
            display: flex;
            margin-bottom: 1.5rem;
            animation: messageSlideIn 0.3s ease-out;
        }

        @keyframes messageSlideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .user-message-container {
            justify-content: flex-end;
        }

        .assistant-message-container {
            justify-content: flex-start;
        }

        .message-bubble {
            max-width: 80%;
            padding: 1rem 1.25rem;
            border-radius: 20px;
            position: relative;
            word-wrap: break-word;
            box-shadow: var(--shadow-sm);
        }

        .user-message {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            color: white;
            border-bottom-right-radius: 6px;
        }

        .assistant-message {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            color: var(--text-primary);
            border-bottom-left-radius: 6px;
            border: 1px solid var(--border);
        }

        /* Avatar mejorado */
        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.875rem;
            margin: 0 0.75rem;
            flex-shrink: 0;
        }

        .user-avatar {
            background: linear-gradient(135deg, var(--accent-color), var(--primary-color));
            color: white;
            order: 1;
        }

        .assistant-avatar {
            background: linear-gradient(135deg, var(--success-color), #059669);
            color: white;
        }

        /* Indicador de escritura mejorado */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            color: var(--text-secondary);
            font-style: italic;
        }

        .typing-dots {
            display: flex;
            gap: 4px;
        }

        .typing-dot {
            width: 6px;
            height: 6px;
            background: var(--primary-color);
            border-radius: 50%;
            animation: typingPulse 1.4s infinite ease-in-out;
        }

        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes typingPulse {
            0%, 80%, 100% {
                transform: scale(0.8);
                opacity: 0.5;
            }
            40% {
                transform: scale(1);
                opacity: 1;
            }
        }

        /* Botones mejorados */
        .modern-button {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.875rem 1.5rem;
            font-weight: 600;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: var(--shadow-md);
            position: relative;
            overflow: hidden;
        }

        .modern-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }

        .modern-button:hover::before {
            left: 100%;
        }

        .modern-button:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-xl);
        }

        .stButton>button {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.875rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: var(--shadow-md) !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-xl) !important;
        }

        /* Input de chat mejorado */
        .stChatInput {
            border-radius: 20px !important;
            box-shadow: var(--shadow-lg) !important;
            border: 2px solid var(--border) !important;
            background: var(--surface) !important;
        }

        .stChatInput:focus-within {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(16, 73, 166, 0.1) !important;
        }

        /* Upload area mejorada */
        .upload-section {
            text-align: center;
            padding: 2rem;
        }

        .upload-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            color: var(--primary-color);
        }

        .upload-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .upload-subtitle {
            color: var(--text-secondary);
            margin-bottom: 2rem;
            line-height: 1.6;
        }

        /* Ejemplos de preguntas mejorados */
        .examples-container {
            margin-top: 2rem;
        }

        .examples-title {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .examples-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 0.75rem;
        }

        .example-chip {
            background: linear-gradient(135deg, #eef4fd 0%, #dbeafe 100%);
            color: var(--primary-color);
            padding: 0.75rem 1rem;
            border-radius: 12px;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid rgba(16, 73, 166, 0.1);
            text-align: center;
        }

        .example-chip:hover {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            color: white;
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }

        /* Stats y métricas */
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .stat-card {
            background: var(--surface);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border);
        }

        .stat-number {
            font-size: 1.875rem;
            font-weight: 700;
            color: var(--primary-color);
        }

        .stat-label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .header {
                padding: 1.5rem 1rem;
            }

            .header-content {
                flex-direction: column;
                text-align: center;
                gap: 1rem;
            }

            .modern-card {
                padding: 1.5rem;
                margin: 0 -0.5rem 2rem -0.5rem;
            }

            .chat-messages {
                height: 50vh;
                padding: 1rem;
            }

            .message-bubble {
                max-width: 90%;
            }

            .examples-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Animaciones adicionales */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .fade-in-up {
            animation: fadeInUp 0.6s ease-out;
        }

        /* Loading spinner */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(16, 73, 166, 0.3);
            border-radius: 50%;
            border-top-color: var(--primary-color);
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_header():
    """Renderiza la cabecera mejorada"""
    st.markdown(
        """
    <div class="header">
        <div class="header-content">
            <div class="logo-container">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="white">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                </svg>
            </div>
            <div class="header-text">
                <h1>Asistente IA Claro</h1>
                <p>🚀 Soporte técnico inteligente • Respuestas instantáneas • 24/7</p>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_upload_section():
    """Renderiza la sección de carga de archivos mejorada"""
    with st.container():
        st.markdown(
            """
        <div class="modern-card fade-in-up">
            <div class="upload-section">
                <div class="upload-icon">📄</div>
                <h3 class="upload-title">Cargar Documentos Técnicos</h3>
                <p class="upload-subtitle">
                    Sube manuales, inventarios o documentos técnicos para consultar.<br>
                    <strong>Formatos soportados:</strong> PDF, Excel (XLS/XLSX), CSV
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Arrastra tu archivo aquí o haz clic para seleccionar",
            type=["pdf", "xlsx", "xls", "csv"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.success(f"✅ **{uploaded_file.name}** cargado correctamente")
            with col2:
                file_size = len(uploaded_file.getvalue()) / 1024 / 1024
                st.info(f"📊 {file_size:.1f} MB")
            with col3:
                process_btn = st.button("🚀 Procesar", key="process_btn", use_container_width=True)

        st.markdown("""</div>""", unsafe_allow_html=True)

        # Ejemplos de preguntas mejorados
        render_examples()

        return uploaded_file, locals().get("process_btn", False)


def render_examples():
    """Renderiza ejemplos de preguntas de forma atractiva"""
    st.markdown(
        """
    <div class="examples-container">
        <div class="examples-title">
            💡 Ejemplos de consultas técnicas
        </div>
        <div class="examples-grid">
            <div class="example-chip">
                🏢 ¿Qué servicios hay en el sitio abad?
            </div>
            <div class="example-chip">
                📦 Características del gabinete
            </div>
            <div class="example-chip">
                🌐 Especificaciones del router Nokia
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_chat_interface():
    """Renderiza la interfaz de chat mejorada"""
    st.markdown(
        """
    <div class="chat-container">
        <div class="chat-header">
            <div class="status-indicator"></div>
            <span style="font-weight: 500; color: var(--text-primary);">Chat Activo</span>
            <span style="margin-left: auto; font-size: 0.875rem; color: var(--text-secondary);">
                {} mensajes
            </span>
        </div>
        <div class="chat-messages" id="chat-messages">
    """.format(
            len(st.session_state.messages)
        ),
        unsafe_allow_html=True,
    )

    # Mostrar mensajes
    for i, message in enumerate(st.session_state.messages):
        render_message(message, i)

    st.markdown(
        """
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_message(message, index):
    """Renderiza un mensaje individual con avatares"""
    if message["role"] == "user":
        st.markdown(
            f"""
        <div class="message-container user-message-container">
            <div class="message-bubble user-message">
                {message["content"]}
            </div>
            <div class="message-avatar user-avatar">
                TU
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="message-container assistant-message-container">
            <div class="message-avatar assistant-avatar">
                🤖
            </div>
            <div class="message-bubble assistant-message">
                {message["content"]}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_typing_indicator():
    """Renderiza el indicador de escritura mejorado"""
    return st.markdown(
        """
    <div class="message-container assistant-message-container">
        <div class="message-avatar assistant-avatar">
            🤖
        </div>
        <div class="message-bubble assistant-message">
            <div class="typing-indicator">
                <span>Procesando respuesta</span>
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_stats(processed_docs=0, total_messages=0):
    """Renderiza estadísticas del sistema"""
    st.markdown(
        f"""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-number">{processed_docs}</div>
            <div class="stat-label">Documentos procesados</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{total_messages}</div>
            <div class="stat-label">Consultas realizadas</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">24/7</div>
            <div class="stat-label">Disponibilidad</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def main():
    # Configuración de página
    st.set_page_config(
        page_title="Asistente IA Claro",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    load_css()
    render_header()

    # Inicialización de estado
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "show_upload" not in st.session_state:
        st.session_state.show_upload = True
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = 0

    # Estadísticas
    render_stats(
        processed_docs=st.session_state.processed_docs,
        total_messages=len(st.session_state.messages),
    )

    # Área de carga de archivos
    if st.session_state.show_upload:
        uploaded_file, process_btn = render_upload_section()

        if uploaded_file and process_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.info("🔍 Analizando documento...")
                progress_bar.progress(25)

                file_name = uploaded_file.name
                embds = retrieve_dataframe(file_name)

                if embds.empty:
                    status_text.info("🧠 Generando embeddings...")
                    progress_bar.progress(50)

                    final_text = extract_context(uploaded_file)
                    progress_bar.progress(75)

                    embds = compute_embeddings(final_text)
                    store_dataframe(file_name, embds)

                progress_bar.progress(100)
                status_text.success("✅ Documento procesado exitosamente")

                time.sleep(1)  # Pequeña pausa para mejor UX

                st.session_state.embds = embds
                st.session_state.processed = True
                st.session_state.show_upload = False
                st.session_state.messages = []
                st.session_state.processed_docs += 1

                st.rerun()

            except Exception as e:
                st.error(f"❌ Error al procesar el documento: {str(e)}")
                logger.error(f"Error processing file: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()

    # Interfaz de chat
    if st.session_state.processed:
        render_chat_interface()

        # Input de chat
        if prompt := st.chat_input("💬 Escribe tu pregunta técnica aquí...", key="chat_input"):
            # Agregar mensaje del usuario
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Mostrar mensaje del usuario inmediatamente
            render_message({"role": "user", "content": prompt}, len(st.session_state.messages))

            # Mostrar indicador de escritura
            typing_placeholder = st.empty()
            with typing_placeholder:
                render_typing_indicator()

            try:
                # Generar respuesta
                response_gen = response_generator(prompt, st.session_state.embds)

                # Contenedor para la respuesta en streaming
                response_placeholder = st.empty()
                full_response = ""

                # Simular streaming (si el generador no es realmente streaming)
                for chunk in response_gen:
                    full_response += chunk
                    # Actualizar respuesta progresivamente
                    with response_placeholder:
                        render_message(
                            {"role": "assistant", "content": full_response},
                            len(st.session_state.messages),
                        )

                # Limpiar indicador de escritura
                typing_placeholder.empty()

                # Guardar respuesta completa
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_msg = f"❌ Error al generar respuesta: {str(e)}"
                typing_placeholder.empty()
                st.error(error_msg)
                logger.error(f"Response generation error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

            st.rerun()

        # Botón para nueva consulta
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Nuevo Documento", use_container_width=True):
                st.session_state.show_upload = True
                st.session_state.processed = False
                st.rerun()


if __name__ == "__main__":
    main()
