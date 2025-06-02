# Chatbot LLM Local

Esta carpeta contiene código para ejecutar modelos de lenguaje pequeños (SmolLM) localmente en tu computadora.

## Qué Contiene

Dos enfoques diferentes para ejecutar modelos de lenguaje locales:

1. `model_local_thf.py` - Implementación simple usando solo Transformers
2. `model_local_lc.py` - Implementación avanzada usando LangChain con Transformers

## Comenzando

Necesitarás Python 3.12+ y estas bibliotecas:
```
pip install torch transformers langchain-huggingface langchain-core
```

## Modelos Disponibles

Puedes elegir entre estos modelos SmolLM:
- `HuggingFaceTB/SmolLM-135M-Instruct` (Más pequeño)
- `HuggingFaceTB/SmolLM2-360M-Instruct` (Mediano)
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` (Más grande)

Los modelos más grandes dan mejores respuestas pero necesitan más potencia de cómputo.

## Cómo Funciona

### Versión Simple (`model_local_thf.py`)
Usa solo la biblioteca Transformers para:
- Cargar el modelo
- Formatear tus mensajes
- Generar una respuesta
- Mostrar el resultado

### Versión Avanzada (`model_local_lc.py`)
Usa Transformers + LangChain para:
- Verificar la disponibilidad de GPU
- Aplicar configuraciones óptimas para tu hardware
- Crear un pipeline de generación de texto personalizable
- Formatear conversaciones usando el sistema de mensajes de LangChain

## Parámetros Clave Explicados

- `max_new_tokens`: Longitud máxima de la respuesta
- `temperature`: Controla la creatividad (mayor = más aleatorio)
- `do_sample`: Cuando es True, añade variedad a las respuestas
- `repetition_penalty`: Evita repeticiones (mayor = menos repetición)
- `no_repeat_ngram_size`: Bloquea la repetición de frases de esta longitud

## Notas de Hardware

El código se adapta automáticamente a tu hardware:
- Con GPU: Usa float16 (más rápido pero menos preciso)
- Sin GPU: Usa float32 (más compatible con CPUs)

## Ejecutando el Código

1. Elige qué archivo ejecutar:
   ```
   python model_local_thf.py
   ```
   o
   ```
   python model_local_lc.py
   ```

2. Para personalizar:
   - Cambia el tamaño del modelo en el código
   - Edita el mensaje del sistema para controlar el comportamiento de la IA
   - Cambia el mensaje del usuario (tu pregunta)

## Consejos para Principiantes

- Comienza con modelos más pequeños si tu computadora no es muy potente
- La primera ejecución descargará el modelo (puede tomar tiempo)
- Si obtienes errores de memoria, prueba un modelo más pequeño
- Estos modelos no son tan inteligentes como ChatGPT pero funcionan sin conexión
- Sé específico en tus preguntas para obtener mejores respuestas

## Problemas Comunes

- **Sin memoria**: Prueba un modelo más pequeño o reduce `max_new_tokens`
- **Respuestas lentas**: Normal para la primera ejecución o con modelos más grandes
- **Texto repetido**: Aumenta `repetition_penalty` o `no_repeat_ngram_size`
- **Respuestas sin sentido**: Prueba un modelo más grande o reformula tu pregunta
