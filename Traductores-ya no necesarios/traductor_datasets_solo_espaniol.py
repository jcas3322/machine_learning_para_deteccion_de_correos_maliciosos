#!/usr/bin/env python3
"""
Script mejorado para traducir datasets de phishing con manejo robusto de errores CUDA
"""

import pandas as pd
import re
import time
import os
import gc
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
from transformers import pipeline
import torch
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RobustPhishingTranslator:
    """Traductor robusto de datasets de phishing con manejo de errores CUDA"""
    
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-es", batch_size: int = 4):
        """
        Inicializa el traductor con configuración robusta
        
        Args:
            model_name: Nombre del modelo de Hugging Face
            batch_size: Tamaño del batch (reducido para evitar problemas de memoria)
        """
        logger.info(f"Inicializando modelo: {model_name}")
        
        # Detectar y configurar dispositivo
        self.device = self._configure_device()
        logger.info(f"Usando dispositivo: {'GPU' if self.device == 0 else 'CPU'}")
        
        # Cargar modelo con manejo de errores
        self.translator = self._load_model(model_name)
        self.batch_size = batch_size
        
        # Configuración de límites
        self.max_length = 300  # Reducido para evitar problemas
        self.max_tokens = 450  # Límite de tokens
        
        # Patrones para preservar elementos
        self.url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+')
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.number_pattern = re.compile(r'\b\d{4,}\b')
        
        # Términos que no deben traducirse
        self.preserve_terms = [
            'PayPal', 'eBay', 'Amazon', 'Microsoft', 'Google', 'Apple',
            'Facebook', 'Instagram', 'WhatsApp', 'Netflix', 'Spotify',
            'Windows', 'Office', 'Gmail', 'Yahoo', 'Outlook', 'Bank of America',
            'Wells Fargo', 'Chase', 'Citibank', 'HSBC', 'Barclays', 'IRS',
            'FBI', 'CIA', 'DHL', 'FedEx', 'UPS', 'USPS'
        ]
        
        logger.info("Traductor inicializado correctamente")
    
    def _configure_device(self) -> int:
        """Configura el dispositivo de manera robusta"""
        if torch.cuda.is_available():
            try:
                # Verificar que CUDA funciona
                torch.cuda.empty_cache()
                device_count = torch.cuda.device_count()
                logger.info(f"CUDA disponible con {device_count} GPU(s)")
                
                # Obtener información de memoria
                if device_count > 0:
                    memory_info = torch.cuda.get_device_properties(0)
                    total_memory = memory_info.total_memory / (1024**3)  # GB
                    logger.info(f"Memoria GPU total: {total_memory:.1f} GB")
                    
                    # Si hay poca memoria, usar CPU
                    if total_memory < 4:
                        logger.warning("Poca memoria GPU disponible, usando CPU")
                        return -1
                
                return 0
            except Exception as e:
                logger.warning(f"Error configurando CUDA, usando CPU: {e}")
                return -1
        else:
            logger.info("CUDA no disponible, usando CPU")
            return -1
    
    def _load_model(self, model_name: str):
        """Carga el modelo con manejo de errores"""
        try:
            translator = pipeline(
                "translation",
                model=model_name,
                device=self.device,
                torch_dtype=torch.float16 if self.device == 0 else torch.float32,
                model_kwargs={"low_cpu_mem_usage": True} if self.device == 0 else {}
            )
            return translator
        except Exception as e:
            logger.error(f"Error cargando modelo en GPU: {e}")
            logger.info("Intentando cargar en CPU...")
            self.device = -1
            return pipeline(
                "translation",
                model=model_name,
                device=-1
            )
    
    def _clean_cuda_memory(self):
        """Limpia la memoria CUDA"""
        if self.device == 0:
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except:
                pass
    
    def _sanitize_text(self, texto: str) -> str:
        """Sanitiza el texto para evitar problemas de tokenización"""
        if pd.isna(texto) or texto == '':
            return texto
        
        texto = str(texto)
        
        # Remover caracteres problemáticos
        texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', texto)
        
        # Limitar longitud extrema
        if len(texto) > 2000:
            texto = texto[:2000] + "..."
        
        # Normalizar espacios
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        return texto
    
    def preservar_elementos_criticos(self, texto: str) -> Tuple[str, Dict]:
        """Extrae y preserva URLs, emails y otros elementos críticos"""
        if pd.isna(texto) or texto == '':
            return texto, {}
        
        texto = self._sanitize_text(texto)
        elementos_preservados = {
            'urls': [],
            'emails': [],
            'numeros': [],
            'terminos': []
        }
        
        # Preservar URLs
        urls = self.url_pattern.findall(texto)
        for i, url in enumerate(urls):
            placeholder = f'URLPLACEHOLDER{i}'
            texto = texto.replace(url, placeholder)
            elementos_preservados['urls'].append((placeholder, url))
        
        # Preservar emails
        emails = self.email_pattern.findall(texto)
        for i, email in enumerate(emails):
            placeholder = f'EMAILPLACEHOLDER{i}'
            texto = texto.replace(email, placeholder)
            elementos_preservados['emails'].append((placeholder, email))
        
        # Preservar números largos
        numeros = self.number_pattern.findall(texto)
        for i, numero in enumerate(numeros):
            placeholder = f'NUMPLACEHOLDER{i}'
            texto = texto.replace(numero, placeholder)
            elementos_preservados['numeros'].append((placeholder, numero))
        
        # Preservar términos de empresas (simplificado)
        for i, termino in enumerate(self.preserve_terms):
            if termino.lower() in texto.lower():
                placeholder = f'TERMPLACEHOLDER{i}'
                texto = re.sub(
                    re.escape(termino), 
                    placeholder, 
                    texto, 
                    flags=re.IGNORECASE
                )
                elementos_preservados['terminos'].append((placeholder, termino))
        
        return texto, elementos_preservados
    
    def restaurar_elementos(self, texto_traducido: str, elementos: Dict) -> str:
        """Restaura los elementos preservados en el texto traducido"""
        if pd.isna(texto_traducido):
            return texto_traducido
        
        # Restaurar en orden
        for categoria in ['urls', 'emails', 'numeros', 'terminos']:
            for placeholder, original in elementos.get(categoria, []):
                texto_traducido = texto_traducido.replace(placeholder, original)
        
        return texto_traducido
    
    def traducir_texto(self, texto: str, max_reintentos: int = 3) -> str:
        """Traduce un texto con manejo robusto de errores"""
        if pd.isna(texto) or texto == '' or len(str(texto).strip()) == 0:
            return texto
        
        for intento in range(max_reintentos):
            try:
                # Limpiar memoria antes de cada traducción
                self._clean_cuda_memory()
                
                # Preservar elementos
                texto_procesado, elementos = self.preservar_elementos_criticos(texto)
                
                if len(texto_procesado.strip()) == 0:
                    return texto
                
                # Verificar longitud
                if len(texto_procesado) > self.max_length:
                    # Dividir en chunks más pequeños
                    chunks = self._split_text(texto_procesado)
                    traducciones = []
                    
                    for chunk in chunks:
                        if len(chunk.strip()) > 0:
                            try:
                                result = self.translator(
                                    chunk, 
                                    max_length=self.max_tokens
                                )
                                traducciones.append(result[0]['translation_text'])
                            except Exception as e:
                                logger.warning(f"Error en chunk, saltando: {e}")
                                traducciones.append(chunk)  # Mantener original
                    
                    texto_traducido = ' '.join(traducciones)
                else:
                    result = self.translator(
                        texto_procesado, 
                        max_length=self.max_tokens
                    )
                    texto_traducido = result[0]['translation_text']
                
                # Restaurar elementos
                texto_final = self.restaurar_elementos(texto_traducido, elementos)
                return texto_final
                
            except RuntimeError as e:
                if "CUDA" in str(e) or "device-side assert" in str(e):
                    logger.warning(f"Error CUDA en intento {intento + 1}: {e}")
                    
                    # Limpiar memoria agresivamente
                    self._clean_cuda_memory()
                    time.sleep(1)  # Esperar un poco
                    
                    # En el último intento, cambiar a CPU
                    if intento == max_reintentos - 1:
                        logger.info("Cambiando a CPU para este texto")
                        try:
                            # Crear traductor temporal en CPU
                            cpu_translator = pipeline(
                                "translation",
                                model="Helsinki-NLP/opus-mt-en-es",
                                device=-1
                            )
                            result = cpu_translator(texto_procesado[:self.max_length])
                            return self.restaurar_elementos(
                                result[0]['translation_text'], 
                                elementos
                            )
                        except Exception as cpu_error:
                            logger.error(f"Error también en CPU: {cpu_error}")
                            return texto
                else:
                    logger.error(f"Error no-CUDA: {e}")
                    return texto
            except Exception as e:
                logger.error(f"Error general traduciendo: {e}")
                if intento == max_reintentos - 1:
                    return texto
                time.sleep(0.5)
        
        return texto
    
    def _split_text(self, texto: str) -> List[str]:
        """Divide texto en chunks más pequeños"""
        # Intentar dividir por oraciones
        oraciones = re.split(r'(?<=[.!?])\s+', texto)
        
        chunks = []
        chunk_actual = ""
        
        for oracion in oraciones:
            if len(chunk_actual) + len(oracion) < self.max_length:
                chunk_actual += " " + oracion if chunk_actual else oracion
            else:
                if chunk_actual:
                    chunks.append(chunk_actual)
                chunk_actual = oracion
        
        if chunk_actual:
            chunks.append(chunk_actual)
        
        # Si aún hay chunks muy largos, dividir por caracteres
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_length:
                # Dividir en pedazos más pequeños
                for i in range(0, len(chunk), self.max_length):
                    final_chunks.append(chunk[i:i+self.max_length])
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def traducir_dataset(self, df: pd.DataFrame, columnas: List[str] = ['body', 'subject']) -> pd.DataFrame:
        """Traduce las columnas especificadas de un DataFrame y reemplaza el contenido original"""
        df_traducido = df.copy()
        
        for columna in columnas:
            if columna not in df.columns:
                logger.warning(f"Columna '{columna}' no encontrada")
                continue
            
            logger.info(f"Traduciendo columna '{columna}'...")
            traducciones = []
            
            # Procesar con progreso y manejo de errores
            textos = df[columna].tolist()
            for i, texto in enumerate(tqdm(textos, desc=f"Traduciendo {columna}")):
                try:
                    texto_traducido = self.traducir_texto(texto)
                    traducciones.append(texto_traducido)
                    
                    # Limpiar memoria cada 10 traducciones
                    if i % 10 == 0:
                        self._clean_cuda_memory()
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Error en fila {i}: {e}")
                    traducciones.append(texto)  # Mantener original como fallback
            
            # REEMPLAZAR la columna original con la traducción
            df_traducido[columna] = traducciones
            logger.info(f"Columna '{columna}' traducida y reemplazada")
            
            # Limpiar memoria después de cada columna
            self._clean_cuda_memory()
        
        return df_traducido
    
    def procesar_solo_body(self, ruta_entrada: str, ruta_salida: str):
        """Procesa un archivo CSV traduciendo únicamente la columna 'body'"""
        logger.info(f"\n🎯 Procesando SOLO columna 'body': {ruta_entrada}")
        
        try:
            # Leer archivo con múltiples encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(ruta_entrada, encoding=encoding)
                    logger.info(f"Archivo leído con encoding: {encoding}")
                    break
                except Exception as e:
                    logger.debug(f"Fallo con encoding {encoding}: {e}")
                    continue
            
            if df is None:
                raise ValueError("No se pudo leer el archivo con ningún encoding")
            
            logger.info(f"Dimensiones: {len(df)} filas x {len(df.columns)} columnas")
            logger.info(f"Columnas encontradas: {list(df.columns)}")
            
            # Verificar si existe la columna 'body'
            if 'body' not in df.columns:
                logger.error("❌ No se encontró la columna 'body' en el dataset")
                print("Columnas disponibles:", list(df.columns))
                return
            
            # Traducir solo la columna 'body'
            df_traducido = self.traducir_dataset(df, ['body'])
            
            # Mostrar información del resultado
            logger.info(f"Dataset traducido - Columnas finales: {list(df_traducido.columns)}")
            logger.info(f"La columna 'body' ahora contiene texto en español")
            
            # Guardar con manejo de errores
            try:
                df_traducido.to_csv(ruta_salida, index=False, encoding='utf-8')
                logger.info(f"Guardado en: {ruta_salida}")
            except Exception as e:
                # Intentar con encoding alternativo
                df_traducido.to_csv(ruta_salida, index=False, encoding='latin-1')
                logger.info(f"Guardado con encoding latin-1 en: {ruta_salida}")
            
            # Mostrar ejemplos
            self._mostrar_ejemplos(df, df_traducido, ['body'])
            
        except Exception as e:
            logger.error(f"Error procesando archivo: {e}")
            raise

    def procesar_archivo(self, ruta_entrada: str, ruta_salida: str, columnas: List[str] = ['body', 'subject']):
        """Procesa un archivo CSV completo con manejo robusto de errores"""
        logger.info(f"\nProcesando: {ruta_entrada}")
        
        try:
            # Leer archivo con múltiples encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(ruta_entrada, encoding=encoding)
                    logger.info(f"Archivo leído con encoding: {encoding}")
                    break
                except Exception as e:
                    logger.debug(f"Fallo con encoding {encoding}: {e}")
                    continue
            
            if df is None:
                raise ValueError("No se pudo leer el archivo con ningún encoding")
            
            logger.info(f"Dimensiones: {len(df)} filas x {len(df.columns)} columnas")
            logger.info(f"Columnas encontradas: {list(df.columns)}")
            
            # Verificar si hay columnas válidas
            columnas_validas = [col for col in columnas if col in df.columns]
            if not columnas_validas:
                logger.warning("No se encontraron columnas válidas para traducir")
                return
            
            # Traducir
            df_traducido = self.traducir_dataset(df, columnas_validas)
            
            # Guardar con manejo de errores
            try:
                df_traducido.to_csv(ruta_salida, index=False, encoding='utf-8')
                logger.info(f"Guardado en: {ruta_salida}")
            except Exception as e:
                # Intentar con encoding alternativo
                df_traducido.to_csv(ruta_salida, index=False, encoding='latin-1')
                logger.info(f"Guardado con encoding latin-1 en: {ruta_salida}")
            
            # Mostrar ejemplos
            self._mostrar_ejemplos(df, df_traducido, columnas_validas)
            
        except Exception as e:
            logger.error(f"Error procesando archivo: {e}")
            raise
    
    def _mostrar_ejemplos(self, df_original: pd.DataFrame, df_traducido: pd.DataFrame, columnas: List[str]):
        """Muestra ejemplos de traducción"""
        print("\n=== Ejemplos de traducción ===")
        for col in columnas:
            if col in df_original.columns and col in df_traducido.columns:
                # Buscar un ejemplo no vacío
                mask = df_original[col].notna() & (df_original[col] != '')
                if mask.any():
                    idx = df_original[mask].index[0]
                    original = str(df_original.iloc[idx][col])
                    traducido = str(df_traducido.iloc[idx][col])
                    
                    print(f"\n{col}:")
                    print(f"EN: {original[:150]}{'...' if len(original) > 150 else ''}")
                    print(f"ES: {traducido[:150]}{'...' if len(traducido) > 150 else ''}")
                    print(f"✅ Columna '{col}' ahora contiene solo el texto en español")

def main():
    """Función principal con manejo robusto"""
    
    # Configuración
    datasets = {
        'dataset_entrenamiento': ['CEAS_08.csv', 'Enron.csv'],
        'dataset_testing': ['Ling.csv', 'Nigerian_Fraud.csv']
    }
    
    try:
        # Crear traductor con configuración robusta
        traductor = RobustPhishingTranslator(batch_size=2)  # Batch pequeño
        
        # Procesar archivos
        for carpeta, archivos in datasets.items():
            print(f"\n{'='*60}")
            print(f"Procesando carpeta: {carpeta}")
            print(f"{'='*60}")
            
            # Crear carpeta de salida
            carpeta_salida = f"{carpeta}_es"
            Path(carpeta_salida).mkdir(exist_ok=True)
            
            for archivo in archivos:
                ruta_entrada = os.path.join(carpeta, archivo)
                nombre_sin_ext = os.path.splitext(archivo)[0]
                ruta_salida = os.path.join(carpeta_salida, f"{nombre_sin_ext}_es.csv")
                
                if os.path.exists(ruta_entrada):
                    try:
                        traductor.procesar_archivo(ruta_entrada, ruta_salida)
                        print(f"✅ {archivo} procesado exitosamente")
                    except Exception as e:
                        logger.error(f"❌ Error con {archivo}: {e}")
                        print(f"❌ Error procesando {archivo}")
                else:
                    logger.warning(f"⚠️  No encontrado: {ruta_entrada}")
        
        print("\n🎉 ¡Proceso completado!")
        print("\nArchivos traducidos guardados en:")
        print("  - dataset_entrenamiento_es/")
        print("  - dataset_testing_es/")
        
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        print(f"❌ Error crítico: {e}")

if __name__ == "__main__":
    main()
