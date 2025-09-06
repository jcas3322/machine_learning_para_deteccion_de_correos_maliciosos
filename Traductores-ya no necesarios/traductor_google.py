#!/usr/bin/env python3
"""
Script para traducir datasets de phishing usando Google Translate
Más ligero y rápido que los modelos locales
"""

import pandas as pd
import re
import time
import os
from pathlib import Path
import logging
from typing import List, Tuple, Dict
from googletrans import Translator
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GooglePhishingTranslator:
    """Traductor de datasets usando Google Translate"""
    
    def __init__(self, delay: float = 0.2):
        """
        Inicializa el traductor
        
        Args:
            delay: Retraso entre traducciones para evitar límites
        """
        logger.info("Inicializando traductor de Google")
        self.translator = Translator()
        self.delay = delay
        
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
            'FBI', 'CIA', 'DHL', 'FedEx', 'UPS', 'USPS', 'Western Union',
            'MoneyGram', 'Bitcoin', 'Ethereum', 'COVID-19', 'COVID'
        ]
        
        logger.info("Traductor inicializado correctamente")
    
    def preservar_elementos_criticos(self, texto: str) -> Tuple[str, Dict]:
        """Extrae y preserva URLs, emails y otros elementos críticos"""
        if pd.isna(texto) or texto == '':
            return texto, {}
        
        texto = str(texto)
        elementos_preservados = {
            'urls': [],
            'emails': [],
            'numeros': [],
            'terminos': []
        }
        
        # Preservar URLs
        urls = self.url_pattern.findall(texto)
        for i, url in enumerate(urls):
            placeholder = f'__URL{i}__'
            texto = texto.replace(url, placeholder)
            elementos_preservados['urls'].append((placeholder, url))
        
        # Preservar emails
        emails = self.email_pattern.findall(texto)
        for i, email in enumerate(emails):
            placeholder = f'__EMAIL{i}__'
            texto = texto.replace(email, placeholder)
            elementos_preservados['emails'].append((placeholder, email))
        
        # Preservar números largos
        numeros = self.number_pattern.findall(texto)
        for i, numero in enumerate(numeros):
            placeholder = f'__NUM{i}__'
            texto = texto.replace(numero, placeholder)
            elementos_preservados['numeros'].append((placeholder, numero))
        
        # Preservar términos de empresas
        for i, termino in enumerate(self.preserve_terms):
            # Buscar todas las ocurrencias del término (case insensitive)
            pattern = re.compile(r'\b' + re.escape(termino) + r'\b', re.IGNORECASE)
            matches = list(pattern.finditer(texto))
            
            # Reemplazar de atrás hacia adelante para no afectar las posiciones
            for j, match in enumerate(reversed(matches)):
                original = match.group()
                placeholder = f'__TERM{i}_{j}__'
                start, end = match.span()
                texto = texto[:start] + placeholder + texto[end:]
                elementos_preservados['terminos'].append((placeholder, original))
        
        return texto, elementos_preservados
    
    def restaurar_elementos(self, texto_traducido: str, elementos: Dict) -> str:
        """Restaura los elementos preservados en el texto traducido"""
        if pd.isna(texto_traducido):
            return texto_traducido
        
        # Restaurar en orden inverso
        for categoria in ['terminos', 'numeros', 'emails', 'urls']:
            for placeholder, original in elementos.get(categoria, []):
                texto_traducido = texto_traducido.replace(placeholder, original)
        
        return texto_traducido
    
    def traducir_texto(self, texto: str, reintentos: int = 3) -> str:
        """Traduce un texto preservando elementos críticos"""
        if pd.isna(texto) or texto == '' or len(str(texto).strip()) == 0:
            return texto
        
        # Preservar elementos
        texto_procesado, elementos = self.preservar_elementos_criticos(texto)
        
        if len(texto_procesado.strip()) == 0:
            return texto
        
        # Intentar traducir con reintentos
        for intento in range(reintentos):
            try:
                # Google Translate tiene un límite de ~5000 caracteres
                if len(texto_procesado) > 4500:
                    # Dividir en chunks por oraciones
                    oraciones = re.split(r'(?<=[.!?])\s+', texto_procesado)
                    traducciones = []
                    
                    chunk_actual = ""
                    for oracion in oraciones:
                        if len(chunk_actual) + len(oracion) < 4500:
                            chunk_actual += " " + oracion if chunk_actual else oracion
                        else:
                            if chunk_actual:
                                traducido = self.translator.translate(chunk_actual, src='en', dest='es').text
                                traducciones.append(traducido)
                                time.sleep(self.delay)
                            chunk_actual = oracion
                    
                    if chunk_actual:
                        traducido = self.translator.translate(chunk_actual, src='en', dest='es').text
                        traducciones.append(traducido)
                    
                    texto_traducido = ' '.join(traducciones)
                else:
                    # Traducir directamente
                    resultado = self.translator.translate(texto_procesado, src='en', dest='es')
                    texto_traducido = resultado.text
                
                # Restaurar elementos
                texto_final = self.restaurar_elementos(texto_traducido, elementos)
                
                # Pausa para evitar límites de tasa
                time.sleep(self.delay)
                
                return texto_final
                
            except Exception as e:
                logger.warning(f"Intento {intento + 1} falló: {e}")
                if intento < reintentos - 1:
                    time.sleep(2 ** intento)  # Backoff exponencial
                else:
                    logger.error(f"Error traduciendo después de {reintentos} intentos")
                    return texto
    
    def traducir_dataset(self, df: pd.DataFrame, columnas: List[str] = ['subject', 'body']) -> pd.DataFrame:
        """Traduce las columnas especificadas de un DataFrame"""
        df_traducido = df.copy()
        
        for columna in columnas:
            if columna not in df.columns:
                logger.warning(f"Columna '{columna}' no encontrada")
                continue
            
            logger.info(f"Traduciendo columna '{columna}'...")
            traducciones = []
            errores = 0
            
            # Usar tqdm para mostrar progreso
            textos = df[columna].tolist()
            pbar = tqdm(textos, desc=f"Traduciendo {columna}")
            
            for i, texto in enumerate(pbar):
                try:
                    texto_traducido = self.traducir_texto(texto)
                    traducciones.append(texto_traducido)
                except Exception as e:
                    logger.error(f"Error en fila {i}: {e}")
                    traducciones.append(texto)  # Mantener original si falla
                    errores += 1
                
                # Actualizar descripción con estadísticas
                if i % 10 == 0:
                    pbar.set_postfix({'errores': errores})
            
            df_traducido[f'{columna}_es'] = traducciones
            logger.info(f"Columna '{columna}' traducida ({errores} errores)")
        
        return df_traducido
    
    def procesar_archivo(self, ruta_entrada: str, ruta_salida: str, columnas: List[str] = ['subject', 'body']):
        """Procesa un archivo CSV completo"""
        logger.info(f"\nProcesando: {ruta_entrada}")
        
        try:
            # Intentar diferentes encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(ruta_entrada, encoding=encoding)
                    logger.info(f"Archivo leído con encoding: {encoding}")
                    break
                except Exception as e:
                    continue
            
            if df is None:
                raise Exception("No se pudo leer el archivo con ningún encoding")
            
            logger.info(f"Dimensiones: {len(df)} filas x {len(df.columns)} columnas")
            logger.info(f"Columnas encontradas: {list(df.columns)}")
            
            # Verificar columnas
            columnas_disponibles = [col for col in columnas if col in df.columns]
            if not columnas_disponibles:
                logger.warning("No se encontraron las columnas especificadas")
                return
            
            # Traducir
            df_traducido = self.traducir_dataset(df, columnas_disponibles)
            
            # Guardar
            df_traducido.to_csv(ruta_salida, index=False, encoding='utf-8')
            logger.info(f"Guardado en: {ruta_salida}")
            
            # Mostrar ejemplos
            print("\n=== Ejemplos de traducción ===")
            for col in columnas_disponibles:
                if f'{col}_es' in df_traducido.columns:
                    # Buscar ejemplos no vacíos
                    mask = df[col].notna() & (df[col].astype(str).str.len() > 10)
                    if mask.any():
                        # Mostrar hasta 3 ejemplos
                        indices = df[mask].index[:3]
                        for idx in indices:
                            print(f"\n{col} - Ejemplo {idx + 1}:")
                            original = str(df.iloc[idx][col])
                            traducido = str(df_traducido.iloc[idx][f'{col}_es'])
                            print(f"EN: {original[:100]}{'...' if len(original) > 100 else ''}")
                            print(f"ES: {traducido[:100]}{'...' if len(traducido) > 100 else ''}")
            
        except Exception as e:
            logger.error(f"Error procesando archivo: {e}")
            raise

def main():
    """Función principal"""
    
    # Configuración
    datasets = {
        'dataset_entrenamiento': ['CEAS_08.csv', 'Enron.csv'],
        'dataset_testing': ['Ling.csv', 'Nigerian_Fraud.csv']
    }
    
    print("=" * 60)
    print("TRADUCTOR DE DATASETS DE PHISHING")
    print("=" * 60)
    print("\nEste script traducirá los datasets preservando:")
    print("- URLs y dominios")
    print("- Direcciones de email")
    print("- Nombres de empresas y organizaciones")
    print("- Números de cuenta y referencias")
    print("\n")
    
    # Crear traductor
    traductor = GooglePhishingTranslator(delay=0.2)
    
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
                except Exception as e:
                    logger.error(f"Error con {archivo}: {e}")
                    print(f"\n⚠️  Error procesando {archivo}")
                    print("   Continuando con el siguiente archivo...")
            else:
                logger.warning(f"No encontrado: {ruta_entrada}")
                print(f"\n⚠️  Archivo no encontrado: {ruta_entrada}")
    
    print("\n" + "="*60)
    print("✅ ¡PROCESO COMPLETADO!")
    print("="*60)
    print("\nArchivos traducidos guardados en:")
    print("  📁 dataset_entrenamiento_es/")
    print("  📁 dataset_testing_es/")
    print("\nRevisa los archivos para verificar la calidad de las traducciones.")

if __name__ == "__main__":
    main()
