#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import re
import email.utils
from email import message_from_string
from email.message import Message
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from urllib.parse import urlparse
import ipaddress
import warnings

# Suprimir warnings de sklearn sobre feature names
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

rutaBase = '003_xgboost_phishing_count_vector_9000/'

# =============================================
# LEER Y CARGAR LOS MODELOS DE LA BASE DE DATOS
# =============================================
def leer_modelos():
    try:
        model = joblib.load(rutaBase+'model.pkl')
        vectorizer = joblib.load(rutaBase+'vectorizer.pkl')
        scaler = joblib.load(rutaBase+'scaler')

        return model, vectorizer, scaler
    except Exception as e:
        print(f"[ERROR] Fallo al leer de la base de datos: {e}")
        return None, None, None, None

# Uso recomendado:
model, vectorizer, scaler = leer_modelos()
if not all([model, vectorizer, scaler]):
    print("[ERROR] No se pudieron cargar los modelos. Saliendo.")
    sys.exit(1)


# Palabras clave de phishing
PHISHING_KEYWORDS = {
    'english': {
        'urgency': ['urgent', 'expire', 'suspend', 'immediate', 'limited time', 'act now', 
                   'hurry', 'deadline', 'critical', 'important notice'],
        'action': ['click here', 'verify', 'confirm', 'update', 'validate', 'secure',
                  'restore', 'unlock', 'activate', 'claim'],
        'money': ['account', 'payment', 'billing', 'credit', 'debit', 'bank', 'paypal',
                 'refund', 'prize', 'winner', 'lottery', 'tax'],
        'threat': ['suspended', 'blocked', 'restricted', 'locked', 'illegal', 'unauthorized',
                  'breach', 'compromised', 'violation', 'terminate']
    },
    'spanish': {
        'urgency': ['urgente', 'expira', 'suspender', 'inmediato', 'tiempo limitado', 
                   'actúa ahora', 'apresúrate', 'fecha límite', 'crítico', 'aviso importante'],
        'action': ['haga clic aquí', 'verificar', 'confirmar', 'actualizar', 'validar',
                  'asegurar', 'restaurar', 'desbloquear', 'activar', 'reclamar'],
        'money': ['cuenta', 'pago', 'facturación', 'crédito', 'débito', 'banco', 'paypal',
                 'reembolso', 'premio', 'ganador', 'lotería', 'impuesto'],
        'threat': ['suspendida', 'bloqueada', 'restringida', 'bloqueado', 'ilegal',
                  'no autorizado', 'violación', 'comprometido', 'infracción', 'terminar']
    }
}

def detect_language(text):
    """Detectar idioma simple"""
    spanish_words = set(['el', 'la', 'de', 'que', 'y', 'en', 'un', 'para'])
    english_words = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that'])
    
    words = text.lower().split()[:50]
    spanish_count = sum(1 for word in words if word in spanish_words)
    english_count = sum(1 for word in words if word in english_words)
    
    if spanish_count > english_count:
        return 'spanish'
    return 'english'

def phishing_preprocessing(text):
    """Preprocesamiento específico para phishing"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Preservar URLs marcándolas temporalmente
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    urls = re.findall(url_pattern, text)
    for i, url in enumerate(urls):
        text = text.replace(url, f" URL{i} ")
    
    # Preservar emails marcándolos temporalmente  
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    for i, email in enumerate(emails):
        text = text.replace(email, f" EMAIL{i} ")
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Restaurar URLs y emails
    for i, url in enumerate(urls):
        text = text.replace(f"url{i}", url.lower())
    for i, email in enumerate(emails):
        text = text.replace(f"email{i}", email.lower())
    
    # Eliminar múltiples guiones bajos
    text = re.sub(r'_{3,}', ' ', text)
    
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_url_features(text):
    """Extraer características de URLs"""
    features = {
        'url_count': 0,
        'shortened_url': 0,
        'has_ip': 0,
        'suspicious_domain': 0,
        'long_url': 0,
        'has_at_symbol': 0,
        'multiple_subdomains': 0,
        'https_count': 0,
        'http_count': 0
    }
    
    # Patrones de URL
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    urls = re.findall(url_pattern, text.lower())
    
    features['url_count'] = len(urls)
    
    # URL shorteners comunes
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 'short.link', 't.co']
    
    for url in urls:
        # URL acortada
        if any(short in url for short in shorteners):
            features['shortened_url'] += 1
        
        # Tiene IP en lugar de dominio
        try:
            parsed = urlparse(url)
            if parsed.hostname:
                ipaddress.ip_address(parsed.hostname)
                features['has_ip'] += 1
        except:
            pass
        
        # URL larga (sospechosa)
        if len(url) > 75:
            features['long_url'] += 1
        
        # Tiene @ (redirección)
        if '@' in url:
            features['has_at_symbol'] += 1
        
        # Múltiples subdominios
        if url.count('.') > 3:
            features['multiple_subdomains'] += 1
        
        # HTTP vs HTTPS
        if url.startswith('https'):
            features['https_count'] += 1
        elif url.startswith('http:'):
            features['http_count'] += 1
    
    return features

def extract_phishing_features(text, language='english'):
    """Extraer características de phishing"""
    features = {}
    text_lower = text.lower()
    
    # Contar palabras clave por categoría
    keywords = PHISHING_KEYWORDS.get(language, PHISHING_KEYWORDS['english'])
    
    for category, words in keywords.items():
        count = sum(1 for word in words if word in text_lower)
        features[f'{category}_words'] = count
    
    # Características adicionales
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['uppercase_words'] = len(re.findall(r'\b[A-Z]{2,}\b', text))
    features['special_chars'] = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', text))
    
    # Detectar emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    features['email_count'] = len(emails)
    
    # Dominios legítimos para comparación
    legitimate_domains = ['paypal.com', 'amazon.com', 'google.com', 'microsoft.com', 
                         'apple.com', 'facebook.com', 'chase.com', 'netflix.com']
    
    # Verificar dominios sospechosos
    suspicious_email = 0
    for email in emails:
        domain = email.split('@')[1].lower()
        # Buscar typosquatting simple
        for legit_domain in legitimate_domains:
            if domain != legit_domain and any([
                domain.replace('0', 'o') == legit_domain,
                domain.replace('1', 'l') == legit_domain,
                legit_domain.split('.')[0] in domain and domain != legit_domain
            ]):
                suspicious_email += 1
                break
    
    features['suspicious_email'] = suspicious_email
    
    return features

def detect_phishing(email_text):
    """Detectar si un email es phishing"""
    # Detectar idioma
    language = detect_language(email_text)
    
    # Preprocesar texto
    processed_text = phishing_preprocessing(email_text)
    
    # Vectorizar
    text_vector = vectorizer.transform([processed_text])
    
    # Extraer características adicionales
    url_features = extract_url_features(email_text)
    phishing_features = extract_phishing_features(email_text, language)
    all_features = {**url_features, **phishing_features}
    
    # Convertir a array en el orden correcto
    feature_names = ['url_count', 'shortened_url', 'has_ip', 'suspicious_domain',
                    'long_url', 'has_at_symbol', 'multiple_subdomains', 'https_count',
                    'http_count', 'urgency_words', 'action_words', 'money_words',
                    'threat_words', 'exclamation_count', 'question_count',
                    'uppercase_words', 'special_chars', 'email_count', 'suspicious_email']
    
    # Crear DataFrame con nombres de características
    features_df = pd.DataFrame([[all_features.get(name, 0) for name in feature_names]], 
                              columns=feature_names)
    features_scaled = scaler.transform(features_df)
    
    # Combinar características
    if hasattr(text_vector, 'toarray'):
        combined_features = np.hstack([text_vector.toarray(), features_scaled])
    else:
        combined_features = np.hstack([text_vector, features_scaled])
    
    # Predecir
    prediction = model.predict(combined_features)[0]
    probability = model.predict_proba(combined_features)[0][1]
    
    # Análisis de características sospechosas
    suspicious_indicators = []
    if url_features['shortened_url'] > 0:
        suspicious_indicators.append("URL acortada")
    if url_features['has_ip'] > 0:
        suspicious_indicators.append("IP en lugar de dominio")
    if phishing_features['urgency_words'] > 2:
        suspicious_indicators.append("Lenguaje urgente")
    if phishing_features['threat_words'] > 1:
        suspicious_indicators.append("Amenazas")
    if url_features['suspicious_domain'] > 0:
        suspicious_indicators.append("Dominio sospechoso")
    
    return {
        'es_phishing': bool(prediction),
        'probabilidad_phishing': float(probability),
        'clasificacion': 'PHISHING' if prediction == 1 else 'LEGÍTIMO',
        'nivel_riesgo': 'ALTO' if probability > 0.8 else 'MEDIO' if probability > 0.5 else 'BAJO',
        'idioma_detectado': language,
        'indicadores_sospechosos': suspicious_indicators,
        'caracteristicas_extraidas': all_features
    }



idioma_detectado = detect_language(texto_email)
resultado = detect_phishing(texto_email)

print(f"\n🔍 RESULTADO: {resultado['clasificacion']}")
print(f"📊 Probabilidad de phishing: {resultado['probabilidad_phishing']:.1%}")
print(f"⚠️  Nivel de riesgo: {resultado['nivel_riesgo']}")
print(f"🌐 Idioma detectado: {resultado['idioma_detectado']}")
    
if resultado['indicadores_sospechosos']:
    print(f"🚨 Indicadores sospechosos: {', '.join(resultado['indicadores_sospechosos'])}")
    
# Mostrar características clave
features = resultado['caracteristicas_extraidas']
print(f"\n📈 Análisis detallado:")
print(f"   - URLs encontradas: {features['url_count']}")
print(f"   - Palabras de urgencia: {features['urgency_words']}")
print(f"   - Palabras de acción: {features['action_words']}")
print(f"   - Signos de exclamación: {features['exclamation_count']}")


probabilidad_str = str(resultado['probabilidad_phishing'])[:5]
