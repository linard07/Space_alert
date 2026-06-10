# 🛰️ SPACE ALERT 2026

> **Sistema Inteligente de Monitoramento Climático e Prevenção de Desastres Naturais baseado em Dados Espaciais e Visão Computacional.**

---

## 👥 Integrantes

| Nome | RM |
|------|----|
| Ali Andrea Mamani Molle | 558052 |
| Guilherme Linardi F. Rgozzi | 555768 |
| Lucas Vasquez Silva | 555159 |

---

## 📌 Descrição do Projeto

O **Space Alert 2026** é um aplicativo de monitoramento climático em tempo real que utiliza dados captados por **webcam** e os processa via pipeline de **Visão Computacional** para identificar riscos ambientais e emitir alertas preventivos em três níveis de severidade.

A solução endereça um problema crescente no Brasil: eventos climáticos extremos estão se tornando mais frequentes — em 2025 foram registrados **7 ondas de calor, 689 mm de chuva em um único município em 24h, 503 municípios em seca severa** — e a falta de alertas rápidos resulta em impactos humanos, ambientais e econômicos significativos.

O sistema se conecta à **indústria espacial** ao simular o uso de dados capturados por satélites (modelados via análise de imagem e câmera), utilizando as mesmas técnicas de análise espectral e detecção de anomalias empregadas em sensoriamento remoto.

---

## 🎯 Objetivos

### Objetivo de Negócio
Fornecer a população, Defesa Civil e órgãos públicos uma ferramenta acessível de **detecção precoce de riscos climáticos** (incêndios, alagamentos, neblina severa, fumaça) por meio de visão computacional, reduzindo o tempo de resposta a emergências e os impactos causados por desastres naturais.

### Objetivo Técnico
Desenvolver um pipeline de inferência em Python com:
- Análise de frame em tempo real via OpenCV (brilho, desfoque, análise espectral)
- Detecção de objetos via **YOLOv8** (fumaça, fogo, veículos, pessoas em risco)
- Detecção de pose humana via **MediaPipe**
- Sistema de avaliação de risco com suavização temporal
- HUD (Heads-Up Display) com métricas e alertas sobrepostos ao vídeo
- Tratamento robusto de falhas de hardware e queda de frames

---

## 🏗️ Arquitetura da Solução

```
┌─────────────────────────────────────────────────────────────┐
│                     SPACE ALERT 2026                        │
│                  Pipeline de Inferência                     │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │  Fonte de Vídeo│  ← Webcam / arquivo de vídeo
       │  cv2.VideoCapture│
       └───────┬────────┘
               │  frame (BGR ndarray)
       ┌───────▼────────────────────────────────┐
       │          FrameAnalyzer (OpenCV)         │
       │  • Brilho médio (luminância)            │
       │  • Blur score (Laplaciana)              │
       │  • Red ratio (possível fogo/incêndio)   │
       │  • Blue ratio (possível alagamento)     │
       └───────┬────────────────────────────────┘
               │ métricas
       ┌───────▼────────────────────────────────┐
       │     Detecção de Objetos (YOLO)          │
       │  yolov8n.pt – classes relevantes:       │
       │  fire, smoke, person, boat, car         │
       └───────┬────────────────────────────────┘
               │ detecções + confiança
       ┌───────▼────────────────────────────────┐
       │    Detecção de Pose (MediaPipe)         │
       │  Pose landmarks → pessoa em risco       │
       └───────┬────────────────────────────────┘
               │
       ┌───────▼────────────────────────────────┐
       │         RiskEvaluator                   │
       │  • Consolida métricas + detecções       │
       │  • Suavização temporal (deque 15f)      │
       │  • Emite nível 0-3 + motivos            │
       └───────┬────────────────────────────────┘
               │ nível de risco
       ┌───────▼────────────────────────────────┐
       │         HUDRenderer (OpenCV)            │
       │  • Barra superior (título + FPS)        │
       │  • Painel de alerta colorido            │
       │  • Métricas numéricas                   │
       │  • Indicador pulsante de nível          │
       │  • Bounding boxes YOLO                  │
       │  • Landmarks MediaPipe                  │
       └───────┬────────────────────────────────┘
               │
       ┌───────▼────────────────────────────────┐
       │  cv2.imshow  →  Janela em tempo real    │
       └────────────────────────────────────────┘
```

---

## 🚨 Sistema de Alertas

| Nível | Nome | Cor | Condição |
|-------|------|-----|----------|
| **0** | NORMAL | 🟢 Verde | Nenhuma anomalia detectada |
| **1** | ATENÇÃO | 🟡 Amarelo | Variações leves: brilho alto, leve avermelhamento, blur moderado |
| **2** | ALERTA | 🟠 Laranja | Luminosidade baixa (fumaça/neblina), alagamento detectado, blur severo |
| **3** | EMERGÊNCIA | 🔴 Vermelho | Alta concentração vermelha (incêndio), detecção YOLO de fogo/fumaça |

---

## 📚 Bibliotecas Utilizadas (Stack)

| Biblioteca | Versão mínima | Uso |
|-----------|--------------|-----|
| `opencv-python` | 4.9.0 | Captura de vídeo, análise de frame, HUD, rendering |
| `numpy` | 1.26.0 | Operações matriciais, análise de canais de cor |
| `ultralytics` | 8.2.0 | Modelo YOLOv8 para detecção de objetos |
| `mediapipe` | 0.10.9 | Detecção de pose humana (landmarks) |
| `Pillow` | 10.3.0 | Dependência interna do Ultralytics |

---

## ⚙️ Explicação do Pipeline de Visão Computacional

### 1. Captura e Tratamento de Falhas
```python
ret, frame = cap.read()
if not ret:
    # reconecta automaticamente até MAX_FAILURES tentativas
```
O pipeline monitora frames inválidos consecutivos e tenta reconectar automaticamente ao hardware, com log de cada evento.

### 2. FrameAnalyzer – Análise Espectral
```
brightness = mean(gray)           # luminância média
blur_score = var(Laplacian(gray)) # nitidez do frame
red_ratio  = pixels_R >> G,B / total  # proporção avermelhada
blue_ratio = pixels_B >> R,G / total  # proporção azulada
```

### 3. Detecção YOLO
O modelo `yolov8n.pt` é executado a cada frame. Classes com risco mapeado geram elevação automática do nível de alerta com base na confiança da detecção (limiar: 0.45).

### 4. Suavização Temporal
Para evitar alertas instáveis (flickering), os últimos 15 níveis de risco são armazenados em `deque` e a média é utilizada como nível final.

### 5. HUD Rendering
Todos os elementos visuais são sobrepostos via OpenCV com transparência (`addWeighted`), sem dependências externas de UI.

---

## 🚀 Instruções de Execução

### Pré-requisitos
- Python **3.10** ou **3.11**
- Webcam USB ou integrada **ou** arquivo de vídeo `.mp4`
- Git

### 1. Clone o repositório
```bash
git clone https://github.com/SEU_USUARIO/space-alert-2026.git
cd space-alert-2026
```

### 2. Crie e ative o ambiente virtual
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute

**Webcam padrão (índice 0):**
```bash
python space_alert.py
```

**Webcam alternativa:**
```bash
python space_alert.py --source 1
```

**Arquivo de vídeo:**
```bash
python space_alert.py --source caminho/para/video.mp4
```

**Modelo YOLO diferente:**
```bash
python space_alert.py --yolo-model yolov8s.pt
```

### 5. Controles durante execução
| Tecla | Ação |
|-------|------|
| `q` | Encerra o pipeline |
| `s` | Salva screenshot com timestamp |

---

## 📂 Estrutura do Repositório

```
space-alert-2026/
├── space_alert.py       # Script principal do pipeline
├── requirements.txt     # Dependências Python
└── README.md            # Este arquivo
```

---

## 🌍 Relação com os ODS

- **ODS 11** – Cidades e Comunidades Sustentáveis: alertas preventivos protegem a população
- **ODS 13** – Ação Contra a Mudança Global do Clima: monitoramento em tempo real de riscos climáticos
- **ODS 9** – Indústria, Inovação e Infraestrutura: uso de IA e tecnologias espaciais para valor social

---

Link do Video explicativo:
https://youtu.be/9UlKdFQs63k

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos no âmbito da Global Solution 2026 – FIAP.
