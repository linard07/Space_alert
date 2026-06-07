"""
=============================================================================
SPACE ALERT 2026 - Sistema Inteligente de Monitoramento Climático
=============================================================================
Autores:
  - Ali Andrea Mamani Molle      (RM 558052)
  - Guilherme Linardi F. Rgozzi  (RM 555768)
  - Lucas Vasquez Silva          (RM 555159)

Descrição:
  Pipeline de Visão Computacional para monitoramento ambiental em tempo real
  via webcam. Detecta condições de risco climático (fumaça, enchentes, fogo,
  oclusão severa por chuva) usando YOLO, MediaPipe e OpenCV, emitindo alertas
  em três níveis (Atenção / Alerta / Emergência).

Dependências: ver requirements.txt
=============================================================================
"""

import sys
import time
import logging
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ── MediaPipe ──────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    logging.warning("MediaPipe não encontrado – detecção de pose desativada.")

# ── Ultralytics YOLO ───────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics não encontrado – detecção YOLO desativada.")

# ── Configuração de log ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SpaceAlert")

# =============================================================================
# CONSTANTES
# =============================================================================

# Classes YOLO relevantes para detecção de risco
RISK_CLASSES = {
    "fire":   3,   # nível de risco base para fogo
    "smoke":  2,
    "person": 1,   # pessoa em área de risco
    "car":    1,
    "boat":   2,   # barco em área inundada
    "flood":  3,
}

# Limiares de análise de imagem (OpenCV puro)
BRIGHTNESS_LOW   = 40    # lux aproximado – escuridão / fumaça densa
BRIGHTNESS_HIGH  = 220   # superexposição / reflexo de incêndio
BLUR_THRESHOLD   = 80    # laplaciana – imagem embaçada (chuva intensa)
RED_RATIO_THRESH = 0.25  # fração de pixels "avermelhados" (fogo/incêndio)
BLUE_RATIO_THRESH = 0.20 # fração de pixels "azulados" (alagamento)

# Cores dos níveis de alerta (BGR)
LEVEL_COLORS = {
    0: (0,   200,  0),    # verde  – OK
    1: (0,   200, 255),   # amarelo – Atenção
    2: (0,   140, 255),   # laranja – Alerta
    3: (0,    0,  220),   # vermelho – Emergência
}

LEVEL_LABELS = {
    0: "NORMAL",
    1: "NIVEL 1 - ATENCAO",
    2: "NIVEL 2 - ALERTA",
    3: "NIVEL 3 - EMERGENCIA",
}

# =============================================================================
# MÓDULOS DE ANÁLISE
# =============================================================================

class FrameAnalyzer:
    """Análise de frame via OpenCV (sem modelo externo)."""

    def analyze(self, frame: np.ndarray) -> dict:
        """
        Retorna um dict com métricas do frame:
          brightness, blur_score, red_ratio, blue_ratio
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Brilho médio
        brightness = float(np.mean(gray))

        # Nitidez (variância do Laplaciano – quanto menor, mais embaçado)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Análise de canais de cor
        b, g, r = cv2.split(frame)
        total_pixels = frame.shape[0] * frame.shape[1]

        # Pixels avermelhados: R muito acima de G e B
        red_mask  = (r.astype(int) - g.astype(int) > 40) & \
                    (r.astype(int) - b.astype(int) > 40) & \
                    (r > 120)
        red_ratio = float(np.sum(red_mask)) / total_pixels

        # Pixels azulados: B muito acima de R e G
        blue_mask  = (b.astype(int) - r.astype(int) > 30) & \
                     (b.astype(int) - g.astype(int) > 20) & \
                     (b > 100)
        blue_ratio = float(np.sum(blue_mask)) / total_pixels

        return {
            "brightness":  brightness,
            "blur_score":  blur_score,
            "red_ratio":   red_ratio,
            "blue_ratio":  blue_ratio,
        }


class RiskEvaluator:
    """
    Consolida métricas de análise + YOLO + MediaPipe e
    determina o nível de risco (0-3).
    """

    def __init__(self):
        # Histórico dos últimos N frames para suavização
        self._history: deque = deque(maxlen=15)

    def evaluate(self, metrics: dict, yolo_detections: list) -> tuple[int, list[str]]:
        """
        Retorna (nível_risco: int, motivos: list[str]).
        """
        reasons: list[str] = []
        level = 0

        b   = metrics["brightness"]
        bl  = metrics["blur_score"]
        rr  = metrics["red_ratio"]
        br  = metrics["blue_ratio"]

        # ── Análise de brilho ──────────────────────────────────────────────
        if b < BRIGHTNESS_LOW:
            reasons.append(f"Baixa luminosidade ({b:.0f}) – possível fumaça/neblina")
            level = max(level, 2)
        elif b > BRIGHTNESS_HIGH:
            reasons.append(f"Alta luminosidade ({b:.0f}) – reflexo intenso/incêndio")
            level = max(level, 1)

        # ── Análise de desfoque (chuva intensa / neblina) ──────────────────
        if bl < BLUR_THRESHOLD:
            reasons.append(f"Imagem embaçada (blur={bl:.1f}) – chuva/neblina intensa")
            level = max(level, 1)

        # ── Análise de cor avermelhada (fogo/incêndio) ─────────────────────
        if rr > RED_RATIO_THRESH:
            pct = rr * 100
            reasons.append(f"Alta concentração vermelha ({pct:.1f}%) – possível incêndio")
            level = max(level, 3)
        elif rr > RED_RATIO_THRESH * 0.5:
            reasons.append("Indício leve de coloração avermelhada")
            level = max(level, 1)

        # ── Análise de cor azulada (alagamento/enchente) ───────────────────
        if br > BLUE_RATIO_THRESH:
            pct = br * 100
            reasons.append(f"Alta concentração azul ({pct:.1f}%) – possível alagamento")
            level = max(level, 2)

        # ── Detecções YOLO ─────────────────────────────────────────────────
        for det in yolo_detections:
            cls_name = det.get("class_name", "").lower()
            conf     = det.get("confidence", 0.0)
            risk_lvl = RISK_CLASSES.get(cls_name, 0)
            if conf > 0.45 and risk_lvl > 0:
                reasons.append(
                    f"YOLO detectou '{cls_name}' (conf={conf:.2f}) – risco nível {risk_lvl}"
                )
                level = max(level, risk_lvl)

        # ── Suavização temporal ────────────────────────────────────────────
        self._history.append(level)
        smoothed = int(round(np.mean(self._history)))

        if not reasons:
            reasons.append("Condições dentro do normal")

        return smoothed, reasons


# =============================================================================
# OVERLAY DE HUD
# =============================================================================

class HUDRenderer:
    """Renderiza o HUD (Heads-Up Display) no frame."""

    # Fonte monoespaçada para estética sci-fi
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_MONO  = cv2.FONT_HERSHEY_PLAIN

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self._start_time = time.time()
        self._fps_buffer: deque = deque(maxlen=30)
        self._last_tick = time.time()

    def tick(self):
        """Registra um frame para cálculo de FPS."""
        now = time.time()
        self._fps_buffer.append(1.0 / max(now - self._last_tick, 1e-6))
        self._last_tick = now

    @property
    def fps(self) -> float:
        return float(np.mean(self._fps_buffer)) if self._fps_buffer else 0.0

    def render(
        self,
        frame: np.ndarray,
        level: int,
        reasons: list[str],
        metrics: dict,
        yolo_count: int,
    ) -> np.ndarray:
        """Desenha o HUD completo no frame e retorna o frame modificado."""
        overlay = frame.copy()
        color   = LEVEL_COLORS[level]
        label   = LEVEL_LABELS[level]

        # ── Barra superior ─────────────────────────────────────────────────
        cv2.rectangle(overlay, (0, 0), (self.w, 48), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        title = "SPACE ALERT 2026"
        cv2.putText(frame, title, (10, 32),
                    self.FONT, 0.9, (200, 200, 200), 2, cv2.LINE_AA)

        fps_txt = f"FPS: {self.fps:.1f}"
        (fw, _), _ = cv2.getTextSize(fps_txt, self.FONT, 0.7, 2)
        cv2.putText(frame, fps_txt, (self.w - fw - 10, 32),
                    self.FONT, 0.7, (160, 220, 160), 2, cv2.LINE_AA)

        # ── Painel de alerta (canto inferior esquerdo) ─────────────────────
        panel_h = 140
        panel_w = 420
        panel_y = self.h - panel_h - 10
        panel_x = 10

        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (15, 15, 15), -1)
        cv2.addWeighted(panel_overlay, 0.70, frame, 0.30, 0, frame)

        # Borda colorida do painel
        cv2.rectangle(frame,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      color, 2)

        # Label do nível
        cv2.putText(frame, label,
                    (panel_x + 10, panel_y + 28),
                    self.FONT, 0.70, color, 2, cv2.LINE_AA)

        # Motivos (máx. 3 linhas)
        for i, reason in enumerate(reasons[:3]):
            txt = reason[:58] + ".." if len(reason) > 60 else reason
            cv2.putText(frame, f"• {txt}",
                        (panel_x + 10, panel_y + 55 + i * 26),
                        self.FONT_MONO, 1.1, (200, 200, 200), 1, cv2.LINE_AA)

        # ── Métricas (canto inferior direito) ──────────────────────────────
        met_x = self.w - 230
        met_y = self.h - 130

        metrics_lines = [
            f"Brilho  : {metrics['brightness']:5.1f}",
            f"Blur    : {metrics['blur_score']:6.1f}",
            f"Cor R   : {metrics['red_ratio']*100:4.1f}%",
            f"Cor B   : {metrics['blue_ratio']*100:4.1f}%",
            f"YOLO det: {yolo_count:3d}",
        ]

        # Fundo semi-transparente
        met_overlay = frame.copy()
        cv2.rectangle(met_overlay,
                      (met_x - 5, met_y - 16),
                      (self.w - 5, met_y + len(metrics_lines) * 22 + 4),
                      (15, 15, 15), -1)
        cv2.addWeighted(met_overlay, 0.65, frame, 0.35, 0, frame)

        for i, line in enumerate(metrics_lines):
            cv2.putText(frame, line,
                        (met_x, met_y + i * 22),
                        self.FONT_MONO, 1.0, (170, 220, 170), 1, cv2.LINE_AA)

        # ── Indicador de nível (círculo pulsante) ──────────────────────────
        cx, cy = self.w - 50, 80
        pulse_r = 22 + int(4 * abs(np.sin(time.time() * 3)))
        cv2.circle(frame, (cx, cy), pulse_r + 6, (30, 30, 30), -1)
        cv2.circle(frame, (cx, cy), pulse_r, color, -1)
        cv2.circle(frame, (cx, cy), pulse_r + 6, color, 2)

        # ── Timestamp ──────────────────────────────────────────────────────
        elapsed = int(time.time() - self._start_time)
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, ts,
                    (self.w // 2 - 40, self.h - 10),
                    self.FONT_MONO, 1.1, (120, 120, 120), 1, cv2.LINE_AA)

        return frame


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

class SpaceAlertPipeline:
    """
    Orquestra a captura de vídeo, análise de frames e exibição do HUD.
    """

    def __init__(self, source: int | str = 0, yolo_model: str = "yolov8n.pt"):
        self.source     = source
        self.yolo_model = yolo_model

        self._cap:      cv2.VideoCapture | None = None
        self._yolo:     object | None = None
        self._mp_pose:  object | None = None

        self.analyzer  = FrameAnalyzer()
        self.evaluator = RiskEvaluator()
        self.hud:       HUDRenderer | None = None

        self._running   = False
        self._skip_yolo = not YOLO_AVAILABLE

    # ── Inicialização ──────────────────────────────────────────────────────

    def _init_capture(self) -> bool:
        """Abre a captura de vídeo com tratamento de falhas de hardware."""
        log.info("Iniciando captura de vídeo (fonte: %s)…", self.source)
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            log.error("Falha ao abrir câmera/vídeo. Verifique o hardware.")
            return False

        # Preferências de resolução
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        log.info("Captura iniciada – %dx%d @ %.0f fps",
                 self._cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                 self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                 self._cap.get(cv2.CAP_PROP_FPS))
        return True

    def _init_yolo(self):
        """Carrega modelo YOLO (baixa automaticamente se necessário)."""
        if self._skip_yolo:
            return
        try:
            log.info("Carregando modelo YOLO '%s'…", self.yolo_model)
            self._yolo = YOLO(self.yolo_model)
            log.info("YOLO pronto.")
        except Exception as exc:
            log.warning("Não foi possível carregar YOLO: %s – seguindo sem ele.", exc)
            self._skip_yolo = True

    def _init_mediapipe(self):
        """Inicializa MediaPipe Pose para detecção de pessoas."""
        if not MP_AVAILABLE:
            return
        try:
            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
            )
            log.info("MediaPipe Pose pronto.")
        except Exception as exc:
            log.warning("MediaPipe falhou: %s", exc)
            self._mp_pose = None

    # ── Loop de inferência ─────────────────────────────────────────────────

    def _run_yolo(self, frame: np.ndarray) -> list[dict]:
        """Executa inferência YOLO e retorna lista de detecções."""
        if self._skip_yolo or self._yolo is None:
            return []
        try:
            results = self._yolo(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                cls_id   = int(box.cls[0])
                conf     = float(box.conf[0])
                cls_name = results.names.get(cls_id, "unknown")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class_id":   cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox":       (x1, y1, x2, y2),
                })
            return detections
        except Exception as exc:
            log.debug("Erro YOLO no frame: %s", exc)
            return []

    def _draw_yolo_boxes(self, frame: np.ndarray, detections: list[dict]):
        """Desenha bounding boxes das detecções YOLO."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf     = det["confidence"]
            risk_lvl = RISK_CLASSES.get(cls_name.lower(), 0)
            color    = LEVEL_COLORS.get(risk_lvl, (180, 180, 180))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f"{cls_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(tag) * 9, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1.1, (10, 10, 10), 1, cv2.LINE_AA)

    def _draw_pose(self, frame: np.ndarray):
        """Sobrepõe landmarks de pose (MediaPipe) se disponível."""
        if self._mp_pose is None:
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._mp_pose.process(rgb)
            if res.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                )
        except Exception as exc:
            log.debug("Erro MediaPipe no frame: %s", exc)

    # ── Execução ───────────────────────────────────────────────────────────

    def run(self):
        """Inicia o pipeline principal."""
        if not self._init_capture():
            sys.exit(1)

        self._init_yolo()
        self._init_mediapipe()

        # Detecta dimensões reais após abertura
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.hud = HUDRenderer(w, h)

        self._running = True
        consecutive_failures = 0
        MAX_FAILURES = 10

        log.info("Pipeline iniciado. Pressione 'q' para sair, 's' para salvar screenshot.")

        try:
            while self._running:
                # ── Leitura do frame ───────────────────────────────────────
                ret, frame = self._cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    log.warning(
                        "Frame inválido (%d/%d). Tentando reconectar…",
                        consecutive_failures, MAX_FAILURES
                    )
                    if consecutive_failures >= MAX_FAILURES:
                        log.error("Muitas falhas consecutivas. Encerrando.")
                        break
                    # Tentativa de reconexão
                    self._cap.release()
                    time.sleep(0.5)
                    self._cap = cv2.VideoCapture(self.source)
                    continue

                consecutive_failures = 0  # reset ao receber frame válido
                self.hud.tick()

                # ── Análise do frame ───────────────────────────────────────
                metrics    = self.analyzer.analyze(frame)
                detections = self._run_yolo(frame)
                level, reasons = self.evaluator.evaluate(metrics, detections)

                # ── Renderização ───────────────────────────────────────────
                self._draw_yolo_boxes(frame, detections)
                self._draw_pose(frame)
                frame = self.hud.render(frame, level, reasons, metrics, len(detections))

                # ── Exibição ───────────────────────────────────────────────
                cv2.imshow("SPACE ALERT 2026 – Monitoramento Climático", frame)

                # ── Teclas ─────────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    log.info("Encerrando por comando do usuário.")
                    break
                elif key == ord("s"):
                    ts_file = time.strftime("screenshot_%Y%m%d_%H%M%S.png")
                    cv2.imwrite(ts_file, frame)
                    log.info("Screenshot salvo: %s", ts_file)

        except KeyboardInterrupt:
            log.info("Interrompido via teclado.")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Libera todos os recursos de hardware e janelas."""
        log.info("Liberando recursos…")
        if self._cap and self._cap.isOpened():
            self._cap.release()
        if self._mp_pose:
            self._mp_pose.close()
        cv2.destroyAllWindows()
        log.info("Encerrado com sucesso.")


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SPACE ALERT 2026 – Pipeline de Monitoramento Climático por Visão Computacional"
    )
    parser.add_argument(
        "--source", default=0,
        help="Fonte de vídeo: índice da webcam (0, 1…) ou caminho de arquivo de vídeo"
    )
    parser.add_argument(
        "--yolo-model", default="yolov8n.pt",
        help="Modelo YOLO a utilizar (ex: yolov8n.pt, yolov8s.pt)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Converte source para int se for dígito
    source = int(args.source) if str(args.source).isdigit() else args.source

    pipeline = SpaceAlertPipeline(source=source, yolo_model=args.yolo_model)
    pipeline.run()
