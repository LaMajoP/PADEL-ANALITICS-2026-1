"""
Analizador de Video para Biomecánica del Tenis
────────────────────────────────────────────────
Flujo:
  1. Abrir video → navegar frames con botones / flechas
  2. Guardar frames de interés → se nombran frame0, frame1, frame2…
  3. Botón "Listo" → modo anotación sobre frames guardados
  4. Marcar 5 puntos (cabeza, hombro_d, codo, muñeca, punta_raqueta)
  5. Panel lateral muestra ángulos en tiempo real
  6. Guardar JSON en angulos2/resultados/<video>/json/

Dependencias:  pip install opencv-python pillow
"""

import os
import json
import math
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk

# Carpeta de salida: angulos2/resultados/<nombre_video>/{frames,json}
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_ROOT = os.path.join(SCRIPT_DIR, "resultados")


def output_dir_for_video(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(RESULTADOS_ROOT, stem)


# ─────────────────────────────────────────────────────────
#  CONFIGURACIÓN DE ÁNGULOS (según tabla de referencia)
# ─────────────────────────────────────────────────────────
ANGLE_DEFINITIONS = [
    ("Codo",              "muñeca",        "codo",    "hombro_d"),
    ("Hombro",            "codo",          "hombro_d","cabeza"),
    ("Incl. Raqueta",     "punta_raqueta", "muñeca",  "codo"),
    ("Raqueta-Hombro",    "punta_raqueta", "muñeca",  "hombro_d"),
    ("Muñeca-Cabeza",     "muñeca",        "codo",    "cabeza"),
]

POINT_ORDER   = ["cabeza", "hombro_d", "codo", "muñeca", "punta_raqueta"]
POINT_LABELS  = {
    "cabeza":        "Cabeza",
    "hombro_d":      "Hombro D",
    "codo":          "Codo",
    "muñeca":        "Muñeca",
    "punta_raqueta": "Raqueta",
}
POINT_COLORS  = {
    "cabeza":        "#FF6B6B",
    "hombro_d":      "#FFD93D",
    "codo":          "#6BCB77",
    "muñeca":        "#4D96FF",
    "punta_raqueta": "#FF922B",
}

# Paleta de colores de la aplicación
BG_DARK    = "#0E0E16"
BG_MID     = "#181826"
BG_PANEL   = "#1D1D2E"
ACCENT     = "#E94560"
ACCENT2    = "#4D96FF"
TEXT_MAIN  = "#F0F0F8"
TEXT_DIM   = "#888899"


# ─────────────────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────────────────
def calc_angle(p1, vertex, p2):
    """Ángulo en 'vertex' formado por los vectores vertex→p1 y vertex→p2."""
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    m1 = math.hypot(*v1)
    m2 = math.hypot(*v2)
    if m1 == 0 or m2 == 0:
        return None
    cos_a = max(-1.0, min(1.0, (v1[0]*v2[0] + v1[1]*v2[1]) / (m1 * m2)))
    return math.degrees(math.acos(cos_a))


def cv2_to_pil(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


# ─────────────────────────────────────────────────────────
#  APLICACIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Analizador Biomecánico · Tenis")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1280x780")
        self.root.minsize(900, 620)

        # ── Estado de video ──
        self.cap             = None
        self.total_frames    = 0
        self.current_idx     = 0
        self.current_bgr     = None
        self.output_dir      = None
        self.frame_counter   = 0          # contador global de frames guardados
        self.saved_frames    = {}         # {name: path}

        # ── Estado de anotación ──
        self.annotation_mode       = False
        self.ann_frame_names       = []   # nombres ordenados
        self.ann_current_name      = None
        self.ann_img_rgb           = None # numpy array RGB
        self.ann_list_idx          = 0
        self.points                = {}   # {label: (x_orig, y_orig)}
        self.next_point_idx        = 0

        # ── Variables de layout de canvas ──
        self.canvas_offset = (0, 0)
        self.canvas_scale  = 1.0
        self.orig_size     = (1, 1)

        self._build_ui()
        self._bind_keys()

    # ─────────────────────────────────────────
    #  CONSTRUCCIÓN DE INTERFAZ
    # ─────────────────────────────────────────
    def _build_ui(self):
        # ── Barra superior ──
        topbar = tk.Frame(self.root, bg=BG_MID, height=54)
        topbar.pack(fill=tk.X, side=tk.TOP)
        topbar.pack_propagate(False)

        self._btn(topbar, "📂  Abrir Video", self._open_video,
                  bg="#253B6E", pad=(18,0)).pack(side=tk.LEFT, padx=12, pady=8)

        self.lbl_frame = tk.Label(topbar, text="Sin video", bg=BG_MID,
                                   fg=TEXT_DIM, font=("Courier New", 11))
        self.lbl_frame.pack(side=tk.LEFT, padx=20)

        self.lbl_saved = tk.Label(topbar, text="", bg=BG_MID,
                                   fg=ACCENT2, font=("Courier New", 11, "bold"))
        self.lbl_saved.pack(side=tk.LEFT, padx=10)

        # ── Área central ──
        center = tk.Frame(self.root, bg=BG_DARK)
        center.pack(fill=tk.BOTH, expand=True)

        # Canvas de imagen
        self.canvas = tk.Canvas(center, bg="#090910", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Panel lateral
        self._build_side_panel(center)

        # ── Barra inferior (controles) ──
        self.ctrl_bar = tk.Frame(self.root, bg=BG_MID, height=60)
        self.ctrl_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.ctrl_bar.pack_propagate(False)
        self._build_video_controls()

    def _build_side_panel(self, parent):
        panel = tk.Frame(parent, bg=BG_PANEL, width=230)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        panel.pack_propagate(False)
        self.side_panel = panel

        tk.Label(panel, text="ÁNGULOS", bg=BG_PANEL, fg=ACCENT,
                  font=("Courier New", 13, "bold")).pack(pady=(16, 8))

        sep = tk.Frame(panel, bg=ACCENT, height=1)
        sep.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.angle_vars = {}
        for name, *_ in ANGLE_DEFINITIONS:
            card = tk.Frame(panel, bg="#252538", relief=tk.FLAT)
            card.pack(fill=tk.X, padx=10, pady=4, ipady=6)
            tk.Label(card, text=name.upper(), bg="#252538", fg=TEXT_DIM,
                      font=("Courier New", 8, "bold")).pack()
            var = tk.StringVar(value="—°")
            tk.Label(card, textvariable=var, bg="#252538", fg=TEXT_MAIN,
                      font=("Courier New", 20, "bold")).pack()
            self.angle_vars[name] = var

        tk.Frame(panel, bg="#252538", height=1).pack(fill=tk.X, padx=12, pady=10)

        self.lbl_next_point = tk.Label(panel, text="", bg=BG_PANEL, fg="#FFD93D",
                                        font=("Courier New", 10, "bold"),
                                        justify=tk.CENTER, wraplength=200)
        self.lbl_next_point.pack(pady=4)

        self.lbl_point_count = tk.Label(panel, text="", bg=BG_PANEL, fg=TEXT_DIM,
                                         font=("Courier New", 9))
        self.lbl_point_count.pack()

    def _build_video_controls(self):
        """Controles del modo video (navegación + guardar + listo)."""
        for w in self.ctrl_bar.winfo_children():
            w.destroy()

        self._btn(self.ctrl_bar, "◀  Anterior", self._prev_frame,
                  bg="#253B6E").pack(side=tk.LEFT, padx=(12, 4), pady=10)

        self._btn(self.ctrl_bar, "Siguiente  ▶", self._next_frame,
                  bg="#253B6E").pack(side=tk.LEFT, padx=4, pady=10)

        self._btn(self.ctrl_bar, "💾  Guardar Frame", self._save_frame,
                  bg="#1A4731").pack(side=tk.LEFT, padx=14, pady=10)

        self.btn_done = self._btn(self.ctrl_bar, "✅  Listo", self._enter_annotation_mode,
                                   bg="#7B1D3A")
        self.btn_done.pack(side=tk.LEFT, padx=4, pady=10)

    def _build_annotation_controls(self):
        """Controles del modo anotación (lista + reset + guardar JSON)."""
        for w in self.ctrl_bar.winfo_children():
            w.destroy()

        tk.Label(self.ctrl_bar, text="Frames guardados:", bg=BG_MID,
                  fg=TEXT_DIM, font=("Courier New", 9)).pack(side=tk.LEFT, padx=(12, 4), pady=14)

        list_frame = tk.Frame(self.ctrl_bar, bg=BG_MID)
        list_frame.pack(side=tk.LEFT, padx=4, pady=8)

        sb = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(
            list_frame, height=2, width=18,
            bg="#1A1A2C", fg=TEXT_MAIN, selectbackground=ACCENT,
            font=("Courier New", 9), yscrollcommand=sb.set,
            activestyle="none", relief=tk.FLAT, borderwidth=0,
        )
        sb.config(command=self.listbox.yview)
        self.listbox.pack(side=tk.LEFT)
        sb.pack(side=tk.LEFT, fill=tk.Y)

        self._refresh_listbox()
        if self.ann_frame_names:
            self.listbox.selection_set(0)

        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        self._btn(self.ctrl_bar, "🔄 Reset", self._reset_points,
                  bg="#5A3F7A").pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_save_json = self._btn(self.ctrl_bar, "💾  Guardar JSON",
                                        self._save_json, bg="#1A4731")
        self.btn_save_json.pack(side=tk.LEFT, padx=4, pady=10)

    # ─────────────────────────────────────────
    #  HELPERS UI
    # ─────────────────────────────────────────
    def _btn(self, parent, text, command, bg="#253B6E", pad=(14, 0)):
        return tk.Button(
            parent, text=text, command=command,
            bg=bg, fg="#000000", activebackground=bg, activeforeground="#000000",
            relief=tk.FLAT, padx=14, pady=9,
            font=("Courier New", 10, "bold"), cursor="hand2",
            bd=0, highlightthickness=0,
        )

    def _flash(self, widget, color_ok, color_orig, ms=400):
        widget.config(bg=color_ok)
        self.root.after(ms, lambda: widget.config(bg=color_orig))

    # ─────────────────────────────────────────
    #  KEY BINDINGS
    # ─────────────────────────────────────────
    def _bind_keys(self):
        self.root.bind("<Left>",  lambda e: self._prev_frame())
        self.root.bind("<Right>", lambda e: self._next_frame())

    # ─────────────────────────────────────────
    #  APERTURA DE VIDEO
    # ─────────────────────────────────────────
    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[
                ("Archivos de video", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not path:
            return

        if self.cap:
            self.cap.release()

        self.cap           = cv2.VideoCapture(path)
        self.total_frames  = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.current_idx   = 0
        self.output_dir    = output_dir_for_video(path)
        self.frame_counter = 0
        self.saved_frames  = {}
        self.annotation_mode = False

        self.lbl_saved.config(text=f"📁 {self.output_dir}")
        self._build_video_controls()
        self._show_video_frame(0)

    # ─────────────────────────────────────────
    #  NAVEGACIÓN DE FRAMES DE VIDEO
    # ─────────────────────────────────────────
    def _show_video_frame(self, idx):
        if not self.cap:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_idx  = idx
        self.current_bgr  = frame
        self.lbl_frame.config(
            text=f"Frame {idx:05d} / {self.total_frames - 1:05d}"
        )
        self._render_image(cv2_to_pil(frame))

    def _prev_frame(self):
        if self.annotation_mode:
            self._navigate_list(-1)
        elif self.cap:
            self._show_video_frame(self.current_idx - 1)

    def _next_frame(self):
        if self.annotation_mode:
            self._navigate_list(1)
        elif self.cap:
            self._show_video_frame(self.current_idx + 1)

    # ─────────────────────────────────────────
    #  GUARDAR FRAME
    # ─────────────────────────────────────────
    def _save_frame(self):
        if self.current_bgr is None:
            messagebox.showinfo("Sin frame", "Carga un video primero.")
            return

        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        name  = f"frame{self.frame_counter}"
        fpath = os.path.join(frames_dir, f"{name}.png")
        cv2.imwrite(fpath, self.current_bgr)

        self.saved_frames[name] = fpath
        self.frame_counter += 1

        self.lbl_saved.config(text=f"💾 {self.frame_counter} frame(s) guardado(s)")
        # Feedback visual en el canvas
        self.canvas.create_rectangle(0, 0, 999, 999, fill="#1A4731",
                                      stipple="gray25", tags="flash")
        self.root.after(250, lambda: self.canvas.delete("flash"))

    # ─────────────────────────────────────────
    #  MODO ANOTACIÓN
    # ─────────────────────────────────────────
    def _enter_annotation_mode(self):
        if not self.saved_frames:
            messagebox.showwarning("Sin frames", "Guarda al menos un frame primero.")
            return

        self.annotation_mode  = True
        self.ann_frame_names  = sorted(
            self.saved_frames.keys(),
            key=lambda n: int(n.replace("frame", "") or 0)
        )
        self.ann_list_idx = 0

        self._build_annotation_controls()
        self._load_ann_frame(self.ann_frame_names[0])

    def _navigate_list(self, delta):
        if not self.ann_frame_names:
            return
        new_idx = max(0, min(self.ann_list_idx + delta, len(self.ann_frame_names) - 1))
        self.ann_list_idx = new_idx
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(new_idx)
        self.listbox.see(new_idx)
        self._load_ann_frame(self.ann_frame_names[new_idx])

    def _on_listbox_select(self, _event):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.ann_list_idx = sel[0]
        self._load_ann_frame(self.ann_frame_names[sel[0]])

    def _json_path_for_frame(self, name):
        if not self.output_dir:
            return None
        return os.path.join(self.output_dir, "json", f"{name}angulos.json")

    def _load_points_from_json(self, name):
        path = self._json_path_for_frame(name)
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            points = {}
            for label, coords in data.get("puntos", {}).items():
                if isinstance(coords, dict) and "x" in coords and "y" in coords:
                    points[label] = (float(coords["x"]), float(coords["y"]))
            return points
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}

    def _apply_points(self, points):
        self.points = dict(points)
        self.next_point_idx = len(POINT_ORDER)
        for i, label in enumerate(POINT_ORDER):
            if label not in self.points:
                self.next_point_idx = i
                break

    def _refresh_listbox(self):
        if not hasattr(self, "listbox"):
            return
        self.listbox.delete(0, tk.END)
        for name in self.ann_frame_names:
            mark = " ✓" if self._load_points_from_json(name) else ""
            self.listbox.insert(tk.END, f"{name}{mark}")
        if self.ann_frame_names:
            self.listbox.selection_set(self.ann_list_idx)
            self.listbox.see(self.ann_list_idx)

    def _load_ann_frame(self, name):
        self.ann_current_name = name
        path = self.saved_frames[name]
        bgr  = cv2.imread(path)
        if bgr is None:
            messagebox.showerror("Error", f"No se puede leer: {path}")
            return
        self.ann_img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self._apply_points(self._load_points_from_json(name))

        self._redraw_annotation()
        self._update_side_panel()
        self._update_angles()

    def _reset_points(self):
        self.points         = {}
        self.next_point_idx = 0
        self._redraw_annotation()
        self._update_side_panel()
        self._update_angles()

    # ─────────────────────────────────────────
    #  RENDER DE IMÁGENES EN CANVAS
    # ─────────────────────────────────────────
    def _render_image(self, pil_img: Image.Image, overlay=True):
        """Dibuja pil_img en el canvas, centrada y escalada."""
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(),  100)
        ch = max(self.canvas.winfo_height(), 100)

        iw, ih = pil_img.size
        scale  = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        pil_img = pil_img.resize((nw, nh), Image.LANCZOS)
        self._photo      = ImageTk.PhotoImage(pil_img)
        self.canvas_scale  = scale
        self.orig_size     = (iw, ih)
        xo = (cw - nw) // 2
        yo = (ch - nh) // 2
        self.canvas_offset = (xo, yo)

        self.canvas.delete("all")
        self.canvas.create_image(xo, yo, anchor=tk.NW, image=self._photo)

    def _redraw_annotation(self):
        if self.ann_img_rgb is None:
            return
        self._render_image(Image.fromarray(self.ann_img_rgb))
        self._draw_overlay()

    def _draw_overlay(self):
        if not self.points:
            return
        xo, yo = self.canvas_offset
        sc     = self.canvas_scale

        def cx(label):
            ox, oy = self.points[label]
            return xo + ox * sc, yo + oy * sc

        # ── Línea de esqueleto ──
        for i in range(len(POINT_ORDER) - 1):
            a, b = POINT_ORDER[i], POINT_ORDER[i + 1]
            if a in self.points and b in self.points:
                ax, ay = cx(a)
                bx, by = cx(b)
                self.canvas.create_line(ax, ay, bx, by,
                                         fill="#FFFFFF", width=2,
                                         dash=(6, 3), tags="overlay")

        # ── Puntos ──
        for label in POINT_ORDER:
            if label not in self.points:
                continue
            px, py = cx(label)
            color  = POINT_COLORS[label]
            r = 7
            self.canvas.create_oval(px-r, py-r, px+r, py+r,
                                     fill=color, outline="#FFFFFF",
                                     width=2, tags="overlay")
            self.canvas.create_text(px + 11, py - 2,
                                     text=POINT_LABELS[label],
                                     fill=color,
                                     font=("Courier New", 9, "bold"),
                                     anchor=tk.W, tags="overlay")

    # ─────────────────────────────────────────
    #  CLICK EN CANVAS → COLOCAR PUNTO
    # ─────────────────────────────────────────
    def _on_canvas_click(self, event):
        if not self.annotation_mode:
            return
        if self.next_point_idx >= len(POINT_ORDER):
            return

        xo, yo = self.canvas_offset
        sc     = self.canvas_scale
        ox = (event.x - xo) / sc
        oy = (event.y - yo) / sc

        iw, ih = self.orig_size
        if ox < 0 or oy < 0 or ox > iw or oy > ih:
            return

        label = POINT_ORDER[self.next_point_idx]
        self.points[label]  = (ox, oy)
        self.next_point_idx += 1

        self._redraw_annotation()
        self._update_side_panel()
        self._update_angles()

    # ─────────────────────────────────────────
    #  PANEL LATERAL
    # ─────────────────────────────────────────
    def _update_side_panel(self):
        if self.next_point_idx < len(POINT_ORDER):
            nxt = POINT_ORDER[self.next_point_idx]
            self.lbl_next_point.config(
                text=f"▶  Click → {POINT_LABELS[nxt]}",
                fg=POINT_COLORS[nxt],
            )
            self.lbl_point_count.config(
                text=f"Punto {self.next_point_idx + 1} de {len(POINT_ORDER)}"
            )
        else:
            self.lbl_next_point.config(text="✅ Todos marcados", fg="#6BCB77")
            self.lbl_point_count.config(text="")

    def _update_angles(self):
        p = self.points
        for name, p1k, vk, p2k in ANGLE_DEFINITIONS:
            if p1k in p and vk in p and p2k in p:
                angle = calc_angle(p[p1k], p[vk], p[p2k])
                txt   = f"{angle:.1f}°" if angle is not None else "—°"
            else:
                txt = "—°"
            self.angle_vars[name].set(txt)

    # ─────────────────────────────────────────
    #  GUARDAR JSON DE ÁNGULOS
    # ─────────────────────────────────────────
    def _save_json(self):
        if not self.points:
            messagebox.showinfo("Sin puntos", "Marca los 5 puntos primero.")
            return
        if not self.output_dir or not self.ann_current_name:
            return

        ang_dir = os.path.join(self.output_dir, "json")
        os.makedirs(ang_dir, exist_ok=True)

        p = self.points
        angulos = {}
        for name, p1k, vk, p2k in ANGLE_DEFINITIONS:
            if p1k in p and vk in p and p2k in p:
                val = calc_angle(p[p1k], p[vk], p[p2k])
                angulos[name] = round(val, 4) if val is not None else None
            else:
                angulos[name] = None

        data = {
            "frame":   self.ann_current_name,
            "puntos":  {k: {"x": round(v[0], 2), "y": round(v[1], 2)}
                        for k, v in p.items()},
            "angulos": angulos,
        }

        filename  = f"{self.ann_current_name}angulos.json"
        json_path = os.path.join(ang_dir, filename)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self._redraw_annotation()
        self._update_side_panel()
        self._update_angles()
        self._refresh_listbox()

        orig_bg = self.btn_save_json.cget("bg")
        self.btn_save_json.config(bg="#2D6A4F", text="✅  Guardado!")
        self.root.after(1200, lambda: self.btn_save_json.config(
            bg=orig_bg, text="💾  Guardar JSON"
        ))


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = App(root)
    root.mainloop()