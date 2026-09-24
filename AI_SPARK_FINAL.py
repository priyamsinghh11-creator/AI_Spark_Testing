import os
import threading
import tkinter as tk
import customtkinter as ctk
import cv2

from tkinter import filedialog, messagebox
from PIL import Image, ImageOps
from ultralytics import YOLO


# ============================================================
# AI SPARK TESTING - V1.4 PURPLE DASHBOARD
# Uses your own images from the assets folder.
#
# Required assets:
#   assets/dashboard.bg.jpg
#   assets/image_card.jpg
#   assets/video_card.jpg
#   assets/camera_card.jpg
#
# Model:
#   runs/classify/train/weights/last.pt
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

MODEL_PATH = os.path.join(
    BASE_DIR, "runs", "classify", "runs", "train_53metal", "weights", "best.pt"
)

BG_PATH = os.path.join(ASSETS_DIR, "dashboard_bg.jpg")
IMAGE_CARD_PATH = os.path.join(ASSETS_DIR, "image_card.jpg")
VIDEO_CARD_PATH = os.path.join(ASSETS_DIR, "video_card.jpg")
CAMERA_CARD_PATH = os.path.join(ASSETS_DIR, "camera_card.jpg")
FLAME_LOGO_PATH = os.path.join(ASSETS_DIR, "flame_logo.png")


# ---------------- COLORS ----------------

BG = "#090611"
SIDEBAR = "#0D0918"
PANEL = "#130D20"
CARD = "#160F26"
CARD_HOVER = "#211337"

PURPLE = "#A855F7"
PURPLE_LIGHT = "#C084FC"
PURPLE_DARK = "#6D28D9"

WHITE = "#FAF5FF"
TEXT = "#EDE9FE"
MUTED = "#B6A7C7"

GREEN = "#4ADE80"
RED = "#F87171"
YELLOW = "#FACC15"


# ---------------- MODEL ----------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found:\n\n" + MODEL_PATH +
        "\n\nCheck that best.pt exists in runs/classify/runs/train_53metal/weights/"
    )

model = YOLO(MODEL_PATH)


class SparkTestingApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("AI Spark Testing • V1.4")
        self.geometry("1450x850")
        self.minsize(1180, 720)
        self.configure(fg_color=BG)

        self.video_cap = None
        self.camera_cap = None

        self.video_running = False
        self.camera_running = False

        self.bg_photo = None
        self.bg_label = None
        self.bg_source = None
        self.card_photos = []

        self.build_ui()

    # ========================================================
    # MAIN UI
    # ========================================================

    def build_ui(self):

        # ---------- SIDEBAR ----------

        self.sidebar = ctk.CTkFrame(
            self,
            width=245,
            corner_radius=0,
            fg_color=SIDEBAR
        )
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        if os.path.exists(FLAME_LOGO_PATH):
            try:
                flame_img = Image.open(FLAME_LOGO_PATH).convert("RGBA")
                self.logo_photo = ctk.CTkImage(
                    light_image=flame_img,
                    dark_image=flame_img,
                    size=(58, 58)
                )
                logo = ctk.CTkLabel(
                    self.sidebar,
                    text="",
                    image=self.logo_photo
                )
            except Exception:
                logo = ctk.CTkLabel(self.sidebar, text="🔥", font=("Segoe UI Emoji", 40))
        else:
            logo = ctk.CTkLabel(self.sidebar, text="🔥", font=("Segoe UI Emoji", 40))
        logo.pack(pady=(30, 0))

        ctk.CTkLabel(
            self.sidebar,
            text="AI SPARK",
            font=("Segoe UI", 22, "bold"),
            text_color=WHITE
        ).pack(pady=(0, 0))

        ctk.CTkLabel(
            self.sidebar,
            text="TESTING",
            font=("Segoe UI", 22, "bold"),
            text_color=PURPLE_LIGHT
        ).pack()

        ctk.CTkLabel(
            self.sidebar,
            text="INTELLIGENT METAL\nIDENTIFICATION",
            font=("Segoe UI", 10, "bold"),
            text_color=MUTED,
            justify="center"
        ).pack(pady=(5, 35))

        self.nav_button("⌂   Dashboard", self.show_dashboard, True)
        self.nav_button("▧   Image Analysis", self.show_image_analysis)
        self.nav_button("▷   Video Analysis", self.show_video_analysis)
        self.nav_button("◉   Live Camera", self.show_camera_analysis)

        # Model status
        status = ctk.CTkFrame(
            self.sidebar,
            fg_color="#111B1A",
            corner_radius=14,
            border_width=1,
            border_color="#174B32"
        )
        status.pack(side="bottom", fill="x", padx=18, pady=25)

        ctk.CTkLabel(
            status,
            text="●  MODEL ONLINE",
            font=("Segoe UI", 12, "bold"),
            text_color=GREEN
        ).pack(anchor="w", padx=15, pady=(13, 2))

        ctk.CTkLabel(
            status,
            text="YOLO Classification",
            font=("Segoe UI", 10),
            text_color=MUTED
        ).pack(anchor="w", padx=15, pady=(0, 13))

        # ---------- CONTENT ----------

        self.content = ctk.CTkFrame(
            self,
            corner_radius=0,
            fg_color="transparent"
        )
        self.content.pack(side="right", fill="both", expand=True)

        self.show_dashboard()

    def nav_button(self, text, command, active=False):

        button = ctk.CTkButton(
            self.sidebar,
            text=text,
            command=command,
            height=48,
            corner_radius=12,
            anchor="w",
            font=("Segoe UI", 13, "bold"),
            fg_color=PURPLE_DARK if active else "transparent",
            hover_color="#28143F",
            text_color=WHITE
        )
        button.pack(fill="x", padx=15, pady=5)

    # ========================================================
    # DASHBOARD
    # ========================================================

    def show_dashboard(self):

        self.stop_video()
        self.stop_camera()
        self.clear_content()

        # Full dashboard background image
        # IMPORTANT: the content frame is transparent so this image is actually visible.
        if os.path.exists(BG_PATH):
            try:
                self.bg_source = Image.open(BG_PATH).convert("RGB")
                self.bg_label = ctk.CTkLabel(
                    self.content,
                    text="",
                    fg_color="transparent"
                )
                self.bg_label.place(relx=0, rely=0, relwidth=1, relheight=1)
                self.bg_label.lower()
                self.update_dashboard_background()
            except Exception:
                self.bg_source = None
        else:
            self.bg_source = None

        self.content.bind("<Configure>", self._on_content_resize)

        # Dashboard heading
        ctk.CTkLabel(
            self.content,
            text="DASHBOARD",
            font=("Segoe UI", 12, "bold"),
            text_color=PURPLE_LIGHT
        ).place(x=40, y=12)

        # Small hero box on LEFT so background stays visible
        hero = ctk.CTkFrame(
            self.content,
            fg_color="#0C0815",
            corner_radius=18,
            border_width=1,
            border_color=PURPLE,
            width=520,
            height=145
        )
        hero.place(x=38, y=35)
        hero.pack_propagate(False)

        ctk.CTkLabel(
            hero,
            text="AI SPARK TESTING",
            font=("Segoe UI", 29, "bold"),
            text_color=WHITE
        ).pack(anchor="w", padx=25, pady=(20, 0))

        ctk.CTkLabel(
            hero,
            text="Identify metals from spark patterns.",
            font=("Segoe UI", 15, "bold"),
            text_color=PURPLE_LIGHT
        ).pack(anchor="w", padx=25, pady=(3, 0))

        ctk.CTkLabel(
            hero,
            text="AI-powered image, video & live analysis",
            font=("Segoe UI", 11),
            text_color=MUTED
        ).pack(anchor="w", padx=25, pady=(4, 0))

        # Three cards
        self.create_card(
            x=38,
            y=190,
            image_path=IMAGE_CARD_PATH,
            small_title="IMAGE ANALYSIS",
            title="Analyze Spark Images",
            description="Upload a spark image and identify the predicted metal.",
            button_text="OPEN IMAGE ANALYSIS   →",
            command=self.show_image_analysis
        )

        self.create_card(
            x=405,
            y=190,
            image_path=VIDEO_CARD_PATH,
            small_title="VIDEO ANALYSIS",
            title="Analyze Spark Videos",
            description="Play a spark video while AI analyzes the spark pattern.",
            button_text="OPEN VIDEO ANALYSIS   →",
            command=self.show_video_analysis
        )

        self.create_card(
            x=772,
            y=190,
            image_path=CAMERA_CARD_PATH,
            small_title="LIVE CAMERA",
            title="Live Camera Analysis",
            description="Run real-time predictions using your camera.",
            button_text="OPEN LIVE CAMERA   →",
            command=self.show_camera_analysis
        )

        ctk.CTkLabel(
            self.content,
            text="V1.4  •  AI SPARK TESTING",
            font=("Segoe UI", 10, "bold"),
            text_color=MUTED
        ).place(relx=0.97, rely=0.96, anchor="se")

    def _on_content_resize(self, event=None):
        if self.bg_source is not None and self.bg_label is not None:
            self.update_dashboard_background()

    def update_dashboard_background(self):
        if self.bg_source is None or self.bg_label is None:
            return
        w = max(1, self.content.winfo_width())
        h = max(1, self.content.winfo_height())
        # Crop to the available dashboard area WITHOUT stretching the image.
        fitted = ImageOps.fit(self.bg_source, (w, h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        self.bg_photo = ctk.CTkImage(light_image=fitted, dark_image=fitted, size=(w, h))
        self.bg_label.configure(image=self.bg_photo)
        self.bg_label.image = self.bg_photo
        self.bg_label.lower()

    def create_card(
        self,
        x,
        y,
        image_path,
        small_title,
        title,
        description,
        button_text,
        command
    ):

        card = ctk.CTkFrame(
            self.content,
            width=345,
            height=555,
            corner_radius=18,
            fg_color="#100A1A",
            border_width=1,
            border_color=PURPLE
        )
        card.place(x=x, y=y)
        card.pack_propagate(False)

        # User's own card image.
        # Fills the whole upper card area WITHOUT stretching/compressing.
        # ImageOps.fit preserves proportions and crops only the excess edges.
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")

                IMAGE_W = 315
                IMAGE_H = 285

                fitted = ImageOps.fit(
                    img,
                    (IMAGE_W, IMAGE_H),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5)
                )

                photo = ctk.CTkImage(
                    light_image=fitted,
                    dark_image=fitted,
                    size=(IMAGE_W, IMAGE_H)
                )
                self.card_photos.append(photo)

                image_label = ctk.CTkLabel(
                    card,
                    text="",
                    image=photo,
                    fg_color="#0B0710",
                    corner_radius=12
                )
                image_label.pack(
                    padx=14,
                    pady=(14, 10)
                )

            except Exception:
                self.image_placeholder(card)
        else:
            self.image_placeholder(card)

        ctk.CTkLabel(
            card,
            text=small_title,
            font=("Segoe UI", 10, "bold"),
            text_color=PURPLE_LIGHT
        ).pack(anchor="w", padx=20, pady=(3, 0))

        ctk.CTkLabel(
            card,
            text=title,
            font=("Segoe UI", 20, "bold"),
            text_color=WHITE
        ).pack(anchor="w", padx=20, pady=(3, 0))

        ctk.CTkLabel(
            card,
            text=description,
            font=("Segoe UI", 11),
            text_color=MUTED,
            justify="left",
            wraplength=295
        ).pack(anchor="w", padx=20, pady=(5, 0))

        ctk.CTkButton(
            card,
            text=button_text,
            command=command,
            height=42,
            corner_radius=12,
            fg_color=PURPLE_DARK,
            hover_color=PURPLE,
            font=("Segoe UI", 11, "bold")
        ).pack(
            fill="x",
            padx=20,
            pady=(18, 18)
        )

    def image_placeholder(self, parent):

        ctk.CTkLabel(
            parent,
            text="IMAGE\nNOT FOUND",
            font=("Segoe UI", 18, "bold"),
            text_color=MUTED,
            fg_color="#21152D",
            width=315,
            height=230,
            corner_radius=10
        ).pack(padx=14, pady=(14, 10))

    # ========================================================
    # IMAGE ANALYSIS
    # ========================================================

    def show_image_analysis(self):

        self.stop_video()
        self.stop_camera()
        self.clear_content()

        ctk.CTkLabel(self.content, text="IMAGE ANALYSIS", font=("Segoe UI", 30, "bold"), text_color=WHITE).pack(pady=(30, 4))
        ctk.CTkLabel(self.content, text="Upload a spark image to identify the metal.", font=("Segoe UI", 13), text_color=MUTED).pack()

        row = ctk.CTkFrame(self.content, fg_color="transparent")
        row.pack(fill="both", expand=True, padx=25, pady=20)

        preview = ctk.CTkFrame(row, fg_color=PANEL, corner_radius=18, width=700, height=500)
        preview.pack(side="left", fill="both", expand=True, padx=(0,15))
        preview.pack_propagate(False)

        image_label = ctk.CTkLabel(preview, text="Upload an image", font=("Segoe UI", 18), text_color=MUTED)
        image_label.pack(fill="both", expand=True, padx=20, pady=20)

        panel = ctk.CTkFrame(row, fg_color=PANEL, corner_radius=18, width=380, height=500)
        panel.pack(side="left", fill="y")
        panel.pack_propagate(False)

        ctk.CTkLabel(panel, text="PREDICTION", font=("Segoe UI", 14, "bold"), text_color=WHITE).pack(anchor="w", padx=25, pady=(25,5))
        prediction_label = ctk.CTkLabel(panel, text="—", font=("Segoe UI", 15), text_color=WHITE)
        prediction_label.pack(anchor="w", padx=25)
        confidence_label = ctk.CTkLabel(panel, text="Confidence: —", font=("Segoe UI", 15), text_color=WHITE)
        confidence_label.pack(anchor="w", padx=25, pady=(3,25))
        ctk.CTkLabel(panel, text="TOP PREDICTIONS", font=("Segoe UI", 14, "bold"), text_color=WHITE).pack(anchor="w", padx=25, pady=(0,12))
        rows=[]
        for i in range(3):
            r=ctk.CTkFrame(panel, fg_color="transparent"); r.pack(fill="x", padx=25, pady=3)
            n=ctk.CTkLabel(r, text=f"{i+1}. —", font=("Segoe UI",15), text_color=WHITE); n.pack(side="left")
            q=ctk.CTkLabel(r, text="—", font=("Segoe UI",15), text_color=WHITE); q.pack(side="right")
            rows.append((n,q))

        def choose_image():
            path = filedialog.askopenfilename(title="Select Spark Image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")])
            if not path: return
            try:
                pil=Image.open(path).convert("RGB")
                display=ctk.CTkImage(light_image=pil,dark_image=pil,size=(680,430))
                image_label.configure(image=display,text=""); image_label.image=display
                results=model(path,verbose=False); probs=results[0].probs
                if probs is None: return
                top=probs.top5[:3]; confs=probs.top5conf.tolist()[:3]; names=results[0].names
                prediction_label.configure(text=names[int(top[0])])
                confidence_label.configure(text=f"Confidence: {confs[0]*100:.1f}%")
                for i,(idx,conf) in enumerate(zip(top,confs)):
                    rows[i][0].configure(text=f"{i+1}. {names[int(idx)]}")
                    rows[i][1].configure(text="")
            except Exception as e: messagebox.showerror("Image Analysis Error",str(e))

        ctk.CTkButton(self.content,text="▧  SELECT IMAGE",command=choose_image,width=250,height=48,corner_radius=12,fg_color=PURPLE_DARK,hover_color=PURPLE,font=("Segoe UI",13,"bold")).pack(pady=(0,12))

    # ========================================================
    # VIDEO ANALYSIS
    # ========================================================

    def show_video_analysis(self):
        self.stop_camera(); self.stop_video(); self.clear_content()
        ctk.CTkLabel(self.content,text="VIDEO ANALYSIS",font=("Segoe UI",30,"bold"),text_color=WHITE).pack(pady=(30,4))
        ctk.CTkLabel(self.content,text="Play the video while AI analyzes each frame.",font=("Segoe UI",13),text_color=MUTED).pack()

        row=ctk.CTkFrame(self.content,fg_color="transparent"); row.pack(fill="both",expand=True,padx=25,pady=18)
        left=ctk.CTkFrame(row,fg_color=PANEL,corner_radius=18); left.pack(side="left",fill="both",expand=True,padx=(0,15))
        self.video_display=ctk.CTkLabel(left,text="Select a spark video",fg_color="#050307",corner_radius=15,text_color=MUTED,font=("Segoe UI",18)); self.video_display.pack(fill="both",expand=True,padx=18,pady=18)
        right=ctk.CTkFrame(row,fg_color=PANEL,corner_radius=18,width=380); right.pack(side="left",fill="y"); right.pack_propagate(False)
        ctk.CTkLabel(right,text="PREDICTION",font=("Segoe UI",14,"bold"),text_color=WHITE).pack(anchor="w",padx=25,pady=(25,5))
        self.video_prediction=ctk.CTkLabel(right,text="—",font=("Segoe UI",15),text_color=WHITE); self.video_prediction.pack(anchor="w",padx=25)
        self.video_confidence=ctk.CTkLabel(right,text="Confidence: —",font=("Segoe UI",15),text_color=WHITE); self.video_confidence.pack(anchor="w",padx=25,pady=(3,25))
        ctk.CTkLabel(right,text="TOP PREDICTIONS",font=("Segoe UI",14,"bold"),text_color=WHITE).pack(anchor="w",padx=25,pady=(0,12))
        self.video_rows=[]
        for i in range(3):
            rr=ctk.CTkFrame(right,fg_color="transparent"); rr.pack(fill="x",padx=25,pady=3)
            n=ctk.CTkLabel(rr,text=f"{i+1}. —",font=("Segoe UI",15),text_color=WHITE); n.pack(side="left")
            q=ctk.CTkLabel(rr,text="—",font=("Segoe UI",15),text_color=WHITE); q.pack(side="right")
            self.video_rows.append((n,q))

        self.video_progress=ctk.CTkProgressBar(self.content,width=520,height=10,progress_color=PURPLE_DARK); self.video_progress.set(0); self.video_progress.pack(pady=(0,3))
        self.video_time=ctk.CTkLabel(self.content,text="00:00 / 00:00",font=("Segoe UI",11),text_color=MUTED); self.video_time.pack(pady=(0,5))
        controls=ctk.CTkFrame(self.content,fg_color="transparent"); controls.pack(pady=(0,12))
        ctk.CTkButton(controls,text="▶  PLAY",command=self.play_video,width=120,height=42,fg_color=PURPLE_DARK,hover_color=PURPLE).pack(side="left",padx=5)
        ctk.CTkButton(controls,text="Ⅱ  PAUSE",command=self.pause_video,width=120,height=42,fg_color="#2563A6",hover_color="#3182CE").pack(side="left",padx=5)
        ctk.CTkButton(controls,text="■  STOP",command=self.stop_video,width=120,height=42,fg_color="#7F1D2D",hover_color="#991B1B").pack(side="left",padx=5)
        ctk.CTkButton(controls,text="UPLOAD VIDEO",command=self.choose_video,width=150,height=42,fg_color=PURPLE_DARK,hover_color=PURPLE).pack(side="left",padx=18)

    def choose_video(self):
        self.stop_video()
        path=filedialog.askopenfilename(title="Select Spark Video",filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv *.webm")])
        if not path: return
        cap=cv2.VideoCapture(path)
        if not cap.isOpened(): messagebox.showerror("Video Error","Could not open this video."); return
        self.video_cap=cap; self.video_total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); self.video_fps=float(cap.get(cv2.CAP_PROP_FPS) or 25); self.video_running=False
        self.video_progress.set(0); self.video_time.configure(text=f"00:00 / {self._fmt_time(self.video_total/self.video_fps)}")
        ok,frame=cap.read();
        if ok: self._show_video_frame(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    def play_video(self):
        if self.video_cap is not None and not self.video_running: self.video_running=True; self.update_video()

    def pause_video(self): self.video_running=False

    def update_video(self):
        if not self.video_running or self.video_cap is None: return
        ok,frame=self.video_cap.read()
        if not ok: self.stop_video(); return
        try:
            results=model(frame,verbose=False); probs=results[0].probs
            if probs is not None:
                top=probs.top5[:3]; confs=probs.top5conf.tolist()[:3]; names=results[0].names
                self.video_prediction.configure(text=names[int(top[0])]); self.video_confidence.configure(text=f"Confidence: {confs[0]*100:.1f}%")
                for i,(idx,conf) in enumerate(zip(top,confs)):
                    self.video_rows[i][0].configure(text=f"{i+1}. {names[int(idx)]}"); self.video_rows[i][1].configure(text="")
        except Exception: pass
        self._show_video_frame(frame)
        pos=self.video_cap.get(cv2.CAP_PROP_POS_FRAMES); total=max(1,self.video_total); self.video_progress.set(min(1,pos/total)); self.video_time.configure(text=f"{self._fmt_time(pos/self.video_fps)} / {self._fmt_time(total/self.video_fps)}")
        self.after(35,self.update_video)

    def _show_video_frame(self,frame):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); pil=Image.fromarray(rgb); display=ctk.CTkImage(light_image=pil,dark_image=pil,size=(760,500)); self.video_display.configure(image=display,text=""); self.video_display.image=display

    def _fmt_time(self,seconds):
        seconds=max(0,int(seconds)); return f"{seconds//60:02d}:{seconds%60:02d}"

    def stop_video(self):
        self.video_running=False
        if getattr(self,"video_cap",None) is not None:
            self.video_cap.release(); self.video_cap=None
        if hasattr(self,"video_progress") and self.video_progress.winfo_exists(): self.video_progress.set(0)

    # ========================================================
    # LIVE CAMERA
    # ========================================================

    def show_camera_analysis(self):
        self.stop_video(); self.stop_camera(); self.clear_content()
        ctk.CTkLabel(self.content,text="LIVE CAMERA",font=("Segoe UI",30,"bold"),text_color=WHITE).pack(pady=(30,4))
        ctk.CTkLabel(self.content,text="Real-time spark pattern analysis.",font=("Segoe UI",13),text_color=MUTED).pack()
        row=ctk.CTkFrame(self.content,fg_color="transparent"); row.pack(fill="both",expand=True,padx=25,pady=18)
        left=ctk.CTkFrame(row,fg_color=PANEL,corner_radius=18); left.pack(side="left",fill="both",expand=True,padx=(0,15))
        self.camera_display=ctk.CTkLabel(left,text="Camera stopped",fg_color="#050307",corner_radius=15,text_color=MUTED,font=("Segoe UI",18)); self.camera_display.pack(fill="both",expand=True,padx=18,pady=18)
        right=ctk.CTkFrame(row,fg_color=PANEL,corner_radius=18,width=380); right.pack(side="left",fill="y"); right.pack_propagate(False)
        ctk.CTkLabel(right,text="PREDICTION",font=("Segoe UI",14,"bold"),text_color=WHITE).pack(anchor="w",padx=25,pady=(25,5))
        self.camera_prediction=ctk.CTkLabel(right,text="—",font=("Segoe UI",15),text_color=WHITE); self.camera_prediction.pack(anchor="w",padx=25)
        self.camera_confidence=ctk.CTkLabel(right,text="Confidence: —",font=("Segoe UI",15),text_color=WHITE); self.camera_confidence.pack(anchor="w",padx=25,pady=(3,25))
        ctk.CTkLabel(right,text="TOP PREDICTIONS",font=("Segoe UI",14,"bold"),text_color=WHITE).pack(anchor="w",padx=25,pady=(0,12))
        self.camera_rows=[]
        for i in range(3):
            rr=ctk.CTkFrame(right,fg_color="transparent"); rr.pack(fill="x",padx=25,pady=3)
            n=ctk.CTkLabel(rr,text=f"{i+1}. —",font=("Segoe UI",15),text_color=WHITE); n.pack(side="left")
            q=ctk.CTkLabel(rr,text="—",font=("Segoe UI",15),text_color=WHITE); q.pack(side="right")
            self.camera_rows.append((n,q))
        controls=ctk.CTkFrame(self.content,fg_color="transparent"); controls.pack(pady=(0,12))
        ctk.CTkButton(controls,text="START CAMERA",command=self.start_camera,width=190,height=45,fg_color=PURPLE_DARK,hover_color=PURPLE).pack(side="left",padx=7)
        ctk.CTkButton(controls,text="STOP CAMERA",command=self.stop_camera,width=190,height=45,fg_color="#7F1D2D",hover_color="#991B1B").pack(side="left",padx=7)

    def start_camera(self):
        self.stop_camera(); self.camera_cap=cv2.VideoCapture(0)
        if not self.camera_cap.isOpened(): messagebox.showerror("Camera Error","Could not open camera."); self.camera_cap=None; return
        self.camera_running=True; self.update_camera()

    def update_camera(self):
        if not self.camera_running or self.camera_cap is None: return
        ok,frame=self.camera_cap.read()
        if not ok: self.stop_camera(); return
        try:
            results=model(frame,verbose=False); probs=results[0].probs
            if probs is not None:
                top=probs.top5[:3]; confs=probs.top5conf.tolist()[:3]; names=results[0].names
                self.camera_prediction.configure(text=names[int(top[0])]); self.camera_confidence.configure(text=f"Confidence: {confs[0]*100:.1f}%")
                for i,(idx,conf) in enumerate(zip(top,confs)):
                    self.camera_rows[i][0].configure(text=f"{i+1}. {names[int(idx)]}"); self.camera_rows[i][1].configure(text="")
        except Exception: pass
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); pil=Image.fromarray(rgb); display=ctk.CTkImage(light_image=pil,dark_image=pil,size=(760,500)); self.camera_display.configure(image=display,text=""); self.camera_display.image=display
        self.after(35,self.update_camera)

    def stop_camera(self):
        self.camera_running=False
        if getattr(self,"camera_cap",None) is not None: self.camera_cap.release(); self.camera_cap=None

    # ========================================================
    # HELPERS
    # ========================================================

    def clear_content(self):

        for widget in self.content.winfo_children():
            widget.destroy()

        self.bg_photo = None
        self.card_photos.clear()

    def on_close(self):

        self.stop_video()
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = SparkTestingApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
