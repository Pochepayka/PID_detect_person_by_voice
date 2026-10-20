#!/usr/bin/env python3
"""Simple tkinter GUI for enrollment and identification."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PROFILES, DATA_RAW, DEFAULT_THRESHOLD, SPEAKER_NAMES  # noqa: E402
from src.enroll import enroll_all  # noqa: E402
from src.identify import identify_file, load_profiles  # noqa: E402


class VoiceIdApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Идентификация личности по голосу")
        self.geometry("640x420")
        self.minsize(520, 360)

        self.threshold = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        self.status = tk.StringVar(value="Готово. Выберите WAV или зарегистрируйте базу.")
        self._build()

    def _build(self) -> None:
        pad = {"padx": 10, "pady": 6}
        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True, **pad)

        ttk.Label(frm, text="Система идентификации по голосу (MFCC + слепок)", font=("", 12, "bold")).pack(
            anchor=tk.W
        )

        thr_row = ttk.Frame(frm)
        thr_row.pack(fill=tk.X, pady=8)
        ttk.Label(thr_row, text="Порог cosine:").pack(side=tk.LEFT)
        ttk.Scale(thr_row, from_=0.5, to=0.95, variable=self.threshold, orient=tk.HORIZONTAL).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )
        ttk.Label(thr_row, textvariable=self.threshold, width=5).pack(side=tk.LEFT)

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="Зарегистрировать всех (enroll)", command=self.on_enroll_all).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btns, text="Идентифицировать WAV…", command=self.on_identify).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Обновить список профилей", command=self.refresh_profiles).pack(
            side=tk.LEFT, padx=4
        )

        ttk.Label(frm, text="Зарегистрированные профили:").pack(anchor=tk.W, pady=(12, 2))
        self.profiles_box = tk.Listbox(frm, height=8)
        self.profiles_box.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, textvariable=self.status, wraplength=600).pack(anchor=tk.W, pady=8)
        self.refresh_profiles()

    def refresh_profiles(self) -> None:
        self.profiles_box.delete(0, tk.END)
        profiles = load_profiles(DATA_PROFILES)
        if not profiles:
            self.profiles_box.insert(tk.END, "(пусто — сначала enroll)")
            return
        for sid in sorted(profiles):
            name = SPEAKER_NAMES.get(sid, sid)
            self.profiles_box.insert(tk.END, f"{name}  [{sid}]")
        self.status.set(f"Профилей в базе: {len(profiles)}")

    def on_enroll_all(self) -> None:
        try:
            saved = enroll_all(DATA_RAW, DATA_PROFILES)
            self.refresh_profiles()
            messagebox.showinfo("Enrollment", f"Сохранено профилей: {len(saved)}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка enrollment", str(exc))

    def on_identify(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите WAV",
            filetypes=[("WAV", "*.wav"), ("Все файлы", "*.*")],
            initialdir=str(DATA_RAW),
        )
        if not path:
            return
        try:
            result = identify_file(path, threshold=float(self.threshold.get()))
            lines = [
                f"Файл: {Path(path).name}",
                f"Результат: {result.display_name}",
                f"Лучший score: {result.score:.4f}",
                f"Порог: {float(self.threshold.get()):.2f}",
                "",
                "Все сходства:",
            ]
            for sid, score in sorted(result.all_scores.items(), key=lambda x: -x[1]):
                lines.append(f"  {SPEAKER_NAMES.get(sid, sid)}: {score:.4f}")
            text = "\n".join(lines)
            self.status.set(f"{result.display_name} (score={result.score:.3f})")
            messagebox.showinfo("Идентификация", text)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка identify", str(exc))


def main() -> None:
    app = VoiceIdApp()
    app.mainloop()


if __name__ == "__main__":
    main()
