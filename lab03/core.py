import tkinter as tk
from tkinter import ttk
import random

CELL_SIZE = 10
GRID_W = 60
GRID_H = 60
FPS = 30

COLOR_BG = "#2E3440"
COLOR_PANEL = "#3B4252"
COLOR_GRID = "#434C5E"
COLOR_TREE = "#A3BE8C"
COLOR_FIRE = "#BF616A"
COLOR_ASH = "#4C566A"
COLOR_TEXT = "#ECEFF4"
COLOR_ACCENT = "#88C0D0"

STATE_EMPTY = 0
STATE_TREE = 1
STATE_BURNING = 2
STATE_ASH = 3

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command, width=120, height=40, radius=20, bg=COLOR_ACCENT, fg=COLOR_BG):
        super().__init__(parent, width=width, height=height, bg=COLOR_PANEL, highlightthickness=0)
        self.command = command
        self.radius = radius
        self.bg_color = bg
        self.fg_color = fg
        self.text_str = text
        
        self.rect = self.create_rounded_rect(2, 2, width-2, height-2, radius, fill=bg, outline=bg)
        self.text = self.create_text(width/2, height/2, text=text, fill=fg, font=("Helvetica", 10, "bold"))
        
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_hover)
        self.bind("<Leave>", self._on_leave)

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r,
            x2, y1+r, x2, y2-r, x2, y2-r, x2, y2, x2-r, y2, x2-r, y2,
            x1+r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y2-r, x1, y1+r,
            x1, y1+r, x1, y1
        ]
        return self.create_polygon(points, **kwargs, smooth=True)

    def _on_click(self, event):
        self.command()
        self.move(self.text, 1, 1)
        self.after(100, lambda: self.move(self.text, -1, -1))

    def _on_hover(self, event):
        self.itemconfig(self.rect, fill="#81A1C1")

    def _on_leave(self, event):
        self.itemconfig(self.rect, fill=self.bg_color)

class ForestFireSim:
    def __init__(self, root):
        self.root = root
        self.root.title("Симуляция лесного пожара")
        self.root.configure(bg=COLOR_BG)
        self.root.geometry(f"{GRID_W * CELL_SIZE + 280}x{GRID_H * CELL_SIZE + 40}")
        
        self.running = False
        
        self.grid = [[STATE_EMPTY for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.next_grid = [[STATE_EMPTY for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.rects = []

        self._setup_ui()
        self._init_graphics()
        
    def _setup_ui(self):
        main_frame = tk.Frame(self.root, bg=COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.canvas = tk.Canvas(
            main_frame, 
            width=GRID_W * CELL_SIZE, 
            height=GRID_H * CELL_SIZE, 
            bg=COLOR_GRID, 
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, padx=(0, 20))

        controls = tk.Frame(main_frame, bg=COLOR_PANEL, width=240, padx=15, pady=15)
        controls.pack(side=tk.RIGHT, fill=tk.Y)
        controls.pack_propagate(False)

        lbl_title = tk.Label(controls, text="Настройки", bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Helvetica", 14, "bold"))
        lbl_title.pack(pady=(0, 20))

        self.scale_growth = self._create_slider(controls, "Вероятность роста", 0.001, 0.05, 0.01)
        self.scale_fire = self._create_slider(controls, "Молния (самовозгорание)", 0.00001, 0.001, 0.0001, decimals=5)
        self.scale_humidity = self._create_slider(controls, "Влажность (защита)", 0.0, 1.0, 0.3)
        self.scale_wind = self._create_slider(controls, "Ветер (Смещение X)", -1.0, 1.0, 0.5)

        btn_frame = tk.Frame(controls, bg=COLOR_PANEL)
        btn_frame.pack(pady=20)
        
        self.btn_start = RoundedButton(btn_frame, "Старт / Пауза", self.toggle_sim)
        self.btn_start.pack(pady=5)
        
        self.btn_reset = RoundedButton(btn_frame, "Сброс поля", self.reset_sim, bg="#BF616A")
        self.btn_reset.pack(pady=5)

        self._create_legend(controls)

    def _create_slider(self, parent, label, vmin, vmax, vdef, decimals=3):
        lbl = tk.Label(parent, text=label, bg=COLOR_PANEL, fg="#D8DEE9", font=("Helvetica", 9))
        lbl.pack(anchor="w", pady=(10, 0))
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Horizontal.TScale", background=COLOR_PANEL, troughcolor=COLOR_BG, bordercolor=COLOR_PANEL, darkcolor=COLOR_PANEL, lightcolor=COLOR_PANEL)
        
        var = tk.DoubleVar(value=vdef)
        scale = ttk.Scale(parent, from_=vmin, to=vmax, variable=var, orient=tk.HORIZONTAL, style="Horizontal.TScale")
        scale.pack(fill=tk.X, pady=(0, 5))
        return var

    def _create_legend(self, parent):
        leg_frame = tk.Frame(parent, bg=COLOR_PANEL)
        leg_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        items = [
            (COLOR_TREE, "Дерево"),
            (COLOR_FIRE, "Огонь"),
            (COLOR_ASH, "Пепел"),
            (COLOR_GRID, "Пусто")
        ]
        
        for col, txt in items:
            row = tk.Frame(leg_frame, bg=COLOR_PANEL)
            row.pack(fill=tk.X, pady=2)
            box = tk.Frame(row, bg=col, width=12, height=12)
            box.pack(side=tk.LEFT, padx=5)
            lbl = tk.Label(row, text=txt, bg=COLOR_PANEL, fg="#D8DEE9", font=("Helvetica", 8))
            lbl.pack(side=tk.LEFT)

    def _init_graphics(self):
        self.rects = []
        for y in range(GRID_H):
            row_rects = []
            for x in range(GRID_W):
                x0, y0 = x * CELL_SIZE, y * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE

                r_id = self.canvas.create_rectangle(x0, y0, x1, y1, fill=COLOR_GRID, outline="")
                row_rects.append(r_id)
            self.rects.append(row_rects)

    def get_neighbors(self, x, y):
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0: 
                    continue

                nx, ny = x + dx, y + dy
                nx = nx % GRID_W
                ny = ny % GRID_H
                neighbors.append(((dx, dy), self.grid[ny][nx]))
        return neighbors

    def apply_rules(self):
        p_grow = self.scale_growth.get()
        p_fire = self.scale_fire.get()
        humidity = self.scale_humidity.get()
        wind_bias = self.scale_wind.get()

        p_ash_decay = 0.05 

        for y in range(GRID_H):
            for x in range(GRID_W):
                state = self.grid[y][x]
                
                if state == STATE_BURNING:
                    self.next_grid[y][x] = STATE_ASH
                
                elif state == STATE_ASH:
                    if random.random() < p_ash_decay:
                        self.next_grid[y][x] = STATE_EMPTY
                    else:
                        self.next_grid[y][x] = STATE_ASH

                elif state == STATE_EMPTY:
                    if random.random() < p_grow:
                        self.next_grid[y][x] = STATE_TREE
                    else:
                        self.next_grid[y][x] = STATE_EMPTY

                elif state == STATE_TREE:
                    neighbors = self.get_neighbors(x, y)
                    catch_fire = False
                    
                    if random.random() < p_fire:
                        catch_fire = True
                    else:
                        for (dx, dy), n_state in neighbors:
                            if n_state == STATE_BURNING:
                                ign_prob = 0.60 
                                
                                if wind_bias > 0: 
                                    if dx < 0: 
                                        ign_prob += abs(wind_bias) * 0.4
                                    if dx > 0: 
                                        ign_prob -= abs(wind_bias) * 0.4

                                elif wind_bias < 0:
                                    if dx > 0: 
                                        ign_prob += abs(wind_bias) * 0.4
                                    if dx < 0: 
                                        ign_prob -= abs(wind_bias) * 0.4
                                
                                ign_prob -= (humidity * 0.5)

                                if random.random() < ign_prob:
                                    catch_fire = True
                                    break
                    
                    if catch_fire:
                        self.next_grid[y][x] = STATE_BURNING
                    else:
                        self.next_grid[y][x] = STATE_TREE
        
        for y in range(GRID_H):
            for x in range(GRID_W):
                self.grid[y][x] = self.next_grid[y][x]

    def update_graphics(self):
        colors = {
            STATE_EMPTY: COLOR_GRID,
            STATE_TREE: COLOR_TREE,
            STATE_BURNING: COLOR_FIRE,
            STATE_ASH: COLOR_ASH
        }
        
        for y in range(GRID_H):
            for x in range(GRID_W):
                state = self.grid[y][x]
                self.canvas.itemconfig(self.rects[y][x], fill=colors[state])

    def loop(self):
        if self.running:
            self.apply_rules()
            self.update_graphics()
            self.root.after(int(1000/FPS), self.loop)

    def toggle_sim(self):
        self.running = not self.running
        if self.running:
            self.loop()

    def reset_sim(self):
        self.running = False
        self.grid = [[STATE_EMPTY for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.update_graphics()

if __name__ == "__main__":
    root = tk.Tk()
    app = ForestFireSim(root)
    root.mainloop()
