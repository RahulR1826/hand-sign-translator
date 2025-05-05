# video_source.py

import tkinter as tk

def capture_selected_area():
    coords = {}

    def on_mouse_down(event):
        coords['start_x'] = event.x_root
        coords['start_y'] = event.y_root
        coords['rect'] = canvas.create_rectangle(
            coords['start_x'], coords['start_y'],
            coords['start_x'], coords['start_y'],
            outline='red', width=2
        )

    def on_mouse_move(event):
        if 'rect' in coords:
            canvas.coords(
                coords['rect'],
                coords['start_x'], coords['start_y'],
                event.x_root, event.y_root
            )

    def on_mouse_up(event):
        coords['end_x'] = event.x_root
        coords['end_y'] = event.y_root
        root.destroy()

    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.attributes('-alpha', 0.3)
    root.configure(bg='gray')

    canvas = tk.Canvas(root, bg='gray', highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)

    root.mainloop()

    x1 = min(coords['start_x'], coords['end_x'])
    y1 = min(coords['start_y'], coords['end_y'])
    x2 = max(coords['start_x'], coords['end_x'])
    y2 = max(coords['start_y'], coords['end_y'])

    width = x2 - x1
    height = y2 - y1

    return x1, y1, width, height

