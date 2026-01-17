import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

# ==========================================
# GOBEST CAB - SAFETY PREDICTION APP (MOCKUP)
# Phase 1: UI Shell Only (No AI Model yet)
# ==========================================

class SafetyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gobest Cab - Driver Safety System (CA2 Sprint 2)")
        self.root.geometry("600x500")
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create Tabs for the two required modes
        self.tab_control = ttk.Notebook(root)
        
        self.tab_batch = ttk.Frame(self.tab_control)
        self.tab_realtime = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab_batch, text='Batch Processing (CSV)')
        self.tab_control.add(self.tab_realtime, text='Real-Time Prediction')
        self.tab_control.pack(expand=1, fill="both")
        
        # Initialize the layouts
        self.setup_batch_tab()
        self.setup_realtime_tab()

    # ==========================================
    # TAB 1: BATCH PROCESSING (Requirement 4a)
    # ==========================================
    def setup_batch_tab(self):
        frame = ttk.LabelFrame(self.tab_batch, text="Bulk Analysis", padding=20)
        frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Step 1: Upload
        lbl_instr = ttk.Label(frame, text="Step 1: Upload Sensor Data (CSV)")
        lbl_instr.pack(pady=5)
        
        self.btn_upload = ttk.Button(frame, text="Browse CSV File", command=self.load_csv)
        self.btn_upload.pack(pady=5)
        
        self.lbl_file_status = ttk.Label(frame, text="No file loaded", foreground="red")
        self.lbl_file_status.pack(pady=5)
        
        # Step 2: Predict
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=15)
        
        lbl_step2 = ttk.Label(frame, text="Step 2: Run AI Model")
        lbl_step2.pack(pady=5)
        
        self.btn_run_batch = ttk.Button(frame, text="Process All Trips", state="disabled", command=self.run_batch_prediction)
        self.btn_run_batch.pack(pady=5)
        
        # Output Area
        self.txt_log = tk.Text(frame, height=10, width=50, state='disabled')
        self.txt_log.pack(pady=10)

    # ==========================================
    # TAB 2: REAL-TIME PREDICTION (Requirement 4b)
    # ==========================================
    def setup_realtime_tab(self):
        frame = ttk.LabelFrame(self.tab_realtime, text="Single Trip Check", padding=20)
        frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Input Fields Grid
        grid_frame = ttk.Frame(frame)
        grid_frame.pack(pady=10)
        
        # We need inputs for the key features we engineered in ETL
        # For the mockup, we'll ask for the raw values that generate them
        inputs = [
            ("Max Speed (m/s):", "entry_speed"),
            ("Max Accel Mag:", "entry_accel"),
            ("Max Gyro Mag:", "entry_gyro"),
            ("Trip Duration (s):", "entry_duration")
        ]
        
        self.entries = {}
        
        for i, (label_text, key) in enumerate(inputs):
            lbl = ttk.Label(grid_frame, text=label_text)
            lbl.grid(row=i, column=0, padx=10, pady=5, sticky="e")
            
            ent = ttk.Entry(grid_frame)
            ent.grid(row=i, column=1, padx=10, pady=5)
            self.entries[key] = ent
            
        # Predict Button
        self.btn_predict = ttk.Button(frame, text="Analyze Safety", command=self.run_realtime_prediction)
        self.btn_predict.pack(pady=20)
        
        # Result Display
        self.lbl_result = ttk.Label(frame, text="Result: WAITING FOR INPUT", font=("Arial", 14, "bold"))
        self.lbl_result.pack(pady=10)

    # ==========================================
    # DUMMY LOGIC (To be replaced in Phase 3)
    # ==========================================
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.lbl_file_status.config(text=f"Loaded: {file_path.split('/')[-1]}", foreground="green")
            self.btn_run_batch.config(state="normal")
            self.log_message(f"File loaded successfully: {file_path}")

    def run_batch_prediction(self):
        # Placeholder for actual model inference
        self.log_message("Running AI model on batch data...")
        self.log_message("Processing 150 trips...")
        self.log_message("Done! Saved results to 'predictions.csv'")
        messagebox.showinfo("Success", "Batch processing complete! 12 Dangerous drivers found.")

    def run_realtime_prediction(self):
        # Placeholder logic
        try:
            speed = float(self.entries["entry_speed"].get())
            if speed > 22.2: # Simple logic for mockup (Speed > 80km/h)
                self.lbl_result.config(text="Result: DANGEROUS ⚠️", foreground="red")
            else:
                self.lbl_result.config(text="Result: SAFE DRIVER ✅", foreground="green")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")

    def log_message(self, msg):
        self.txt_log.config(state='normal')
        self.txt_log.insert(tk.END, ">> " + msg + "\n")
        self.txt_log.config(state='disabled')

# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = SafetyApp(root)
    root.mainloop()