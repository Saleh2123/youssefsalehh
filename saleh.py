import os
import torch
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from torch import nn
from networks.dcase2023t2_ae.network import AENet  # Ensure this path is correct
import librosa

class AnomalyDetectorGUI:
    def __init__(self, root, model, csv_path, threshold=0.9):
        self.root = root
        self.model = model
        self.csv_path = csv_path
        self.threshold = threshold
        self.model.eval()

        self.root.title("Anomaly Detection GUI")

        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title label
        self.label = ttk.Label(main_frame, text="Load a data file or directory to check for anomalies")
        self.label.grid(row=0, column=0, columnspan=2, pady=10)

        # Load file button
        self.load_file_button = ttk.Button(main_frame, text="Load Data File", command=self.load_file)
        self.load_file_button.grid(row=1, column=0, pady=5, sticky=tk.W)

        # Load directory button
        self.load_dir_button = ttk.Button(main_frame, text="Load Data Directory", command=self.load_directory)
        self.load_dir_button.grid(row=1, column=1, pady=5, sticky=tk.E)

        # Result text with scrollbar
        self.result_frame = ttk.Frame(main_frame)
        self.result_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.result_text = tk.Text(self.result_frame, height=15, width=60, wrap='word', borderwidth=2, relief='groove')
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.result_frame, orient="vertical", command=self.result_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")

        self.result_text.config(yscrollcommand=self.scrollbar.set)

    def load_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                # Extract filename to find in the CSV
                filename = os.path.basename(file_path)
                label = self.get_label_from_csv(filename)
                if label is None:
                    raise ValueError("Label not found in CSV file.")
                result = self.compare_with_threshold(label)
                self.display_results([(filename, label, result)])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process the file: {str(e)}")

    def load_directory(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            try:
                results = []
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        if file.endswith(('.wav', '.npy')):
                            filename = os.path.basename(file)
                            print(f"Processing file: {filename}")  # Debugging statement
                            label = self.get_label_from_csv(filename)
                            if label is not None:
                                result = self.compare_with_threshold(label)
                                results.append((filename, label, result))
                            else:
                                results.append((filename, None, "Label not found in CSV"))
                self.display_results(results)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process the directory: {str(e)}")

    def get_label_from_csv(self, filename):
        df = pd.read_csv(self.csv_path, header=None)
        for index, row in df.iterrows():
            if row[0] == filename:
                return float(row[1])
        return None

    def compare_with_threshold(self, label):
        if label > self.threshold:
            return "Anomaly"
        else:
            return "Normal"

    def display_results(self, results):
        self.result_text.delete(1.0, tk.END)
        for i, (filename, label, result) in enumerate(results):
            self.result_text.insert(tk.END, f"File: {filename}\n")
            self.result_text.insert(tk.END, f"Label: {label}\n")
            self.result_text.insert(tk.END, f"Result: {result}\n")
            self.result_text.insert(tk.END, "\n")

def load_model(model_path, input_dim, block_size):
    model = AENet(input_dim=input_dim, block_size=block_size)
    try:
        state_dict = torch.load(model_path)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
    return model

if __name__ == "__main__":
    model_path = "/Users/haniasaleh/Desktop/dcase/dcase2023_task2_baseline_ae/models/saved_model/baseline/DCASE2023T2-AE_DCASE2023T2gearbox_id(0_)_seed13711.pth"
    csv_path = "/Users/haniasaleh/Desktop/dcase/dcase2023_task2_baseline_ae/results/dev_data/baseline_MSE/decision_result_DCASE2023T2gearbox_section_00_test_seed13711_id(0_).csv"
    input_dim = 128
    block_size = 64
    threshold = 0.9

    try:
        model = load_model(model_path, input_dim, block_size)
    except RuntimeError as e:
        print(e)
        exit(1)

    root = tk.Tk()
    app = AnomalyDetectorGUI(root, model, csv_path, threshold)
    root.mainloop()
