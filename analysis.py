import pandas as pd
import numpy as np

class AdaptivePushupAnalyzer:
    def __init__(self, name):
        self.name = name
        
        # Global boundaries (used only for initial calibration)
        self.global_max = -float('inf')
        self.global_min = float('inf')
        
        # Adaptive tracking variables
        self.is_calibrated = False
        self.avg_amp = 0.0          # Moving average of the rep amplitudes
        self.last_max_val = None    # The height of the last confirmed mountain
        self.last_min_val = None    # The depth of the last confirmed valley
        
        # State machine variables
        self.state = 'CALIBRATING'
        self.local_extreme_val = None
        self.local_extreme_time = None
        
        self.peaks = []
        self.rep_count = 0

    def process_point(self, current_time, current_val):
        # 1. Track global limits
        if current_val > self.global_max: self.global_max = current_val
        if current_val < self.global_min: self.global_min = current_val
        
        # Before any reps are confirmed, base the amplitude on the total range seen so far
        if not self.is_calibrated:
            self.avg_amp = self.global_max - self.global_min
            
        # The absolute minimum threshold to prevent tracking micro-jitters (10% of average rep)
        noise_floor = 0.10 * self.avg_amp

        # 2. CALIBRATION PHASE
        if self.state == 'CALIBRATING':
            if self.avg_amp > 0: 
                calib_threshold = 0.30 * self.avg_amp
                
                if current_val < self.global_max - calib_threshold:
                    self.state = 'LOOKING_FOR_MIN'
                    self.local_extreme_val = current_val
                    self.local_extreme_time = current_time
                    self.last_max_val = self.global_max # Anchor the top
                    self.is_calibrated = True
                    print(f"[{self.name}] {current_time:>5.2f}s | Calibrated: Started HIGH. Tracking down.")
                
                elif current_val > self.global_min + calib_threshold:
                    self.state = 'LOOKING_FOR_MAX'
                    self.local_extreme_val = current_val
                    self.local_extreme_time = current_time
                    self.last_min_val = self.global_min # Anchor the bottom
                    self.is_calibrated = True
                    print(f"[{self.name}] {current_time:>5.2f}s | Calibrated: Started LOW. Tracking up.")

        # 3. LOOKING FOR A MAXIMUM (Upward phase)
        elif self.state == 'LOOKING_FOR_MAX':
            if current_val > self.local_extreme_val:
                # Still going up, keep pushing the ceiling higher
                self.local_extreme_val = current_val
                self.local_extreme_time = current_time
            else:
                # Data is dropping. Has it dropped enough to confirm the peak?
                # The threshold dynamically scales to 30% of the CURRENT upward sweep
                current_ascent_amp = self.local_extreme_val - self.last_min_val
                dynamic_threshold = max(noise_floor, 0.30 * current_ascent_amp)
                
                if current_val < self.local_extreme_val - dynamic_threshold:
                    # Peak confirmed!
                    self._register_peak('Max', self.local_extreme_time, self.local_extreme_val)
                    
                    # Update moving average amplitude (70% old, 30% new)
                    self.avg_amp = (self.avg_amp * 0.7) + (current_ascent_amp * 0.3)
                    self.last_max_val = self.local_extreme_val
                    
                    # Switch directions
                    self.state = 'LOOKING_FOR_MIN'
                    self.local_extreme_val = current_val
                    self.local_extreme_time = current_time

        # 4. LOOKING FOR A MINIMUM (Downward phase)
        elif self.state == 'LOOKING_FOR_MIN':
            if current_val < self.local_extreme_val:
                # Still going down, keep pushing the floor lower
                self.local_extreme_val = current_val
                self.local_extreme_time = current_time
            else:
                # Data is rising. Has it risen enough to confirm the valley?
                # The threshold dynamically scales to 30% of the CURRENT downward sweep
                current_descent_amp = self.last_max_val - self.local_extreme_val
                dynamic_threshold = max(noise_floor, 0.30 * current_descent_amp)
                
                if current_val > self.local_extreme_val + dynamic_threshold:
                    # Valley confirmed!
                    self._register_peak('Min', self.local_extreme_time, self.local_extreme_val)
                    
                    # Update moving average amplitude
                    self.avg_amp = (self.avg_amp * 0.7) + (current_descent_amp * 0.3)
                    self.last_min_val = self.local_extreme_val
                    
                    # Switch directions
                    self.state = 'LOOKING_FOR_MAX'
                    self.local_extreme_val = current_val
                    self.local_extreme_time = current_time

    def _register_peak(self, peak_type, t, val):
        self.peaks.append({'type': peak_type, 'time': t, 'value': val})
        print(f"[{self.name}] {t:>5.2f}s | Confirmed {peak_type:<3} | Amplitude: {val:.4f}")
        
        # Calculate Period on the fly
        same_type_peaks = [p for p in self.peaks if p['type'] == peak_type]
        if len(same_type_peaks) >= 2:
            period = same_type_peaks[-1]['time'] - same_type_peaks[-2]['time']
            
            # Ensure we only log rep periods based on the starting position's phase
            if peak_type == self.peaks[0]['type']: 
                self.rep_count += 1
                print(f"Rep {self.rep_count} completed! Period: {period:.2f} seconds | Local Adaptive Range: {self.avg_amp:.4f}\n")


# --- EXECUTION ---
def stream_csv_data(csv_filepath):
    df = pd.read_csv(csv_filepath)
    
    shoulder_analyzer = AdaptivePushupAnalyzer("Shoulder")
    #waist_analyzer = AdaptivePushupAnalyzer("Waist")
        
    for index, row in df.iterrows():
        t = row['Time(s)']
        shoulder_analyzer.process_point(t, row['Shoulder'])
        #waist_analyzer.process_point(t, row['Waist'])

if __name__ == "__main__":
    stream_csv_data('data/pushup2.csv')