import pandas as pd
import numpy as np

class AdaptivePushupAnalyzer:
    def __init__(self, name='Push Up Analyser'):
        self.name = 'Push Up Analyser'
        # Global boundaries for Angle (Arms)
        self.global_max_angle = -float('inf')
        self.global_min_angle = float('inf')
        
        # Global boundaries for Shoulder (Body)
        self.global_max_shoulder = -float('inf')
        self.global_min_shoulder = float('inf')
        
        # Adaptive tracking variables
        self.is_calibrated = False
        self.avg_amp_angle = 0.0
        self.avg_amp_shoulder = 0.0
        
        self.last_max_angle = None
        self.last_min_angle = None
        self.last_max_shoulder = None
        self.last_min_shoulder = None
        
        # State machine variables
        self.state = 'CALIBRATING'
        self.local_extreme_angle = None
        self.local_extreme_shoulder = None
        self.local_extreme_time = None
        
        self.peaks = []
        self.rep_count = 0

    def process_point(self, current_time, right, left, shoulder):
        # 1. Process Angle Extremes (Highest max or lowest min between both arms)
        val_max_angle = max(right, left)
        val_min_angle = min(right, left)
        
        # 2. Track Shoulder
        val_shoulder = shoulder
        
        # Track global limits
        if val_max_angle > self.global_max_angle: self.global_max_angle = val_max_angle
        if val_min_angle < self.global_min_angle: self.global_min_angle = val_min_angle
        
        if val_shoulder > self.global_max_shoulder: self.global_max_shoulder = val_shoulder
        if val_shoulder < self.global_min_shoulder: self.global_min_shoulder = val_shoulder
        
        # Before calibration, base the amplitude on the total range seen so far
        if not self.is_calibrated:
            self.avg_amp_angle = self.global_max_angle - self.global_min_angle
            self.avg_amp_shoulder = self.global_max_shoulder - self.global_min_shoulder
            
        # Noise floors (20% of average rep)
        noise_floor_angle = 0.20 * self.avg_amp_angle
        noise_floor_shoulder = 0.20 * self.avg_amp_shoulder

        # 3. CALIBRATION PHASE
        if self.state == 'CALIBRATING':
            if self.avg_amp_angle > 0 and self.avg_amp_shoulder > 0:
                calib_thresh_angle = 0.30 * self.avg_amp_angle
                calib_thresh_shoulder = 0.30 * self.avg_amp_shoulder
                
                # BOTH requirements must start moving down
                if (val_min_angle < self.global_max_angle - calib_thresh_angle and 
                    val_shoulder < self.global_max_shoulder - calib_thresh_shoulder):
                    
                    self.state = 'LOOKING_FOR_MIN'
                    self.local_extreme_angle = val_min_angle
                    self.local_extreme_shoulder = val_shoulder
                    self.local_extreme_time = current_time
                    self.last_max_angle = self.global_max_angle
                    self.last_max_shoulder = self.global_max_shoulder
                    self.is_calibrated = True
                    print(f"[{self.name}] {current_time:>5.2f}s | Calibrated: Started HIGH. Tracking down.")
                
                # BOTH requirements must start moving up
                elif (val_max_angle > self.global_min_angle + calib_thresh_angle and 
                      val_shoulder > self.global_min_shoulder + calib_thresh_shoulder):
                    
                    self.state = 'LOOKING_FOR_MAX'
                    self.local_extreme_angle = val_max_angle
                    self.local_extreme_shoulder = val_shoulder
                    self.local_extreme_time = current_time
                    self.last_min_angle = self.global_min_angle
                    self.last_min_shoulder = self.global_min_shoulder
                    self.is_calibrated = True
                    print(f"[{self.name}] {current_time:>5.2f}s | Calibrated: Started LOW. Tracking up.")

        # 4. LOOKING FOR A MAXIMUM (Upward phase)
        elif self.state == 'LOOKING_FOR_MAX':
            # Push ceilings higher if still ascending
            if val_max_angle > self.local_extreme_angle:
                self.local_extreme_angle = val_max_angle
                self.local_extreme_time = current_time
            if val_shoulder > self.local_extreme_shoulder:
                self.local_extreme_shoulder = val_shoulder
                self.local_extreme_time = current_time
                
            # Have they dropped enough to confirm the peak?
            current_ascent_amp_angle = self.local_extreme_angle - self.last_min_angle
            current_ascent_amp_shoulder = self.local_extreme_shoulder - self.last_min_shoulder
            
            dyn_thresh_angle = max(noise_floor_angle, 0.30 * current_ascent_amp_angle)
            dyn_thresh_shoulder = max(noise_floor_shoulder, 0.30 * current_ascent_amp_shoulder)
            
            # Check if BOTH have started dropping
            angle_dropped = val_max_angle < self.local_extreme_angle - dyn_thresh_angle
            shoulder_dropped = val_shoulder < self.local_extreme_shoulder - dyn_thresh_shoulder
            
            # REQUIREMENT: Both must be fulfilled to transition states
            if angle_dropped and shoulder_dropped:
                if current_ascent_amp_angle <= 60:
                    print(f"[{self.name}] {current_time:>5.2f}s | Ignored Max | Angle Amplitude {current_ascent_amp_angle:.2f} <= 60")
                else:
                    self._register_peak('Max', self.local_extreme_time, self.local_extreme_angle, self.local_extreme_shoulder)
                    
                    # Update moving averages independently
                    self.avg_amp_angle = (self.avg_amp_angle * 0.7) + (current_ascent_amp_angle * 0.3)
                    self.avg_amp_shoulder = (self.avg_amp_shoulder * 0.7) + (current_ascent_amp_shoulder * 0.3)
                
                self.last_max_angle = self.local_extreme_angle
                self.last_max_shoulder = self.local_extreme_shoulder
                
                # Switch directions
                self.state = 'LOOKING_FOR_MIN'
                self.local_extreme_angle = val_min_angle
                self.local_extreme_shoulder = val_shoulder
                self.local_extreme_time = current_time

        # 5. LOOKING FOR A MINIMUM (Downward phase)
        elif self.state == 'LOOKING_FOR_MIN':
            # Push floors lower if still descending
            if val_min_angle < self.local_extreme_angle:
                self.local_extreme_angle = val_min_angle
                self.local_extreme_time = current_time
            if val_shoulder < self.local_extreme_shoulder:
                self.local_extreme_shoulder = val_shoulder
                self.local_extreme_time = current_time
                
            # Have they risen enough to confirm the valley?
            current_descent_amp_angle = self.last_max_angle - self.local_extreme_angle
            current_descent_amp_shoulder = self.last_max_shoulder - self.local_extreme_shoulder
            
            dyn_thresh_angle = max(noise_floor_angle, 0.30 * current_descent_amp_angle)
            dyn_thresh_shoulder = max(noise_floor_shoulder, 0.30 * current_descent_amp_shoulder)
            
            # Check if BOTH have started rising
            angle_risen = val_min_angle > self.local_extreme_angle + dyn_thresh_angle
            shoulder_risen = val_shoulder > self.local_extreme_shoulder + dyn_thresh_shoulder
            
            # REQUIREMENT: Both must be fulfilled to transition states
            if angle_risen and shoulder_risen:
                
                if current_descent_amp_angle <= 50:
                    print(f"[{self.name}] {current_time:>5.2f}s | Ignored Min | Angle Amplitude {current_descent_amp_angle:.2f} <= 50")
                else:
                    self._register_peak('Min', self.local_extreme_time, self.local_extreme_angle, self.local_extreme_shoulder)
                    
                    # Update moving averages independently
                    self.avg_amp_angle = (self.avg_amp_angle * 0.7) + (current_descent_amp_angle * 0.3)
                    self.avg_amp_shoulder = (self.avg_amp_shoulder * 0.7) + (current_descent_amp_shoulder * 0.3)
                    
                self.last_min_angle = self.local_extreme_angle
                self.last_min_shoulder = self.local_extreme_shoulder
                
                # Switch directions
                self.state = 'LOOKING_FOR_MAX'
                self.local_extreme_angle = val_max_angle
                self.local_extreme_shoulder = val_shoulder
                self.local_extreme_time = current_time

    def _register_peak(self, peak_type, t, val_angle, val_shoulder):
        self.peaks.append({
            'type': peak_type, 
            'time': t, 
            'angle': val_angle,
            'shoulder': val_shoulder
        })
        print(f"[{self.name}] {t:>5.2f}s | Confirmed {peak_type:<3} | Angle Ex.: {val_angle:.2f}, Shoulder Ex.: {val_shoulder:.2f}")
        
        # Calculate Period on the fly
        same_type_peaks = [p for p in self.peaks if p['type'] == peak_type]
        if len(same_type_peaks) >= 2:
            period = same_type_peaks[-1]['time'] - same_type_peaks[-2]['time']
            
            # Ensure we only log rep periods based on the starting position's phase
            if peak_type == self.peaks[0]['type']: 
                self.rep_count += 1
                print(f"-> Rep {self.rep_count} completed! Period: {period:.2f}s | Avg Angle Range: {self.avg_amp_angle:.2f} | Avg Shoulder Range: {self.avg_amp_shoulder:.2f}\n")


# --- EXECUTION ---
def stream_csv_data(angles_filepath, shoulder_filepath):
    print(f"\n--- Starting Full Body Pushup Analysis ---")
    
    df_angles = pd.read_csv(angles_filepath)
    df_shoulder = pd.read_csv(shoulder_filepath)
    df = pd.merge(df_angles, df_shoulder, on='Time(s)', how='inner')
    df = df.sort_values('Time(s)').reset_index(drop=True)
    
    analyzer = AdaptivePushupAnalyzer()
        
    # 3. Stream the combined data
    for index, row in df.iterrows():
        t = row['Time(s)']
        right = row['Right(deg)']
        left = row['Left(deg)']
        shoulder = row['Shoulder'] 
        
        # Evaluate all body metrics concurrently
        analyzer.process_point(t, right, left, shoulder)

    print(f"\n--- Final Statistics ---")
    print(f"Total Valid Reps (Arms + Shoulder): {analyzer.rep_count}")
    print(f"Absolute Angle Extremes: Max {analyzer.global_max_angle:.2f}, Min {analyzer.global_min_angle:.2f}")
    print(f"Absolute Shoulder Extremes: Max {analyzer.global_max_shoulder:.2f}, Min {analyzer.global_min_shoulder:.2f}\n")

if __name__ == "__main__":
    # Pass both file paths to the function
    stream_csv_data('data/angles_pushup1.csv', 'data/pushup1.csv')