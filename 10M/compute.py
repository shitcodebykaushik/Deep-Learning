import pynvml
import time
import csv
import os
from datetime import datetime

def monitor_gpu(interval=5, log_file="gpu_comprehensive_log.csv"):
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"Failed to initialize NVML: {e}")
        return

    temps = []
    utils = []

    print(f"--- Pro GPU Monitor Active ---")
    print(f"Logging to: {log_file}")
    
    try:
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if os.stat(log_file).st_size == 0:
                writer.writerow([
                    "Timestamp", "Util (%)", "Mem_Used (MB)", "Mem_Total (MB)", 
                    "Temp (C)", "Clock (MHz)", "Power (W)", "Throttle_Status"
                ])

            while True:
                # 1. Core Metrics (Highly Reliable)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                
                # 2. Power Usage (Can fail on some laptop drivers)
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except:
                    power = 0.0
                
                # 3. Throttling Check (The fix for your AttributeError)
                # We use the bitmask for Thermal Throttling directly (0x0000000000000001)
                try:
                    reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                    # Checking if the 1st bit (Thermal) or 8th bit (Power) is active
                    is_thermal = bool(reasons & 0x0000000000000001)
                    is_power_limit = bool(reasons & 0x0000000000000080)
                    
                    if is_thermal: throttle_status = "THERMAL"
                    elif is_power_limit: throttle_status = "POWER_LIMIT"
                    else: throttle_status = "CLEAR"
                except:
                    throttle_status = "UNSUPPORTED"

                timestamp_full = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                timestamp_short = datetime.now().strftime("%H:%M:%S")
                
                temps.append(temp)
                utils.append(util.gpu)

                # Live Display
                print(f"[{timestamp_short}] {util.gpu}% | {temp}C | {clock}MHz | {power:.1f}W | {throttle_status}    ", end="\r")

                # Write to Log
                writer.writerow([
                    timestamp_full, util.gpu, round(mem.used/1024**2, 2), 
                    round(mem.total/1024**2, 2), temp, clock, round(power, 2), throttle_status
                ])
                file.flush()
                
                time.sleep(interval)

    except KeyboardInterrupt:
        if temps:
            print(f"\n\n--- Session Summary ---")
            print(f"Avg Temp: {sum(temps)/len(temps):.1f}°C (Max: {max(temps)}°C)")
            print(f"Avg Load: {sum(utils)/len(utils):.1f}%")
        print("Monitoring ended safely.")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

if __name__ == "__main__":
    monitor_gpu(interval=10)