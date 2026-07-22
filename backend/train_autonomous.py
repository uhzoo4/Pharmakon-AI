import sys
import time
import subprocess
import os
from pathlib import Path

def main():
    print("==================================================")
    print("🛡️  PHARMAKON AUTONOMOUS SUPERVISOR ONLINE 🛡️")
    print("==================================================")
    print("Monitoring training loop for memory leaks, crashes, and NaNs.")
    
    script_path = Path(__file__).parent / "train_world.py"
    state_file = Path(__file__).parent / "weights" / "checkpoints" / "training_state.json"
    
    restart_count = 0
    
    while True:
        cmd = [sys.executable, str(script_path), "--worker"]
        if state_file.exists():
            cmd.extend(["--resume_from", str(state_file)])
            
        print(f"\n[Supervisor] Launching worker (Restart #{restart_count})...")
        
        # Start the worker process
        process = subprocess.Popen(cmd)
        
        try:
            # Wait for the process to finish
            process.wait()
        except KeyboardInterrupt:
            print("\n[Supervisor] Received interrupt. Terminating worker...")
            process.terminate()
            process.wait()
            print("[Supervisor] Exiting gracefully.")
            break
            
        if process.returncode == 0:
            print("\n[Supervisor] Training completed successfully!")
            break
        elif process.returncode == 88:
            print("\n🚨 [Supervisor] WARNING: Worker triggered SELF-KILL (Exit 88) due to NaN/Inf!")
            print("[Supervisor] Memory leaked context has been flushed by OS.")
            print("[Supervisor] Hot-swapping to last known good JSON checkpoint...")
            time.sleep(2)
            restart_count += 1
        else:
            print(f"\n❌ [Supervisor] ERROR: Worker crashed unexpectedly with exit code {process.returncode} (possible OOM).")
            print("[Supervisor] Memory leaked context has been flushed by OS.")
            print("[Supervisor] Rebooting from last known good JSON checkpoint...")
            time.sleep(2)
            restart_count += 1

if __name__ == "__main__":
    main()
