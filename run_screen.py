import subprocess
import sys

def launch_app_with_restart():
    # --- Configuration ---
    session_name = "main_async_llm_bielik"
    # Use absolute paths for reliability
    venv_activate = "/dane/eutest/.venv/bin/activate"
    script_to_run = "/dane/eutest/main_async_llm_bielik.py"
    
    # --- The Restart Logic (Shell Loop) ---
    # This is the core change: 
    # 1. 'while true; do ... done' creates an infinite loop.
    # 2. '$?' holds the exit code of the last command (python).
    # 3. 'sleep 5' waits 5 seconds before trying again.
    
    restart_loop_command = f"""
    source {venv_activate}
    echo '--- Starting Application Loop ---'
    
    while true; do
        echo "Starting process at $(date)"
        
        # Execute the python script
        python {script_to_run}
        
        EXIT_CODE=$?
        echo "Process exited with code $EXIT_CODE."
        
        # Check if exit code is 0 (normal exit)
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Process finished normally. Stopping restart loop."
            break # Exit the while loop
        fi
        
        # If exit code is non-zero (crash/error)
        echo "Restarting in 5 seconds..."
        sleep 5
    done
    
    exec bash # Keep the screen open after the loop exits
    """

    # --- Execute Screen ---
    try:
        print(f"Starting screen session: {session_name}")
        # Execute screen with the entire multi-line restart command
        subprocess.run([
            "screen", 
            "-dmS", session_name,
            "bash", "-c", restart_loop_command
        ], check=True, text=True) # text=True ensures proper handling of the multiline command
        
        print(f"\n✅ Launched '{session_name}' with auto-restart functionality.")
        print("To view output and crashes: screen -r " + session_name)

    except subprocess.CalledProcessError as e:
        print(f"🚨 Failed to start screen.", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    launch_app_with_restart()