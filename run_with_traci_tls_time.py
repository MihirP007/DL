import os
import sys
import traci
import time

# --- Configuration ---
SUMO_BINARY = "sumo-gui"
CONFIG_FILE = "intersection.sumocfg"
TLS_ID = "center"
POI_ID = "tls_time_display_minimal" # Different ID
POI_TYPE = "tls_info_minimal"
TEXT_Y_OFFSET = 30        # Even further offset
TEXT_COLOR = (0, 255, 0, 255) # Changed color to GREEN for visibility

# --- Check for SUMO_HOME ---
# (Keep the SUMO_HOME check as before)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# --- SUMO Command ---
sumoCmd = [SUMO_BINARY, "-c", CONFIG_FILE, "--start"]

# --- Main Simulation Loop ---
print("Starting SUMO with TraCI...")
traci.start(sumoCmd)

print(f"Getting position for TLS: {TLS_ID}")
poi_created_successfully = False
try:
    posX, posY = traci.junction.getPosition(TLS_ID)
    poi_pos_x = posX
    poi_pos_y = posY + TEXT_Y_OFFSET
    print(f"TLS Position: ({posX}, {posY}), POI Position: ({poi_pos_x}, {poi_pos_y})")

    # --- Create Minimal POI ---
    print(f"Attempting to create MINIMAL POI '{POI_ID}'...")
    traci.poi.add(
        poiID=POI_ID,
        x=poi_pos_x,
        y=poi_pos_y,
        color=TEXT_COLOR # Just ID, position, color
    )
    # Do not set type or other parameters initially
    print(f"Successfully ADDED minimal POI '{POI_ID}'. Check GUI.")
    poi_created_successfully = True

except traci.TraCIException as e:
     print(f"ERROR setting up minimal POI {POI_ID}: {e}")

step = 0
max_steps = 500
while step < max_steps:
    try:
        if traci.simulation.getMinExpectedNumber() <= 0: break
        traci.simulationStep()
    except traci.TraCIException as e:
        print(f"TraCI Error during simulation step {step}: {e}")
        break
    if step % 100 == 0: print(f"Simulation Step: {step}")
    step += 1

if poi_created_successfully:
    print(f"Simulation finished {max_steps} steps. Check GUI for a GREEN dot/square north of the intersection center.")
else:
    print(f"Simulation finished {max_steps} steps, but POI creation failed.")

print("Closing TraCI connection.")
traci.close()
print("Script finished. SUMO GUI might remain open.")