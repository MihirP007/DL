
import random
import xml.etree.ElementTree as ET
import argparse # To easily change parameters from the command line

# --- Configuration ---
# These are default values, can be overridden by command-line arguments.

# --- IMPORTANT: Replace these with ACTUAL edge IDs from your .net.xml file ---
DEFAULT_ORIGIN_EDGES = ["edge_N_in", "edge_S_in", "edge_E_in", "edge_W_in"]
DEFAULT_DESTINATION_EDGES = ["edge_N_out", "edge_S_out", "edge_E_out", "edge_W_out"]

DEFAULT_NUM_VEHICLES = 1000     # Increased Total number of vehicles
DEFAULT_SIM_DURATION = 1800   # Duration in seconds over which vehicles depart (30 min)
DEFAULT_OUTPUT_FILE = "random_traffic.rou.xml"

# Define Vehicle Types and their probabilities
VEHICLE_TYPES = {
    "CAR": {
        "params": {
            "accel": "2.9", "decel": "4.5", "sigma": "0.5", "length": "5.0",
            "minGap": "2.5", "maxSpeed": "13.89", "guiShape": "passenger", "width": "1.8",
            "vClass": "passenger"
        },
        "probability": 0.60 # 70% Cars
    },
    "TRUCK": {
        "params": {
            "accel": "1.5", "decel": "3.5", "sigma": "0.5", "length": "12.0",
            "minGap": "3.0", "maxSpeed": "11.11", "guiShape": "truck", "width": "2.5", # Slower speed (40km/h)
            "vClass": "truck"
        },
        "probability": 0.15 # 15% Trucks
    },
    "MOTORCYCLE": {
        "params": {
            "accel": "3.5", "decel": "5.0", "sigma": "0.5", "length": "2.5",
            "minGap": "1.5", "maxSpeed": "16.67", "guiShape": "motorcycle", "width": "0.8", # Faster speed (60km/h), NARROWER
            "vClass": "motorcycle"
        },
        "probability": 0.25 # 15% Motorcycles
    }
}

def generate_random_routes(num_vehicles, sim_duration, output_file, origin_edges, destination_edges, vehicle_types_config):
    """
    Generates a SUMO route file (.rou.xml) with vehicles of different types assigned random trips
    between specified origin and destination edges, sorted by departure time.

    Args:
        num_vehicles (int): Total number of vehicles to generate.
        sim_duration (int): Time in seconds over which vehicles will depart.
        output_file (str): Path to the output .rou.xml file.
        origin_edges (list): List of possible starting edge IDs.
        destination_edges (list): List of possible ending edge IDs.
        vehicle_types_config (dict): Configuration dictionary for vehicle types and probabilities.
    """
    # Basic input validation
    if not origin_edges or not destination_edges:
        print("Error: Origin and destination edge lists cannot be empty.")
        print("Please provide actual edge IDs using --origins and --destinations arguments.")
        return
    if num_vehicles <= 0 or sim_duration <= 0:
        print("Error: Number of vehicles and simulation duration must be positive.")
        return
    # Validate probabilities sum to 1
    total_prob = sum(v['probability'] for v in vehicle_types_config.values())
    if not abs(total_prob - 1.0) < 1e-6: # Check for floating point equality
         print(f"Error: Vehicle type probabilities must sum to 1.0 (current sum: {total_prob})")
         return

    print(f"Generating {num_vehicles} vehicles with types:")
    for vtype_id, config in vehicle_types_config.items():
        print(f"- {vtype_id}: {config['probability']*100:.1f}%")
    print(f"Possible origins: {origin_edges}")
    print(f"Possible destinations: {destination_edges}")
    print(f"Departures spread over {sim_duration} seconds.")

    # Prepare lists for weighted random choice
    vtype_ids = list(vehicle_types_config.keys())
    probabilities = [vehicle_types_config[vtid]['probability'] for vtid in vtype_ids]

    # --- Store generated trip data temporarily ---
    vehicle_trips_data = []

    # Generate each vehicle's trip data
    for i in range(num_vehicles):
        veh_id = f"veh_{i}"
        # Random departure time (store as integer for sorting)
        depart_time_int = random.randint(0, sim_duration - 1)

        # Choose random origin and destination
        origin_edge = random.choice(origin_edges)
        dest_edge = random.choice(destination_edges)

        # Ensure origin and destination are different, if possible
        attempts = 0
        max_attempts = len(origin_edges) * len(destination_edges)
        # Don't allow direct U-turns from in-edge to out-edge of the same direction
        # e.g. edge_N_in -> edge_N_out
        origin_direction = origin_edge.split('_')[1] # N, S, E, W
        destination_direction = dest_edge.split('_')[1] # N, S, E, W

        while (origin_direction == destination_direction or origin_edge == dest_edge) and len(set(destination_edges)) > 1 and attempts < max_attempts:
            dest_edge = random.choice(destination_edges)
            destination_direction = dest_edge.split('_')[1]
            attempts += 1

        if origin_direction == destination_direction and len(set(destination_edges)) > 1 and attempts == max_attempts:
             print(f"Warning: Could not easily find a non-U-turn destination from '{origin_edge}' for {veh_id}. Allowing trip.")

        # Choose vehicle type based on probability
        chosen_vtype_id = random.choices(vtype_ids, weights=probabilities, k=1)[0]

        # Store trip details in the list
        vehicle_trips_data.append({
            'id': veh_id,
            'type': chosen_vtype_id,
            'depart': depart_time_int, # Store as integer
            'from': origin_edge,
            'to': dest_edge
        })

    # --- Sort the vehicle trip data by departure time ---
    print("Sorting trips by departure time...")
    vehicle_trips_data.sort(key=lambda trip: trip['depart'])

    # --- Build the XML structure ---
    print("Building XML structure...")
    routes = ET.Element("routes")
    routes.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    routes.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")

    # Define the vehicle types
    for vtype_id, config in vehicle_types_config.items():
        ET.SubElement(routes, "vType", id=vtype_id, **config['params']) # Use ** to unpack the params dict

    generated_count = 0
    # Add the SORTED <trip> elements to the <routes>
    for trip_data in vehicle_trips_data:
        # Convert depart time back to string for XML attribute
        trip_attributes = {
            'id': trip_data['id'],
            'type': trip_data['type'],
            'depart': str(trip_data['depart']), # Convert back to string
            'from': trip_data['from'],
            'to': trip_data['to']
        }
        ET.SubElement(routes, "trip", attrib=trip_attributes)
        generated_count += 1

    # --- Write the XML file ---
    tree = ET.ElementTree(routes)
    # Pretty print the XML (requires Python 3.9+)
    try:
        ET.indent(tree, space="\t", level=0)
    except AttributeError:
        print("Warning: XML indentation not available (requires Python 3.9+). Output will be compact.")
        pass # Fallback for older Python versions (no indentation)
    except TypeError:
         print("Warning: XML indentation failed (possibly due to ElementTree version). Output will be compact.")
         pass

    try:
        tree.write(output_file, encoding="UTF-8", xml_declaration=True)
        print(f"\nSuccessfully generated and SORTED {generated_count} vehicle trips in '{output_file}'")
    except IOError as e:
        print(f"\nError writing file '{output_file}': {e}")


# --- Main execution block ---
if __name__ == "__main__":
    # Set up argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Generate a sorted SUMO route file (.rou.xml) with diverse vehicle types using <trip> elements.")

    parser.add_argument("-n", "--num-vehicles", type=int, default=DEFAULT_NUM_VEHICLES,
                        help=f"Total number of vehicles to generate (default: {DEFAULT_NUM_VEHICLES})")
    parser.add_argument("-d", "--duration", type=int, default=DEFAULT_SIM_DURATION,
                        help=f"Simulation duration in seconds for spreading departures (default: {DEFAULT_SIM_DURATION})")
    parser.add_argument("-o", "--output", type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f"Output route file name (default: {DEFAULT_OUTPUT_FILE})")
    # We don't need a single --vtype anymore, as we define multiple types
    parser.add_argument("--origins", nargs='+', required=True,
                        help="Space-separated list of possible origin edge IDs (required). Example: --origins edge_N_in edge_S_in")
    parser.add_argument("--destinations", nargs='+', required=True,
                        help="Space-separated list of possible destination edge IDs (required). Example: --destinations edge_N_out edge_S_out")

    # Parse arguments from command line
    args = parser.parse_args()

    # Run the generation function with parsed arguments
    generate_random_routes(
        num_vehicles=args.num_vehicles,
        sim_duration=args.duration,
        output_file=args.output,
        origin_edges=args.origins,
        destination_edges=args.destinations,
        vehicle_types_config=VEHICLE_TYPES # Pass the whole config dict
    )
