import h5py
import ast
import numpy as np

hdf5_path = "/Users/jep/Personal Projects/LockedIn_FindMyForce/data/training_data.hdf5"

with h5py.File(hdf5_path, "r") as f:
    keys = list(f.keys())
    print(f"Total keys: {len(keys)}")
    print(f"Sample keys: {keys[:5]}")
    
    first_key = keys[0]
    data = f[first_key][()]
    print(f"Data shape for {first_key}: {data.shape}")
    
    # Analyze instance distribution
    instances = {}
    for key in keys:
        try:
            t = ast.literal_eval(key)
            # (mod, label, snr, idx)
            instance_id = t[:3] 
            instances[instance_id] = instances.get(instance_id, 0) + 1
        except:
            continue
            
    print(f"Total unique instances (mod, label, snr): {len(instances)}")
    counts = list(instances.values())
    print(f"Samples per instance: min={min(counts)}, max={max(counts)}, mean={np.mean(counts)}")
