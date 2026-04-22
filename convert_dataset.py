import numpy as np
import json
import os
from collections import defaultdict

def main():
    print("Reading config.json...")
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    num_mon_sites = config['num_mon_sites']
    num_mon_inst = config['num_mon_inst_train'] + config['num_mon_inst_test']
    num_unmon_sites = config['num_unmon_sites_train'] + config['num_unmon_sites_test']
    data_dir = config['data_dir']
    
    print(f"Target: {num_mon_sites} monitored sites, {num_mon_inst} instances each.")
    print(f"Target: {num_unmon_sites} unmonitored sites.")
    
    mon_file = os.path.join(data_dir, 'tor_200w_2500tr.npz')
    unmon_file = os.path.join(data_dir, 'tor_open_400000w.npz')
    
    # 1. Monitored Data
    print(f"Loading {mon_file}...")
    mon_data = np.load(mon_file, allow_pickle=True)
    m_data = mon_data['data']
    m_labels = mon_data['labels']
    
    print("Filtering monitored data...")
    site_counts = {}
    selected_sites = set()
    
    out_mon_dir = []
    out_mon_labels = []
    
    for i in range(len(m_labels)):
        site = m_labels[i]
        
        # Select site if we haven't reached the limit
        if site not in selected_sites:
            if len(selected_sites) >= num_mon_sites:
                continue
            selected_sites.add(site)
            
        # Initialize count if new site
        if site not in site_counts:
            site_counts[site] = 0
            
        # Collect instance if we need more for this site
        if site_counts[site] < num_mon_inst:
            out_mon_dir.append(m_data[i])
            out_mon_labels.append(site)
            site_counts[site] += 1
            
        # Break early if all done
        if len(selected_sites) == num_mon_sites and all(count == num_mon_inst for count in site_counts.values()):
            break
            
    print(f"Extracted {len(out_mon_dir)} monitored instances.")
    
    # Convert to appropriate shapes and dummy targets for missing features
    dir_seq_mon = np.array(out_mon_dir, dtype=np.int8)
    labels_mon = np.array(out_mon_labels)
    time_seq_mon = np.zeros_like(dir_seq_mon, dtype=np.float32)
    metadata_mon = np.zeros((len(out_mon_dir), 7), dtype=np.float32)
    
    out_mon_path = os.path.join(data_dir, 'all_closed_world.npz')
    print(f"Saving to {out_mon_path}...")
    np.savez_compressed(out_mon_path, dir_seq=dir_seq_mon, time_seq=time_seq_mon, 
                        metadata=metadata_mon, labels=labels_mon)
    
    # Free memory
    del mon_data, m_data, m_labels, dir_seq_mon, time_seq_mon, metadata_mon, labels_mon
    
    # 2. Unmonitored Data
    print(f"Loading {unmon_file}...")
    unmon_data = np.load(unmon_file, allow_pickle=True)
    u_data = unmon_data['data']
    u_labels = unmon_data['labels']
    
    print("Filtering unmonitored data...")
    out_unmon_dir = []
    out_unmon_labels = []
    
    for i in range(min(num_unmon_sites, len(u_labels))):
        out_unmon_dir.append(u_data[i])
        out_unmon_labels.append(u_labels[i])
        
    print(f"Extracted {len(out_unmon_dir)} unmonitored instances.")
    
    dir_seq_unmon = np.array(out_unmon_dir, dtype=np.int8)
    labels_unmon = np.array(out_unmon_labels)
    time_seq_unmon = np.zeros_like(dir_seq_unmon, dtype=np.float32)
    metadata_unmon = np.zeros((len(out_unmon_dir), 7), dtype=np.float32)
    
    out_unmon_path = os.path.join(data_dir, 'all_open_world.npz')
    print(f"Saving to {out_unmon_path}...")
    np.savez_compressed(out_unmon_path, dir_seq=dir_seq_unmon, time_seq=time_seq_unmon, 
                        metadata=metadata_unmon, labels=labels_unmon)
    
    print("Done! Data has been formatted for Var-CNN.")

if __name__ == '__main__':
    main()
